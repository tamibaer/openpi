"""Offline-Test für pi0-FAST: Wie gut trifft das Modell die Action-Tokens?

Läuft auf den Trainingsdaten und misst nur die reinen FAST-Tokens (ohne "Action: " und "|").
Die Vorhersage ist teacher-forced: Jedes Token wird mit den korrekten Vorgängern vorhergesagt.
Das zeigt also, wie gut das Modell die Tokens gelernt hat, ohne Folgefehler.

Aufruf (im openpi-Ordner, Policy-Server vorher beenden):
    export HF_LEROBOT_HOME=~/ros2_ws/lerobot_datasets
    uv run examples/ur5e/eval_fast_tokens.py \
        --config pi0_fast_ur5e_lora_test_1 \
        --ckpt checkpoints/pi0_fast_ur5e_lora_test_1/<exp_name>/29999

Ausgabe:
  Token-CE      Cross-Entropy nur auf FAST-Tokens (kleiner = besser)
  Top-1/Top-5   Anteil der FAST-Tokens, bei denen das richtige Token vorne liegt
  Roundtrip     Fehler, wenn man die Original-Aktionen nur kodiert und wieder dekodiert (rad).
                Das ist die Untergrenze für den Modell-Fehler in eval_offline.py.

Den Lauf mit beiden Checkpoints (LoRA und LoRA + Embedder) wiederholen und vergleichen.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib

import numpy as np


def fast_token_mask(loss_mask: np.ndarray, n_prefix: int, n_suffix: int) -> np.ndarray:
    """Beschränkt die Loss-Maske auf die FAST-Tokens (ohne "Action: " davor und "|" + EOS danach)."""
    out = np.zeros_like(loss_mask, dtype=bool)
    for i, row in enumerate(loss_mask):
        idx = np.flatnonzero(row)
        if len(idx) > n_prefix + n_suffix:
            out[i, idx[n_prefix : len(idx) - n_suffix]] = True
    return out


def jit_with_frozen_state(model, fn):
    """Wie openpi.shared.nnx_utils.module_jit, aber für freie Funktionen.

    Nicht nnx.jit verwenden: Das gibt den Modellzustand zusätzlich als Ausgabe zurück und braucht
    dadurch mehrere GB mehr GPU-Speicher.
    """
    import flax.nnx as nnx
    import jax

    graphdef, state = nnx.split(model)

    def fun(state, *args):
        return fn(nnx.merge(graphdef, state), *args)

    jitted = jax.jit(fun)
    return lambda *args: jitted(state, *args)


def predict_tokens(model, obs):
    """Gibt für jedes Ziel-Token die Log-Wahrscheinlichkeit und die Top-5-Token-IDs zurück."""
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model
    from openpi.models import pi0_fast

    obs = _model.preprocess_observation(None, obs, train=False, image_keys=list(obs.images.keys()))
    embeddings, input_mask, ar_mask = model.embed_inputs(obs)
    attn_mask = pi0_fast.make_attn_mask(input_mask, ar_mask)
    targets = obs.tokenized_prompt[:, 1:]

    # Wie compute_loss: Jede Eingabe sagt das nächste Token vorher.
    pre_logits, _, _ = model.PaliGemma.llm(
        embedded_prefix=embeddings[:, :-1],
        mask=attn_mask[:, :-1, :-1],
        return_prelogits=True,
    )
    logits, _ = model.PaliGemma.llm(pre_logits=pre_logits[:, -targets.shape[1] :])
    logits = logits.astype(jnp.float32)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    _, top5 = jax.lax.top_k(logits, 5)
    return targets, target_log_probs, top5


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True, help="Checkpoint-Ordner mit params/")
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4, help="klein halten: Logits über das ganze Vokabular")
    args = p.parse_args()

    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model
    from openpi.models import tokenizer as _tokenizer
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader

    config = dataclasses.replace(_config.get_config(args.config), batch_size=args.batch_size)
    model_config = config.model

    print("Lade Modell ...")
    params = _model.restore_params(pathlib.Path(args.ckpt).resolve() / "params", dtype=jnp.bfloat16)
    model = model_config.load(params)
    del params
    predict = jit_with_frozen_state(model, predict_tokens)

    # Gleicher Seed und gleiche Reihenfolge, damit beide Checkpoints dieselben Frames sehen.
    loader = _data_loader.create_data_loader(config, shuffle=True, num_batches=args.batches)
    data_config = loader.data_config()
    action_stats = data_config.norm_stats["actions"]
    # Quantil-Normierung auf [-1, 1]: Faktor zurück in die Originaleinheit (rad, bzw. Greifer 0..1).
    to_original_units = (np.asarray(action_stats.q99) - np.asarray(action_stats.q01)) / 2

    tokenizer = _tokenizer.FASTTokenizer(model_config.max_token_len)
    n_prefix = len(tokenizer._paligemma_tokenizer.encode("Action: "))
    n_suffix = len(tokenizer._paligemma_tokenizer.encode("|", add_eos=True))

    n_tokens, sum_ce, sum_top1, sum_top5 = 0, 0.0, 0, 0
    roundtrip_errors, token_counts = [], []
    n_truncated = 0

    for obs, actions in loader:
        targets, target_log_probs, top5 = jax.device_get(predict(obs))
        loss_mask = np.asarray(jax.device_get(obs.token_loss_mask))[:, 1:]
        mask = fast_token_mask(loss_mask, n_prefix, n_suffix)

        n_tokens += int(mask.sum())
        sum_ce += float(-target_log_probs[mask].sum())
        sum_top1 += int((top5[..., 0] == targets)[mask].sum())
        sum_top5 += int((top5 == targets[..., None]).any(-1)[mask].sum())
        token_counts.extend(mask.sum(1).tolist())

        tokens = np.asarray(jax.device_get(obs.tokenized_prompt))
        token_mask = np.asarray(jax.device_get(obs.tokenized_prompt_mask))
        actions = np.asarray(jax.device_get(actions))
        for i in range(len(tokens)):
            # Ohne EOS am Ende wurde die Sequenz von max_token_len abgeschnitten.
            if tokens[i][token_mask[i]][-1] != tokenizer._paligemma_tokenizer.eos_id():
                n_truncated += 1
                continue
            decoded = tokenizer.extract_actions(
                tokens[i][token_mask[i]], model_config.action_horizon, model_config.action_dim
            )
            roundtrip_errors.append(np.abs(decoded - actions[i]).mean(0) * to_original_units)

    print(f"\nFrames: {len(token_counts)}, FAST-Tokens pro Frame im Schnitt: {np.mean(token_counts):.1f}")
    print(f"Token-CE: {sum_ce / n_tokens:.4f}")
    print(f"Top-1:    {sum_top1 / n_tokens:.1%}")
    print(f"Top-5:    {sum_top5 / n_tokens:.1%}")
    if n_truncated:
        print(f"WARNUNG: {n_truncated} Frames abgeschnitten, max_token_len erhöhen.")

    error = np.mean(roundtrip_errors, axis=0)
    joints = ["pan", "lift", "elbow", "w1", "w2", "w3"]
    print(f"\nRoundtrip-Fehler (nur Tokenizer, ohne Modell): {error[:6].mean():.4f} rad im Mittel")
    print("pro Gelenk: " + "  ".join(f"{name}={value:.4f}" for name, value in zip(joints, error[:6])))
    print(f"Greifer:    {error[6]:.4f}")


if __name__ == "__main__":
    main()