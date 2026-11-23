"""export_for_hf.py – openpi-Checkpoint (pi0-FAST) so exportieren, dass ihn auch LLM-Leute ohne JAX/openpi nutzen können.

Erzeugt einen Ordner zum Hochladen auf Hugging Face:
  model-0000X-of-0000N.safetensors   alle Gewichte (float32), Namen wie im openpi-Checkpoint,
  model.safetensors.index.json       z.B. "PaliGemma/llm/embedder/input_embedding"
  embeddings_finetuned.safetensors   nur die Token-Embedding-Matrix (257152 x 2048) des Fine-Tunes
  embeddings_base.safetensors        (optional, --base-params) dieselbe Matrix aus pi0_fast_base zum Vergleich
  openpi_checkpoint/                 Original-Checkpoint (params/ + assets/) für openpi serve_policy
  config.json, README.md             Architektur, Prompt-Format, FAST-Token-Abbildung, Trainingsdaten

Laden ohne openpi (PyTorch oder NumPy):
    from safetensors.numpy import load_file
    emb = load_file("embeddings_finetuned.safetensors")["input_embedding"]

Aufruf (in der openpi-Umgebung, dort wo der Checkpoint liegt):
    uv run python examples/ur5e/export_for_hf.py checkpoints/pi0_fast_ur5e_full_v2/v1/19999 \
        --out hf_export/pi0_fast_ur5e_full_v2 \
        --base-params ~/models/openpi-assets/checkpoints/pi0_fast_base/params
"""
import argparse
import json
import pathlib
import shutil

import numpy as np

EMBED_KEY = "PaliGemma/llm/embedder/input_embedding"
FAST_SKIP_TOKENS = 128  # openpi FASTTokenizer: letzte 128 PaliGemma-Tokens sind Sondertokens


def flatten(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = np.ascontiguousarray(np.asarray(v))
    return out


def save_sharded(flat, out, shard_bytes):
    from safetensors.numpy import save_file
    shards, cur, size = [], {}, 0
    for k in sorted(flat):
        nbytes = flat[k].nbytes
        if cur and size + nbytes > shard_bytes:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = flat[k]
        size += nbytes
    if cur:
        shards.append(cur)
    weight_map, total = {}, 0
    for i, shard in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(out / name), metadata={"format": "np"})
        for k, v in shard.items():
            weight_map[k] = name
            total += v.nbytes
        print(f"  {name}: {len(shard)} Tensoren")
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    return len(shards)


def load_params(path):
    import openpi.models.model as _model
    return _model.restore_params(pathlib.Path(path).expanduser(), restore_type=np.ndarray)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=pathlib.Path, help="openpi-Checkpoint-Ordner mit params/ und assets/")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--base-params", type=pathlib.Path, help="params/ von pi0_fast_base für die Vergleichs-Embeddings")
    ap.add_argument("--config-name", default="pi0_fast_ur5e_full_v2")
    ap.add_argument("--dataset", default="tamibaer/ur5e_pi0-FAST")
    ap.add_argument("--shard-gb", type=float, default=4.0)
    ap.add_argument("--no-orbax", action="store_true", help="Original-Checkpoint nicht mitkopieren")
    args = ap.parse_args()

    from safetensors.numpy import save_file

    ckpt = args.checkpoint.expanduser().resolve()
    out = args.out.expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} ist nicht leer – anderen --out wählen oder Ordner vorher löschen.")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Lade {ckpt / 'params'} ...")
    flat = flatten(load_params(ckpt / "params"))
    if EMBED_KEY not in flat:
        raise SystemExit(f"{EMBED_KEY} nicht gefunden. Vorhandene Schlüssel z.B.: {sorted(flat)[:5]}")
    n_params = sum(v.size for v in flat.values())
    print(f"{len(flat)} Tensoren, {n_params / 1e9:.2f} Mrd. Parameter")

    n_shards = save_sharded(flat, out, int(args.shard_gb * 1024**3))
    save_file({"input_embedding": flat[EMBED_KEY]}, str(out / "embeddings_finetuned.safetensors"))
    print("  embeddings_finetuned.safetensors")

    if args.base_params:
        print(f"Lade Basis-Embeddings aus {args.base_params} ...")
        base = flatten(load_params(args.base_params))
        base_emb = base[EMBED_KEY]
        save_file({"input_embedding": base_emb}, str(out / "embeddings_base.safetensors"))
        diff = np.linalg.norm(flat[EMBED_KEY] - base_emb, axis=1)
        print(f"  embeddings_base.safetensors | Tokens mit veränderter Zeile: {(diff > 0).sum()} von {len(diff)}")
        del base

    if not args.no_orbax:
        print("Kopiere Original-Checkpoint (params/ + assets/) ...")
        shutil.copytree(ckpt / "params", out / "openpi_checkpoint" / "params")
        if (ckpt / "assets").exists():
            shutil.copytree(ckpt / "assets", out / "openpi_checkpoint" / "assets")

    vocab = flat[EMBED_KEY].shape[0]
    cfg = {
        "model": "pi0-FAST (openpi), PaliGemma 3B = SigLIP So400m/14 + Gemma 2B",
        "openpi_config": args.config_name,
        "checkpoint_step": ckpt.name,
        "base_checkpoint": "gs://openpi-assets/checkpoints/pi0_fast_base",
        "finetuning": "full (all weights incl. token embeddings)",
        "num_parameters": int(n_params),
        "vocab_size": int(vocab),
        "embedding_dim": int(flat[EMBED_KEY].shape[1]),
        "tokenizer": "PaliGemma SentencePiece (gs://big_vision/paligemma_tokenizer.model)",
        "prompt_format": "Task: {prompt lowercased}, State: {7 state values discretized to 256 bins};\\nAction: {FAST tokens}|",
        "fast_action_token_mapping": f"paligemma_id = {vocab} - 1 - {FAST_SKIP_TOKENS} - fast_token",
        "action_dim": 7, "action_horizon": 10, "control_hz": 10, "max_token_len": 180,
        "actions": "6 UR5e joint deltas relative to current state (rad) + gripper absolute (0 open .. 1 closed)",
        "normalization": "quantile (q01/q99) -> [-1, 1], see openpi_checkpoint/assets",
        "dataset": args.dataset,
        "weights_format": "float32, tensor names = openpi param tree joined with '/'",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (out / "README.md").write_text(README.format(**{k: v for k, v in cfg.items() if isinstance(v, (str, int))}))
    print(f"\nFertig: {out}")


README = """---
tags: [robotics, vla, pi0-fast, openpi]
datasets: [{dataset}]
---
# pi0-FAST, UR5e pick-and-place ({openpi_config}, step {checkpoint_step})

openpi pi0-FAST (PaliGemma 3B), fully fine-tuned from `pi0_fast_base` on 250 UR5e demonstrations (`{dataset}`).

- `model-*.safetensors`: all weights (float32), e.g. `PaliGemma/llm/embedder/input_embedding`
- `embeddings_finetuned.safetensors` / `embeddings_base.safetensors`: token embeddings after / before fine-tuning
- `openpi_checkpoint/`: original checkpoint for openpi
- `config.json`: prompt format, FAST token mapping, action space

```python
from safetensors.numpy import load_file
emb = load_file("embeddings_finetuned.safetensors")["input_embedding"]
```
"""


if __name__ == "__main__":
    main()