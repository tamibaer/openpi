"""verify_export.py – prüft einen mit export_for_hf.py erzeugten Ordner, bevor er hochgeladen wird.

  1. Vollständig: alle Dateien da, Index zeigt auf vorhandene Dateien, README verlinkt den Datensatz
  2. Identisch:   jeder Tensor in den safetensors-Dateien ist bitgenau gleich dem Original-Checkpoint
  3. Embeddings:  embeddings_finetuned == Modell-Embedding; Vergleich mit dem Basismodell, inkl.
                  Plausibilitätscheck: Tokens aus euren Prompts sollten sich stärker verändert haben
                  als zufällige Tokens des Vokabulars

    cd ~/openpi
    uv run python examples/ur5e/verify_export.py ~/hf_export/pi0_fast_ur5e_full_v2 \
        --checkpoint checkpoints/pi0_fast_ur5e_full_v2/v1/19999
"""
import argparse
import json
import pathlib

import numpy as np

EMBED_KEY = "PaliGemma/llm/embedder/input_embedding"
PROMPT_WORDS = ["strawberry", "screwdriver", "towel", "rope", "ruler", "folding", "block", "cube",
                "yellow", "green", "orange", "blue", "white", "cardboard", "box", "pick", "put", "take"]


def flatten(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        out.update(flatten(v, key) if isinstance(v, dict) else {key: v})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=pathlib.Path)
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Original-Checkpoint (Ordner mit params/)")
    args = ap.parse_args()
    from safetensors import safe_open

    ex = args.export.expanduser()
    problems = []

    # ---- 1. Vollständigkeit
    print("1. Dateien")
    index = json.loads((ex / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    for f in [*shards, "embeddings_finetuned.safetensors", "config.json", "README.md"]:
        ok = (ex / f).exists()
        print(f"   {'✓' if ok else '✗'} {f}")
        if not ok:
            problems.append(f"{f} fehlt")
    has_base = (ex / "embeddings_base.safetensors").exists()
    print(f"   {'✓' if has_base else '–'} embeddings_base.safetensors {'(nicht exportiert)' if not has_base else ''}")
    orbax = ex / "openpi_checkpoint" / "params"
    print(f"   {'✓' if orbax.exists() else '–'} openpi_checkpoint/ {'' if orbax.exists() else '(mit --no-orbax weggelassen)'}")
    norm = list((ex / "openpi_checkpoint" / "assets").rglob("norm_stats.json")) if orbax.exists() else []
    if orbax.exists():
        print(f"   {'✓' if norm else '✗'} Normierungswerte (norm_stats.json)")
        if not norm:
            problems.append("openpi_checkpoint ohne norm_stats.json – serve_policy bräuchte sie")
    readme = (ex / "README.md").read_text() if (ex / "README.md").exists() else ""
    ds_line = next((l for l in readme.splitlines() if l.startswith("datasets:")), None)
    print(f"   {'✓' if ds_line else '✗'} Datensatz-Link in README: {ds_line or 'fehlt'}")
    if not ds_line:
        problems.append("README ohne 'datasets:'-Zeile")

    # ---- 2. Bitgenau gleich wie das Original
    print("\n2. Vergleich mit dem Original-Checkpoint (dauert etwas)")
    import openpi.models.model as _model
    orig = flatten(_model.restore_params(args.checkpoint.expanduser().resolve() / "params", restore_type=np.ndarray))
    missing = sorted(set(orig) - set(index["weight_map"]))
    extra = sorted(set(index["weight_map"]) - set(orig))
    diff = []
    for shard in shards:
        with safe_open(str(ex / shard), framework="np") as f:
            for k in f.keys():
                if k in orig and not np.array_equal(f.get_tensor(k), np.asarray(orig[k])):
                    diff.append(k)
    print(f"   Tensoren: Original {len(orig)}, Export {len(index['weight_map'])} | "
          f"fehlend {len(missing)}, zusätzlich {len(extra)}, abweichend {len(diff)}")
    for name, lst in (("fehlt", missing), ("zusätzlich", extra), ("weicht ab", diff)):
        for k in lst[:5]:
            print(f"   ✗ {name}: {k}")
        if lst:
            problems.append(f"{len(lst)} Tensor(en) {name}")
    if not (missing or extra or diff):
        print("   ✓ alle Gewichte bitgenau identisch")

    # ---- 3. Embeddings
    print("\n3. Embeddings")
    with safe_open(str(ex / "embeddings_finetuned.safetensors"), framework="np") as f:
        ft = f.get_tensor("input_embedding")
    same = np.array_equal(ft, np.asarray(orig[EMBED_KEY]))
    print(f"   {'✓' if same else '✗'} embeddings_finetuned = Modell-Embedding {ft.shape}")
    if not same:
        problems.append("embeddings_finetuned weicht vom Modell ab")
    del orig
    if has_base:
        with safe_open(str(ex / "embeddings_base.safetensors"), framework="np") as f:
            base = f.get_tensor("input_embedding")
        change = np.linalg.norm(ft - base, axis=1)
        rel = change / (np.linalg.norm(base, axis=1) + 1e-8)
        print(f"   veränderte Token-Zeilen: {(change > 0).sum()} von {len(change)} | "
              f"relative Änderung Median {np.median(rel):.4f}, 99 % {np.percentile(rel, 99):.4f}")
        try:
            import sentencepiece
            import openpi.shared.download as download
            path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            sp = sentencepiece.SentencePieceProcessor(model_proto=path.read_bytes())
            ids = sorted({i for w in PROMPT_WORDS for i in sp.encode(" " + w)})
            print(f"   Prompt-Tokens ({len(ids)}): mittlere relative Änderung {rel[ids].mean():.4f} "
                  f"vs. ganzes Vokabular {rel.mean():.4f}")
            top = sorted(((rel[i], sp.id_to_piece(int(i))) for i in ids), reverse=True)[:8]
            print("   am stärksten verändert: " + ", ".join(f"{p!r} {r:.3f}" for r, p in top))
            if rel[ids].mean() <= rel.mean():
                print("   ⚠ Prompt-Tokens nicht stärker verändert als der Durchschnitt – unerwartet, bitte ansehen")
        except Exception as e:  # Tokenizer nicht erreichbar: nur dieser Zusatzcheck fällt weg
            print(f"   (Token-Check übersprungen: {e})")

    print("\nErgebnis: " + ("✓ Export passt, kann hochgeladen werden" if not problems else
                            "✗ Probleme:\n   - " + "\n   - ".join(problems)))


if __name__ == "__main__":
    main()