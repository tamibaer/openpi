"""
inspect_lerobot_dataset.py – Einen LeRobot-Datensatz anschauen und verstehen.

Zeigt:
  1. die Ordnerstruktur auf der Festplatte
  2. meta/info.json        (Bauplan: Spalten, fps, Anzahl Episoden/Frames)
  3. meta/tasks.jsonl      (alle Sprachanweisungen)
  4. meta/episodes.jsonl   (Länge jeder Episode)
  5. eine Zeile der Parquet-Tabelle (ein Frame)
  6. speichert die ersten Bilder als PNG, damit du sie dir ansehen kannst
  7. den Zugriff, wie openpi ihn beim Training macht (dataset[i])

Aufruf (im openpi-Ordner):
    uv run inspect_lerobot_dataset.py --repo-id local/ur5e_dataset
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq


def tree(root: pathlib.Path, max_files: int = 3, indent: str = ""):
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    files = [e for e in entries if e.is_file()]
    for d in [e for e in entries if e.is_dir()]:
        print(f"{indent}{d.name}/")
        tree(d, max_files, indent + "    ")
    for f in files[:max_files]:
        print(f"{indent}{f.name}  ({f.stat().st_size / 1024:.0f} KB)")
    if len(files) > max_files:
        print(f"{indent}... {len(files) - max_files} weitere")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="local/ur5e_dataset")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("dataset_preview"))
    args = p.parse_args()

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    root = HF_LEROBOT_HOME / args.repo_id

    print("=" * 70, "\n1) ORDNERSTRUKTUR:", root, "\n" + "=" * 70)
    tree(root)

    print("\n" + "=" * 70, "\n2) meta/info.json  (der Bauplan)\n" + "=" * 70)
    info = json.loads((root / "meta/info.json").read_text())
    for k in ("codebase_version", "robot_type", "fps", "total_episodes", "total_frames", "total_tasks"):
        print(f"  {k:18s} {info[k]}")
    print("  features (Spalten):")
    for name, ft in info["features"].items():
        print(f"    {name:15s} dtype={ft['dtype']:8s} shape={ft['shape']}")

    print("\n" + "=" * 70, "\n3) meta/tasks.jsonl  (Sprachanweisungen)\n" + "=" * 70)
    print((root / "meta/tasks.jsonl").read_text().strip())

    print("\n" + "=" * 70, "\n4) meta/episodes.jsonl  (pro Episode)\n" + "=" * 70)
    print((root / "meta/episodes.jsonl").read_text().strip())

    print("\n" + "=" * 70, "\n5) EIN FRAME aus data/chunk-000/episode_000000.parquet\n" + "=" * 70)
    table = pq.read_table(root / "data/chunk-000/episode_000000.parquet")
    print(f"  {table.num_rows} Zeilen (Frames), Spalten: {table.column_names}")
    row = table.slice(0, 1).to_pylist()[0]
    for k, v in row.items():
        if isinstance(v, dict) and "bytes" in v:
            print(f"  {k:15s} <PNG-Bild, {len(v['bytes']) / 1024:.0f} KB>")
        else:
            print(f"  {k:15s} {np.round(v, 4) if isinstance(v, list) else v}")

    print("\n" + "=" * 70, "\n6) BILDER speichern ->", args.out.resolve(), "\n" + "=" * 70)
    args.out.mkdir(exist_ok=True)
    ds = LeRobotDataset(args.repo_id)
    for i in (0, len(ds) // 2, len(ds) - 1):
        item = ds[i]
        for key in ("image", "wrist_image"):
            img = (item[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # (C,H,W) float -> (H,W,C) uint8
            from PIL import Image
            path = args.out / f"frame{i:05d}_{key}.png"
            Image.fromarray(img).save(path)
            print("  ", path)

    print("\n" + "=" * 70, "\n7) SO SIEHT openpi EINEN FRAME (dataset[0])\n" + "=" * 70)
    item = ds[0]
    for k, v in item.items():
        desc = f"Tensor {tuple(v.shape)} {v.dtype}" if hasattr(v, "shape") else repr(v)
        print(f"  {k:15s} {desc}")


if __name__ == "__main__":
    main()
