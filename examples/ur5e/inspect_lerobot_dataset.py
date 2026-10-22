"""Einen LeRobot-Datensatz (aus rosbag_to_lerobot.py) prüfen und anschauen.

Zeigt Metadaten und Prompts, prüft jede Episode auf typische Fehler und speichert nach --out:
  episode_XXX.png   Gelenke, Greifer-Zustand und Greifer-Aktion über die Zeit
  frames_XXX.png    Basis- und Handgelenkbild am Anfang, in der Mitte und am Ende

    uv run --with matplotlib examples/ur5e/inspect_lerobot_dataset.py --repo-id local/ur5e_blocks
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

MAX_JOINT_STEP = 0.2  


def load_episodes(root: pathlib.Path) -> list[dict]:
    """Nur die Zahlenspalten jeder Episode (ohne Bilder) laden."""
    episodes = []
    for f in sorted(root.glob("data/*/episode_*.parquet")):
        t = pq.read_table(f, columns=["state", "actions", "task_index"]).to_pydict()
        state = np.array(t["state"])  # (T, 7): 6 Armgelenke + Greifer
        episodes.append({
            "name": f.stem,
            "joints": state[:, :6],
            "gripper": state[:, 6],
            "actions": np.array(t["actions"]),
            "task_index": t["task_index"][0],
        })
    return episodes


def check(ep: dict) -> list[str]:
    """Liste von Warnungen; leer = Episode sieht gut aus."""
    warn = []
    j, a, g = ep["joints"], ep["actions"], ep["gripper"]
    if not np.isin(a[:, 6], (0.0, 1.0)).all():
        warn.append("Greifer-Aktion nicht binär")
    closed = a[:, 6].mean()
    if closed in (0.0, 1.0):
        warn.append(f"Greifer-Aktion immer {int(closed)} -> gripper_open/closed prüfen")
    if not np.allclose(a[:-1, :6], j[1:], atol=1e-5):
        warn.append("actions[t] != joints[t+1]")
    step = np.abs(np.diff(j, axis=0)).max()
    if step > MAX_JOINT_STEP:
        warn.append(f"Gelenksprung {step:.2f} rad in einem Frame")
    if g.min() < 0 or g.max() > 1:
        warn.append("Greifer-Zustand außerhalb 0..1")
    return warn


def plot_episode(ep: dict, fps: int, path: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(ep["joints"])) / fps
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t, ep["joints"])
    ax1.set_ylabel("Gelenke [rad]")
    ax1.legend(["pan", "lift", "elbow", "w1", "w2", "w3"], ncol=6, fontsize=8)
    ax2.plot(t, ep["gripper"], label="Zustand (gemessen)")
    ax2.step(t, ep["actions"][:, 6], where="post", label="Aktion (binär)")
    ax2.set_ylabel("Greifer")
    ax2.set_xlabel("Zeit [s]")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_frames(ds, ep_index: int, path: pathlib.Path):
    from PIL import Image

    start = ds.episode_data_index["from"][ep_index].item()
    end = ds.episode_data_index["to"][ep_index].item() - 1
    rows = []
    for i in (start, (start + end) // 2, end):
        item = ds[i]
        imgs = [(item[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for k in ("image", "wrist_image")]
        rows.append(np.concatenate(imgs, axis=1)) 
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="local/ur5e_blocks")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("dataset_preview"))
    p.add_argument("--no-images", action="store_true", help="keine Bilder/Plots speichern (schneller)")
    args = p.parse_args()

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    root = HF_LEROBOT_HOME / args.repo_id
    info = json.loads((root / "meta/info.json").read_text())
    tasks = {r["task_index"]: r["task"] for r in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}

    print(f"Datensatz: {root}")
    print(f"  {info['total_episodes']} Episoden, {info['total_frames']} Frames @ {info['fps']} Hz " f"({info['total_frames'] / info['fps'] / 60:.1f} min), Format {info['codebase_version']}")
    print("  Spalten:", ", ".join(f"{k}{tuple(v['shape'])}" for k, v in info["features"].items() if k not in ("timestamp", "frame_index", "episode_index", "index", "task_index")))

    episodes = load_episodes(root)
    print(f"\n{'Episode':22s} {'Frames':>6s} {'zu':>5s}  Prompt / Warnungen")
    for ep in episodes:
        warn = check(ep)
        status = "  ⚠ " + "; ".join(warn) if warn else ""
        print(f"{ep['name']:22s} {len(ep['joints']):6d} {ep['actions'][:, 6].mean():5.0%}  "
              f"\"{tasks[ep['task_index']]}\"{status}")

    all_j = np.concatenate([e["joints"] for e in episodes])
    print("\nGelenkbereiche [rad]:")
    for name, lo, hi in zip(info["features"]["state"]["names"][:6], all_j.min(0), all_j.max(0)):
        print(f"  {name:20s} {lo:+.2f} .. {hi:+.2f}")

    if args.no_images:
        return
    args.out.mkdir(exist_ok=True)
    ds = LeRobotDataset(args.repo_id)
    for k, ep in enumerate(episodes):
        plot_episode(ep, info["fps"], args.out / f"episode_{k:03d}.png")
        save_frames(ds, k, args.out / f"frames_{k:03d}.png")
    print(f"\nPlots und Bilder: {args.out.resolve()}")
    print("  frames_XXX.png: Zeilen = Anfang/Mitte/Ende, links Basis-, rechts Handgelenkkamera")


if __name__ == "__main__":
    main()