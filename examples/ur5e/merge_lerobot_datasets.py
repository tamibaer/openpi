"""Führt mehrere LeRobot-Datensätze (aus rosbag_to_lerobot.py) zu einem zusammen.

openpi trainiert immer auf genau einem Datensatz (repo_id in der Config). Wenn die Daten schon
in mehreren LeRobot-Datensätzen liegen, z. B. local/ur5e_blocks und local/ur5e_vla_2, erzeugt
dieses Skript daraus einen neuen gemeinsamen Datensatz. Die Quellen bleiben unverändert.

Ausführen im openpi-Ordner:
    export HF_LEROBOT_HOME=~/lerobot_datasets
    uv run examples/ur5e/merge_lerobot_datasets.py \
        --sources local/ur5e_blocks local/ur5e_vla_2 --target local/ur5e_vla_all --dry-run
    uv run examples/ur5e/merge_lerobot_datasets.py \
        --sources local/ur5e_blocks local/ur5e_vla_2 --target local/ur5e_vla_all

Optionen:
    --dry-run           nur anzeigen, wie viele Episoden/Frames pro Aufgabe zusammenkämen
    --max-episodes N    höchstens N Episoden pro Quelle übernehmen (zum Ausbalancieren)
    --skip-task TEXT    Episoden mit genau diesem Prompt weglassen (mehrfach angebbar)
"""

from __future__ import annotations

import argparse
import collections
import shutil

import numpy as np

FEATURE_KEYS = ("image", "wrist_image", "state", "actions")


def to_hwc_uint8(img) -> np.ndarray:
    """LeRobot liefert Bilder als Tensor (C, H, W) in [0, 1] -> numpy (H, W, C) uint8."""
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    return arr


def episode_ranges(ds):
    starts = ds.episode_data_index["from"]
    ends = ds.episode_data_index["to"]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", required=True, help="repo_ids der Quell-Datensätze")
    p.add_argument("--target", required=True, help="repo_id des neuen gemeinsamen Datensatzes")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-episodes", type=int, default=None, help="höchstens so viele Episoden pro Quelle")
    p.add_argument("--skip-task", action="append", default=[], help="Episoden mit diesem Prompt weglassen")
    args = p.parse_args()

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    if args.target in args.sources:
        raise SystemExit("--target darf keiner der Quellen entsprechen.")

    sources = [LeRobotDataset(repo_id) for repo_id in args.sources]

    # Alle Quellen müssen zueinander passen, sonst wäre der gemeinsame Datensatz unbrauchbar.
    ref = sources[0]
    for repo_id, ds in zip(args.sources, sources):
        if ds.fps != ref.fps:
            raise SystemExit(f"{repo_id}: fps {ds.fps} passt nicht zu {args.sources[0]} ({ref.fps}).")
        for key in FEATURE_KEYS:
            if key not in ds.features:
                raise SystemExit(f"{repo_id}: Merkmal '{key}' fehlt.")
            if tuple(ds.features[key]["shape"]) != tuple(ref.features[key]["shape"]):
                raise SystemExit(f"{repo_id}: Form von '{key}' {ds.features[key]['shape']} "
                                 f"passt nicht zu {ref.features[key]['shape']}.")

    # Plan: welche Episoden aus welcher Quelle, mit welchem Prompt.
    plan = []
    per_task = collections.Counter()
    frames_per_task = collections.Counter()
    for repo_id, ds in zip(args.sources, sources):
        taken = 0
        for ep, (start, end) in enumerate(episode_ranges(ds)):
            task = ds[start]["task"]
            if task in args.skip_task:
                continue
            if args.max_episodes is not None and taken >= args.max_episodes:
                break
            plan.append((repo_id, ds, ep, start, end, task))
            per_task[task] += 1
            frames_per_task[task] += end - start
            taken += 1
        print(f"{repo_id}: {ds.num_episodes} Episoden, {taken} übernommen")

    print(f"\nGemeinsamer Datensatz {args.target}: {len(plan)} Episoden")
    for task, n in per_task.most_common():
        print(f"  {n:4d} Episoden, {frames_per_task[task]:6d} Frames  \"{task}\"")
    if args.dry_run or not plan:
        return

    shutil.rmtree(HF_LEROBOT_HOME / args.target, ignore_errors=True)
    features = {key: {k: v for k, v in ref.features[key].items() if k != "info"} for key in FEATURE_KEYS}
    target = LeRobotDataset.create(
        repo_id=args.target,
        robot_type=ref.meta.robot_type,
        fps=ref.fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for n, (repo_id, ds, ep, start, end, task) in enumerate(plan, start=1):
        for i in range(start, end):
            item = ds[i]
            frame = {
                "image": to_hwc_uint8(item["image"]),
                "wrist_image": to_hwc_uint8(item["wrist_image"]),
                "state": np.asarray(item["state"], dtype=np.float32),
                "actions": np.asarray(item["actions"], dtype=np.float32),
                "task": task,
            }
            target.add_frame(frame)
        target.save_episode()
        print(f"[{n}/{len(plan)}] {repo_id} Episode {ep}: {end - start} Frames  \"{task}\"")

    print(f"\nFertig: {target.root}")


if __name__ == "__main__":
    main()
