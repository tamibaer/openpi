"""make_video_dataset.py – macht aus einem fertigen LeRobot-Bild-Datensatz eine Video-Kopie (MP4)
und lädt sie auf Hugging Face hoch. Die Kopie spielt im Visualizer auf Hugging Face ab und kann
trotzdem direkt mit openpi (pi0-FAST) trainiert werden: LeRobot liefert beim Laden dieselben
Bild-Tensoren, nur werden sie aus dem Video dekodiert statt aus PNGs.

Der Quell-Datensatz (Bilder) bleibt unverändert.

Ausführen in der openpi-Umgebung (gleiche LeRobot-Version wie beim Training):
    export HF_LEROBOT_HOME=~/ros2_ws/lerobot_datasets
    cd ~/openpi
    uv run python examples/ur5e/make_video_dataset.py \
        ~/ros2_ws/lerobot_datasets/local/ur5e_pi0-FAST --repo-id tamibaer/ur5e_pi0-FAST
    # danach hochladen (vorher einmal: uv run huggingface-cli login):
    uv run python examples/ur5e/make_video_dataset.py \
        ~/ros2_ws/lerobot_datasets/local/ur5e_pi0-FAST --repo-id tamibaer/ur5e_pi0-FAST --push-only

Am Ende vergleicht das Skript zufällige Frames aus dem Video mit den Original-Bildern
(Abweichung in Graustufen 0..255) und misst, wie lange das Dekodieren pro Frame dauert.
"""
from __future__ import annotations

import argparse
import functools
import io
import json
import pathlib
import random
import shutil
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

DEFAULT_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def decode(cell) -> np.ndarray:
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def load_source(src: pathlib.Path):
    info = json.loads((src / "meta" / "info.json").read_text())
    tasks = {}
    for line in (src / "meta" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            tasks[t["task_index"]] = t["task"]
    files = sorted((src / "data").glob("chunk-*/episode_*.parquet"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise SystemExit(f"Keine Episoden in {src}/data gefunden.")
    return info, tasks, files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=pathlib.Path, help="Ordner des Bild-Datensatzes")
    ap.add_argument("--repo-id", required=True, help="Ziel auf Hugging Face, z.B. tamibaer/ur5e_pi0-FAST")
    ap.add_argument("--vcodec", default="h264", choices=["h264", "libsvtav1"],
                    help="h264 spielt in jedem Browser; libsvtav1 (AV1) ist kleiner, aber nicht überall abspielbar")
    ap.add_argument("--crf", type=int, default=23, help="Qualität: kleiner = besser und größer (h264: 18-28)")
    ap.add_argument("--overwrite", action="store_true", help="vorhandene lokale Video-Kopie ersetzen")
    ap.add_argument("--push", action="store_true", help="nach dem Erstellen hochladen")
    ap.add_argument("--push-only", action="store_true", help="nur die vorhandene Video-Kopie hochladen")
    ap.add_argument("--public", action="store_true", help="öffentlich hochladen (Standard: privat)")
    ap.add_argument("--verify", type=int, default=200, help="so viele zufällige Frames gegen das Original prüfen")
    args = ap.parse_args()

    import lerobot.common.datasets.lerobot_dataset as lr
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    src = args.src.expanduser()
    info, tasks, files = load_source(src)
    root = HF_LEROBOT_HOME / args.repo_id
    if root.resolve() == src.resolve():
        raise SystemExit("Ziel und Quelle sind derselbe Ordner – bitte eine andere --repo-id wählen.")
    cams = [k for k, v in info["features"].items() if v["dtype"] in ("image", "video")]

    if not args.push_only:
        if root.exists():
            if not args.overwrite:
                raise SystemExit(f"Video-Kopie existiert schon: {root}\n"
                                 f"--overwrite zum Ersetzen, oder --push-only zum Hochladen.")
            shutil.rmtree(root)

        # LeRobot kodiert standardmäßig AV1 mit crf 30; hier Codec und Qualität wählbar
        lr.encode_video_frames = functools.partial(lr.encode_video_frames, vcodec=args.vcodec, crf=args.crf)

        features = {}
        for k, v in info["features"].items():
            if k in DEFAULT_KEYS:
                continue
            v = dict(v, shape=tuple(v["shape"]))          # info.json speichert Listen, LeRobot erwartet Tupel
            if k in cams:
                v = {"dtype": "video", "shape": v["shape"], "names": v["names"]}
            features[k] = v
        dst = LeRobotDataset.create(
            repo_id=args.repo_id, fps=info["fps"], robot_type=info.get("robot_type"),
            features=features, use_videos=True, image_writer_threads=10, image_writer_processes=5,
        )

        other = [k for k in features if k not in cams]
        t0 = time.time()
        for n, f in enumerate(files, 1):
            df = pq.read_table(f).to_pandas()
            for _, row in df.iterrows():
                frame = {k: np.asarray(row[k], dtype=np.float32) for k in other}
                for c in cams:
                    frame[c] = decode(row[c])
                frame["task"] = tasks[int(row["task_index"])]
                dst.add_frame(frame)
            dst.save_episode()
            eta = (time.time() - t0) / n * (len(files) - n) / 60
            print(f"\r{n}/{len(files)} Episoden kodiert (noch ca. {eta:.0f} min)", end="", flush=True)
        print(f"\nFertig: {root}")

        verify(src, files, cams, args, LeRobotDataset)

    if args.push or args.push_only:
        dst = LeRobotDataset(args.repo_id, root=root)
        print(f"Lade hoch nach https://huggingface.co/datasets/{args.repo_id} "
              f"({'öffentlich' if args.public else 'privat'}) ...")
        dst.push_to_hub(private=not args.public, tags=["ur5e", "openpi"], upload_large_folder=False)  # ein einziger Commit; viele parallele Anfragen lösen 429 aus
        print("Hochgeladen.")


def verify(src, files, cams, args, LeRobotDataset):
    """Vergleicht zufällige Frames der Video-Kopie mit den Original-PNGs."""
    if args.verify <= 0:
        return
    dst = LeRobotDataset(args.repo_id)
    # Originale in derselben Reihenfolge wie der globale Index der Kopie
    lengths = [pq.read_metadata(f).num_rows for f in files]
    if sum(lengths) != len(dst):
        print(f"⚠ Frame-Anzahl verschieden: Original {sum(lengths)}, Video {len(dst)}")
    offsets = np.cumsum([0] + lengths)
    rng = random.Random(0)
    idx = sorted(rng.sample(range(len(dst)), min(args.verify, len(dst))))
    errs = {c: [] for c in cams}
    state_err, t_dec = 0.0, []
    tables = {}
    for i in idx:
        ep = int(np.searchsorted(offsets, i, side="right") - 1)
        if ep not in tables:
            tables = {ep: pq.read_table(files[ep]).to_pandas()}
        row = tables[ep].iloc[i - offsets[ep]]
        t = time.time()
        item = dst[i]
        t_dec.append(time.time() - t)
        for c in cams:
            vid = (item[c].permute(1, 2, 0).numpy() * 255).astype(np.float32)
            errs[c].append(np.abs(vid - decode(row[c]).astype(np.float32)).mean())
        state_err = max(state_err, float(np.abs(item["state"].numpy() - np.asarray(row["state"])).max()))
    print(f"\nPrüfung an {len(idx)} zufälligen Frames:")
    for c in cams:
        e = np.array(errs[c])
        print(f"  {c}: mittlere Abweichung {e.mean():.2f} von 255 (max {e.max():.2f})")
    print(f"  state identisch: {'ja' if state_err < 1e-6 else f'nein (max {state_err:.2e})'}")
    print(f"  Dekodieren: {np.median(t_dec) * 1000:.0f} ms pro Frame (beide Kameras)")
    worst = max(np.mean(v) for v in errs.values())
    print("  Ergebnis: " + ("Bilder praktisch gleich ✓" if worst < 3 else
                            "spürbare Abweichung – kleineres --crf versuchen (z.B. 18)"))


if __name__ == "__main__":
    main()