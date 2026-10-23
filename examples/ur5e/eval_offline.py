"""Offline-Test: Hat das Modell die Demos gelernt?

Schickt Bilder + Gelenkwinkel aus aufgenommenen Episoden an das trainierte Modell und
vergleicht die vorhergesagten Aktionen mit dem, was in der Demo tatsächlich passiert ist.
Kein Roboter nötig.

Aufruf (im openpi-Ordner, Policy-Server vorher beenden – sonst reicht der GPU-Speicher evtl. nicht):
    export HF_LEROBOT_HOME=~/ros2_ws/lerobot_datasets
    uv run examples/ur5e/eval_offline.py \
        --ckpt checkpoints/pi05_ur5e_lora/blocks_v1/19999 \
        --episodes 10 12 15 25 35

Ausgabe pro Episode:
  Modell-Fehler   mittlere Abweichung Vorhersage vs. Demo (rad, über 1 s Horizont)
  Stillstand      Fehler, wenn man einfach "nicht bewegen" vorhersagen würde
  -> Modell-Fehler deutlich KLEINER als Stillstand = Modell hat gelernt
  Streuung        wie stark zwei Vorhersagen für dieselbe Szene voneinander abweichen
  -> große Streuung = Modell ist unsicher (erklärt Hin-und-her-Fahren)
  Greifer         Anteil richtiger auf/zu-Entscheidungen
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pyarrow.parquet as pq


def to_hwc_uint8(img) -> np.ndarray:
    img = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255).round().astype(np.uint8)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pi05_ur5e_lora")
    p.add_argument("--ckpt", required=True, help="Checkpoint-Ordner mit params/ und assets/")
    p.add_argument("--repo-id", default="local/ur5e_blocks")
    p.add_argument("--episodes", type=int, nargs="+", default=[10, 12, 15, 30, 50])
    p.add_argument("--every", type=int, default=20, help="jeden N-ten Frame testen")
    p.add_argument("--horizon", type=int, default=10)
    args = p.parse_args()

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    from openpi.policies import policy_config
    from openpi.training import config as _config

    print("Lade Modell ...")
    policy = policy_config.create_trained_policy(_config.get_config(args.config), args.ckpt)
    ds = LeRobotDataset(args.repo_id)
    root = HF_LEROBOT_HOME / args.repo_id
    files = {int(f.stem.split("_")[-1]): f for f in root.glob("data/*/episode_*.parquet")}

    names = ["pan", "lift", "elbow", "w1", "w2", "w3"]
    print(f"\n{'Episode':8s} {'Frames':>6s} {'Modell-Fehler':>14s} {'Stillstand':>11s} {'Streuung':>9s} {'Greifer':>8s}  Prompt")
    totals = []
    for ep in args.episodes:
        t = pq.read_table(files[ep], columns=["state", "actions"]).to_pydict()
        state, actions = np.array(t["state"]), np.array(t["actions"])
        start = int(ds.episode_data_index["from"][ep])
        n = len(state)

        err_model, err_still, spread, grip_ok, per_joint = [], [], [], [], []
        prompt = None
        for i in range(0, n - args.horizon, args.every):
            item = ds[start + i]
            prompt = item["task"]
            obs = {
                "base_rgb": to_hwc_uint8(item["image"]),
                "wrist_rgb": to_hwc_uint8(item["wrist_image"]),
                "joints": state[i, :6].astype(np.float32),
                "gripper": state[i, 6:7].astype(np.float32),
                "prompt": prompt,
            }
            pred1 = np.asarray(policy.infer(obs)["actions"])[: args.horizon]
            pred2 = np.asarray(policy.infer(obs)["actions"])[: args.horizon]
            gt = actions[i : i + args.horizon]

            e = np.abs(pred1[:, :6] - gt[:, :6])
            err_model.append(e.mean())
            per_joint.append(e.mean(0))
            err_still.append(np.abs(state[i, :6] - gt[:, :6]).mean())
            spread.append(np.abs(pred1[:, :6] - pred2[:, :6]).mean())
            grip_ok.append(((pred1[:, 6] > 0.5) == (gt[:, 6] > 0.5)).mean())

        row = (np.mean(err_model), np.mean(err_still), np.mean(spread), np.mean(grip_ok))
        totals.append(row)
        print(f"{ep:8d} {n:6d} {row[0]:14.4f} {row[1]:11.4f} {row[2]:9.4f} {row[3]:8.0%}  \"{prompt}\"")
        pj = np.mean(per_joint, 0)
        print("         pro Gelenk: " + "  ".join(f"{k}={v:.3f}" for k, v in zip(names, pj)))

    m = np.mean(totals, 0)
    print(f"\n{'Gesamt':8s} {'':6s} {m[0]:14.4f} {m[1]:11.4f} {m[2]:9.4f} {m[3]:8.0%}")
    ratio = m[0] / max(m[1], 1e-9)
    print(f"\nModell-Fehler / Stillstand = {ratio:.2f}")
    if ratio < 0.4:
        print("-> Modell sagt die Demos gut voraus. Problem liegt eher bei der Ausführung am Roboter.")
    elif ratio < 0.8:
        print("-> Modell hat etwas gelernt, ist aber ungenau.")
    else:
        print("-> Modell ist kaum besser als Stillstand. Training/Daten prüfen.")
    if m[2] > 0.5 * m[0]:
        print("-> Hohe Streuung: Modell ist unsicher, zwei Vorhersagen für dieselbe Szene weichen stark ab.")


if __name__ == "__main__":
    main()
