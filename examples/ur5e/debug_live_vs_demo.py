"""Debug: Welche Eingabe am Roboter bringt pi0-FAST aus dem Tritt?

Vergleicht eine gespeicherte Live-Beobachtung mit dem Startframe einer Demo und tauscht dann die
Eingaben einzeln aus. So siehst du, ob Kamera, Gelenkwinkel oder Greifer die Vorhersage kippen.

1. Live-Beobachtung speichern (Roboter bewegt sich nicht, Greifer offen, Szene wie in der Demo):
    python ur5e_policy_client.py --prompt "pick up the red block and put it in the blue box" \
        --save-obs live_obs.npz

2. Vergleichen (im openpi-Ordner, Policy-Server vorher beenden):
    export HF_LEROBOT_HOME=~/ros2_ws/lerobot_datasets
    uv run examples/ur5e/debug_live_vs_demo.py \
        --config pi0_fast_ur5e_lora_test_1 \
        --ckpt checkpoints/pi0_fast_ur5e_lora_test_1/v2/29999 \
        --live live_obs.npz --episode 10

Tabelle:
  Delta-Ende   geplante Gelenkänderung nach dem letzten Schritt des Chunks [rad]
  Cos          Richtungsähnlichkeit zur echten Demo-Bewegung (1 = gleich, 0 = quer, -1 = entgegengesetzt)
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

JOINTS = ["pan", "lift", "elbow", "w1", "w2", "w3"]
INPUT_KEYS = ("base_rgb", "wrist_rgb", "joints", "gripper")


def to_hwc_uint8(img) -> np.ndarray:
    img = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255).round().astype(np.uint8)
    return img


def build_variants(demo: dict, live: dict) -> list[tuple[str, dict]]:
    """Demo komplett, Live komplett und Demo mit je einer einzelnen Live-Eingabe."""
    variants = [("Demo komplett", dict(demo)), ("Live komplett", {**demo, **{k: live[k] for k in INPUT_KEYS}})]
    labels = {"base_rgb": "Basiskamera", "wrist_rgb": "Handkamera", "joints": "Gelenke", "gripper": "Greifer"}
    for key in INPUT_KEYS:
        variants.append((f"Demo + Live-{labels[key]}", {**demo, key: live[key]}))
    return variants


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-9 else float("nan")


def summarize(name: str, result: dict, obs: dict, demo_motion: np.ndarray) -> str:
    if not result.get("decode_ok", True):
        return f"{name:26s} Decoding fehlgeschlagen"
    actions = np.asarray(result["actions"])
    motion = actions[min(len(demo_motion), len(actions)) - 1, :6] - obs["joints"]
    deltas = " ".join(f"{value:+7.3f}" for value in motion)
    return f"{name:26s} {deltas}   {cosine(motion, demo_motion):+5.2f}   {actions[0, 6]:6.2f}"


def save_comparison(path: pathlib.Path, demo: dict, live: dict) -> None:
    top = np.concatenate([demo["base_rgb"], live["base_rgb"]], axis=1)
    bottom = np.concatenate([demo["wrist_rgb"], live["wrist_rgb"]], axis=1)
    Image.fromarray(np.concatenate([top, bottom], axis=0)).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--live", required=True, help="npz aus ur5e_policy_client.py --save-obs")
    p.add_argument("--repo-id", default="local/ur5e_blocks")
    p.add_argument("--episode", type=int, default=10)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--prompt", default=None, help="Standard: Prompt der Demo")
    p.add_argument(
        "--alt-prompt",
        action="append",
        default=[],
        help="zusätzlicher Prompt für den Sprachtest, mehrfach möglich",
    )
    args = p.parse_args()

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    from openpi.policies import policy_config
    from openpi.training import config as _config

    print("Lade Modell ...")
    policy = policy_config.create_trained_policy(_config.get_config(args.config), args.ckpt)
    ds = LeRobotDataset(args.repo_id)
    root = HF_LEROBOT_HOME / args.repo_id
    files = {int(f.stem.split("_")[-1]): f for f in root.glob("data/*/episode_*.parquet")}
    table = pq.read_table(files[args.episode], columns=["state", "actions"]).to_pydict()
    state, actions = np.array(table["state"]), np.array(table["actions"])
    item = ds[int(ds.episode_data_index["from"][args.episode]) + args.frame]

    prompt = args.prompt or item["task"]
    demo = {
        "base_rgb": to_hwc_uint8(item["image"]),
        "wrist_rgb": to_hwc_uint8(item["wrist_image"]),
        "joints": state[args.frame, :6].astype(np.float32),
        "gripper": state[args.frame, 6:7].astype(np.float32),
        "prompt": prompt,
    }
    with np.load(args.live) as data:
        live = {key: data[key] for key in INPUT_KEYS}

    print(f"\nPrompt: {prompt!r}")
    print("\nEingaben Live minus Demo:")
    joint_difference = live["joints"] - demo["joints"]
    print("  Gelenke [rad]: " + "  ".join(f"{n}={v:+.3f}" for n, v in zip(JOINTS, joint_difference)))
    print(f"  Greifer:       Live={float(live['gripper'][0]):.3f}  Demo={float(demo['gripper'][0]):.3f}")
    for key in ("base_rgb", "wrist_rgb"):
        print(f"  {key}: mittlere Pixeldifferenz {np.abs(live[key].astype(float) - demo[key].astype(float)).mean():.1f}/255")

    image_path = pathlib.Path("live_vs_demo.png")
    save_comparison(image_path, demo, live)
    print(f"  Bildvergleich: {image_path} (links Demo, rechts Live; oben Basis, unten Handgelenk)")

    demo_motion = actions[args.frame + args.horizon - 1, :6] - state[args.frame, :6]
    print("\nEchte Demo-Bewegung: " + " ".join(f"{value:+7.3f}" for value in demo_motion))
    print(f"\n{'Variante':26s} " + " ".join(f"{name:>7s}" for name in JOINTS) + "     Cos  Greifer")
    for name, obs in build_variants(demo, live):
        print(summarize(name, policy.infer(obs), obs, demo_motion))

    if args.alt_prompt:
        print("\nSprachtest mit Live-Eingaben:")
        live_obs = {**demo, **live}
        for alternative in args.alt_prompt:
            print(summarize(alternative[:26], policy.infer({**live_obs, "prompt": alternative}), live_obs, demo_motion))


if __name__ == "__main__":
    main()
