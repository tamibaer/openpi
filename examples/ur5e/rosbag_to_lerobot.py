"""Konvertiert ROS2-Bags (UR5e + Greifer, D415 + D405) ins LeRobot-Format für openpi.

Ausführen in der openpi-Umgebung (passende LeRobot-Version):
    export HF_LEROBOT_HOME=~/ros2_ws/lerobot_datasets
    cd ~/openpi
    uv run --with rosbags python examples/ur5e/rosbag_to_lerobot.py --bags ~/ros2_ws/dataset_vla_1 --dry-run
    uv run --with rosbags python examples/ur5e/rosbag_to_lerobot.py --bags ~/ros2_ws/dataset_vla_1
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import shutil

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

DEFAULTS = dict(
    joint_topic="/joint_states",
    base_cam_topic="/camera/D415/color/image_raw",
    wrist_cam_topic="/camera/D405/color/image_raw",
    fps=10,
    image_size=224,
    gripper_open=0.098,
    gripper_closed=0.79,
    gripper_threshold=0.05,   
)

ARM_JOINTS = ["shoulder_pan_joint", 
              "shoulder_lift_joint", 
              "elbow_joint",
              "wrist_1_joint", 
              "wrist_2_joint", 
              "wrist_3_joint"]

GRIPPER_HINTS = ("knuckle", "finger", "gripper")
TYPESTORE = get_typestore(Stores.ROS2_JAZZY)


def joint_indices(names: list[str]) -> list[int]:
    idx = []
    for joint in ARM_JOINTS:
        hits = [i for i, n in enumerate(names) if n.endswith(joint)]
        if len(hits) != 1:
            raise SystemExit(f"Gelenk '{joint}' nicht eindeutig in {names}")
        idx += hits
    grip = [i for i, n in enumerate(names) if any(h in n.lower() for h in GRIPPER_HINTS)]
    if not grip:
        raise SystemExit(f"Kein Greifergelenk in {names}")
    return idx + grip[:1]


def read_bag(bag: pathlib.Path, cfg) -> dict:
    topics = [cfg.joint_topic, cfg.base_cam_topic, cfg.wrist_cam_topic]
    times = {t: [] for t in topics}
    q = []
    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        missing = set(topics) - {c.topic for c in reader.connections}
        if missing:
            raise SystemExit(f"{bag.name}: Topics fehlen: {sorted(missing)}")
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, t_ns, raw in reader.messages(connections=conns):
            times[conn.topic].append(t_ns * 1e-9)
            if conn.topic == cfg.joint_topic:
                msg = reader.deserialize(raw, conn.msgtype)
                q.append(np.asarray(msg.position)[joint_indices(list(msg.name))])
    return {"q": np.array(q), **{t: np.array(v) for t, v in times.items()}}


def to_rgb(msg) -> np.ndarray:
    if hasattr(msg, "format"):
        return np.asarray(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB"))
    img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
    img = img.reshape(msg.height, msg.width, 3)
    enc = msg.encoding.lower()
    if enc not in ("rgb8", "bgr8"):
        raise SystemExit(f"Bild-Encoding {msg.encoding} nicht unterstützt.")
    return img[:, :, ::-1].copy() if enc == "bgr8" else img.copy()


def resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    small = Image.fromarray(img).resize((round(w * s), round(h * s)), Image.BILINEAR)
    out = Image.new("RGB", (size, size))
    out.paste(small, ((size - small.width) // 2, (size - small.height) // 2))
    return np.asarray(out)


def load_images(bag: pathlib.Path, topic: str, wanted: np.ndarray, size: int) -> dict[int, np.ndarray]:
    wanted, out = set(wanted.tolist()), {}
    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for i, (conn, _, raw) in enumerate(reader.messages(connections=conns)):
            if i in wanted:
                out[i] = resize_with_pad(to_rgb(reader.deserialize(raw, conn.msgtype)), size)
    return out


def synchronize(raw: dict, cfg) -> dict:
    topics = [cfg.joint_topic, cfg.base_cam_topic, cfg.wrist_cam_topic]
    start = max(raw[t][0] for t in topics)
    end = min(raw[t][-1] for t in topics)
    grid = np.arange(start, end, 1.0 / cfg.fps)
    last = {t: np.clip(np.searchsorted(raw[t], grid, side="right") - 1, 0, len(raw[t]) - 1) for t in topics}

    q = raw["q"][last[cfg.joint_topic]]
    grip = (q[:, 6] - cfg.gripper_open) / (cfg.gripper_closed - cfg.gripper_open)
    state = np.column_stack([q[:, :6], np.clip(grip, 0, 1)]).astype(np.float32)
    action = state[1:].copy()
    action[:, 6] = (action[:, 6] > cfg.gripper_threshold).astype(np.float32)
    return {
        "state": state[:-1],
        "action": action,
        "base_idx": last[cfg.base_cam_topic][:-1],
        "wrist_idx": last[cfg.wrist_cam_topic][:-1],
    }


def create_dataset(cfg):
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    shutil.rmtree(HF_LEROBOT_HOME / cfg.repo_id, ignore_errors=True)
    image = {"dtype": "image", "shape": (cfg.image_size, cfg.image_size, 3), "names": ["height", "width", "channel"]}
    return LeRobotDataset.create(
        repo_id=cfg.repo_id,
        robot_type="ur5e",
        fps=cfg.fps,
        features={
            "image": image,           
            "wrist_image": image,     
            "state": {"dtype": "float32", "shape": (7,), "names": ARM_JOINTS + ["gripper"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ARM_JOINTS + ["gripper"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


def get_task(bag: pathlib.Path, fallback: str | None) -> str:
    meta = bag / "meta.json"
    task = json.loads(meta.read_text()).get("prompt") if meta.exists() else None
    if not (task or fallback):
        raise SystemExit(f"{bag.name}: keine Sprachanweisung (meta.json fehlt, --task nicht gesetzt)")
    return task or fallback


def find_bags(path: pathlib.Path) -> list[pathlib.Path]:
    if (path / "metadata.yaml").exists():
        return [path]
    bags = sorted(p for p in path.iterdir() if (p / "metadata.yaml").exists())
    if not bags:
        raise SystemExit(f"Keine Bags (Ordner mit metadata.yaml) in {path}")
    return bags


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bags", type=pathlib.Path, nargs="+", required=True, help="Bag-Ordner oder Ordner mit Bags")
    p.add_argument("--repo-id", default="local/ur5e_blocks")
    p.add_argument("--task", help="Sprachanweisung, falls keine meta.json existiert")
    p.add_argument("--dry-run", action="store_true", help="nur lesen und Statistik zeigen")
    for k, v in DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    cfg = p.parse_args()

    bags = [b for path in cfg.bags for b in find_bags(path.expanduser())]
    print(f"{len(bags)} Bag(s) gefunden.\n")
    dataset = None if cfg.dry_run else create_dataset(cfg)

    for bag in bags:
        task = get_task(bag, cfg.task)
        raw = read_bag(bag, cfg)
        ep = synchronize(raw, cfg)
        g = raw["q"][:, 6]
        closed = ep["action"][:, 6].mean()
        print(f"{bag.name}: {len(ep['state'])} Frames | Greifer roh {g.min():.3f}..{g.max():.3f}, "
              f"{closed:.0%} der Frames 'zu' | \"{task}\"")
        if dataset is None:
            continue

        base = load_images(bag, cfg.base_cam_topic, ep["base_idx"], cfg.image_size)
        wrist = load_images(bag, cfg.wrist_cam_topic, ep["wrist_idx"], cfg.image_size)
        for s, a, bi, wi in zip(ep["state"], ep["action"], ep["base_idx"], ep["wrist_idx"]):
            dataset.add_frame({
                "image": base[bi], "wrist_image": wrist[wi],
                "state": s, "actions": a, "task": task,
            })
        dataset.save_episode()

    if dataset is not None:
        print(f"\nFertig: {dataset.root}")


if __name__ == "__main__":
    main()