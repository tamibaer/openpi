"""Konvertiert ROS2-Bags (UR5e + Greifer, D415 + D405) ins LeRobot-Format für openpi.

Ausführen in der openpi-Umgebung (passende LeRobot-Version):
    export HF_LEROBOT_HOME=~/ros2_ws/lerobot_datasets
    cd ~/openpi
    uv run --with rosbags python examples/ur5e/rosbag_to_lerobot.py --bags ~/ros2_ws/dataset_vla_1 --dry-run
    uv run --with rosbags python examples/ur5e/rosbag_to_lerobot.py --bags ~/ros2_ws/dataset_vla_1
    (vorhandenen Datensatz ersetzen: --overwrite anhängen)

Greifer-Aktion (--gripper-mode):
    raw     (Standard) die gemessene Greiferposition (0 = offen, 1 = zu), um --gripper-lead
            Frames vorgezogen, weil der Greifer verzögert reagiert. Behält alle Stufen
            (erst leicht zu, hinfahren, dann ganz zu) und die echte Greifbreite.
    smooth  geglättet mit gripper_smoothing.py (Rampe auf die Greifbreite).
    binary  1, sobald der Greifer mehr als --gripper-threshold geschlossen ist.
Der State ist immer die gemessene Greiferposition.

Ein vorhandener Datensatz mit derselben --repo-id wird nur mit --overwrite ersetzt.

Pausen (--max-pause): Jeder Stillstand, in dem sich weder der Arm noch die Greifer-Aktion
ändert, wird auf --max-pause Sekunden gekürzt (Standard 0,8 s; 0 = aus). Sonst lernt das Modell,
nach dem Greifen lange stillzustehen, und bleibt am Roboter darin hängen. Die gemessene
Greiferposition zählt dabei nicht: Langsames Nachdrücken per Tippen kommt beim Ausführen nicht
vor, weil der Greifer dort in einem Rutsch auf die Zielbreite fährt. 0,8 s reichen ihm dafür.

Hängende Kamera: Die Episode läuft immer über die ganze Länge der Gelenkdaten. Hängt eine
Kamera oder startet sie zu spät, wird für die fehlende Zeit ihr letztes (bzw. erstes) Bild
wiederholt. Wie viel Prozent der Frames betroffen sind, steht in der Ausgabe.
Übersprungen wird eine Episode nur, wenn eine Kamera gar kein Bild hat oder die Episode kürzer
als 2 s ist. Am Ende steht eine Liste der übersprungenen Episoden.

Bilder: Standard sind die komprimierten Topics (.../image_raw/compressed). Fehlen sie in einem
Bag (alte Aufnahmen mit Rohbildern), wird automatisch .../image_raw genommen und umgekehrt.
Alte und neue Bags können also gemeinsam in einen Datensatz konvertiert werden.
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
    base_cam_topic="/camera/D415/color/image_raw/compressed",
    wrist_cam_topic="/camera/D405/color/image_raw/compressed",
    fps=10,
    image_size=224,
    gripper_open=0.098,
    gripper_closed=0.79,
    gripper_threshold=0.05,   # nur für --gripper-mode binary
    gripper_mode="raw",       # raw | smooth | binary
    gripper_lead=2,           # nur raw: Aktion = gemessene Position so viele Frames weiter vorn (0,2 s)
    ramp_frames=4,            # nur smooth: Länge der Greifer-Rampe in Frames
    max_pause=0.8,            # s: längere Stillstände werden gekürzt (0 = aus)
    still_eps=0.001,          # rad bzw. Greifereinheiten pro Frame: darunter gilt als "steht"
    pause_drift=0.01,         # rad: so weit darf der Arm während einer Pause insgesamt kriechen, sonst endet sie
)

ARM_JOINTS = ["shoulder_pan_joint",
              "shoulder_lift_joint",
              "elbow_joint",
              "wrist_1_joint",
              "wrist_2_joint",
              "wrist_3_joint"]

GRIPPER_HINTS = ("knuckle", "finger", "gripper")
TYPESTORE = get_typestore(Stores.ROS2_JAZZY)
MIN_SECONDS = 2.0   # kürzere Episoden (nach dem Synchronisieren) werden übersprungen


class EpisodeError(Exception):
    """Diese Episode ist unbrauchbar und wird übersprungen."""


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


def pick_topic(wanted: str, available: set[str]) -> str | None:
    """Gewünschtes Topic, sonst die komprimierte bzw. rohe Variante davon."""
    alt = wanted[: -len("/compressed")] if wanted.endswith("/compressed") else wanted + "/compressed"
    for t in (wanted, alt):
        if t in available:
            return t
    return None


def read_bag(bag: pathlib.Path, cfg) -> dict:
    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        available = {c.topic for c in reader.connections}
        base_t = pick_topic(cfg.base_cam_topic, available)
        wrist_t = pick_topic(cfg.wrist_cam_topic, available)
        missing = [w for w, t in ((cfg.joint_topic, cfg.joint_topic if cfg.joint_topic in available else None),
                                  (cfg.base_cam_topic, base_t), (cfg.wrist_cam_topic, wrist_t)) if t is None]
        if missing:
            raise EpisodeError(f"Topics fehlen (auch ohne/mit /compressed): {missing}")
        topics = [cfg.joint_topic, base_t, wrist_t]
        times = {t: [] for t in topics}
        q = []
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, t_ns, raw in reader.messages(connections=conns):
            times[conn.topic].append(t_ns * 1e-9)
            if conn.topic == cfg.joint_topic:
                msg = reader.deserialize(raw, conn.msgtype)
                q.append(np.asarray(msg.position)[joint_indices(list(msg.name))])
    return {"q": np.array(q), "base_topic": base_t, "wrist_topic": wrist_t,
            **{t: np.array(v) for t, v in times.items()}}


def to_rgb(msg) -> np.ndarray:
    if hasattr(msg, "format"):  # sensor_msgs/CompressedImage: PIL dekodiert JPEG/PNG direkt als RGB
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


def gripper_action(grip_state: np.ndarray, cfg) -> np.ndarray:
    """Greifer-Aktion für jeden Frame des Rasters (gleiche Länge wie grip_state)."""
    if cfg.gripper_mode == "raw":
        idx = np.minimum(np.arange(len(grip_state)) + cfg.gripper_lead, len(grip_state) - 1)
        return grip_state[idx].astype(np.float32)
    if cfg.gripper_mode == "binary":
        return (grip_state > cfg.gripper_threshold).astype(np.float32)
    if cfg.gripper_mode == "smooth":
        from gripper_smoothing import clean_gripper_action  # nur hier gebraucht
        return clean_gripper_action(grip_state, ramp_frames=cfg.ramp_frames).astype(np.float32)
    raise SystemExit(f"Unbekannter --gripper-mode {cfg.gripper_mode!r} (raw | smooth | binary)")


def idle_keep_mask(state_full: np.ndarray, grip_action_full: np.ndarray, cfg) -> np.ndarray:
    """Maske für die Frames 0..N-1: False = Teil einer zu langen Pause, wird entfernt.

    Frame t gilt als Stillstand, wenn sich von t nach t+1 weder ein Armgelenk noch die
    Greifer-Aktion um mehr als still_eps ändert. Von jeder Pause bleiben die ersten max_pause
    Sekunden erhalten. Zusätzlich endet eine Pause, sobald der Arm seit ihrem Beginn insgesamt
    mehr als pause_drift gekrochen ist: Sonst würde langsames Kriechen herausgeschnitten und
    an der Schnittstelle entstünde ein Sprung in den Gelenken.
    """
    n = len(state_full) - 1
    keep = np.ones(n, dtype=bool)
    max_frames = int(round(cfg.max_pause * cfg.fps))
    if cfg.max_pause <= 0:
        return keep
    arm = np.abs(np.diff(state_full[:, :6], axis=0)).max(axis=1)
    grip_a = np.abs(np.diff(grip_action_full))
    idle = (arm < cfg.still_eps) & (grip_a < cfg.still_eps)
    t = 0
    while t < n:
        if idle[t]:
            anchor = state_full[t, :6]
            end = t
            while (end < n and idle[end]
                   and np.abs(state_full[end + 1, :6] - anchor).max() < cfg.pause_drift):
                end += 1
            keep[t + max_frames:end] = False   # [t, end) ist eine Pause
            t = max(end, t + 1)
        else:
            t += 1
    return keep


def synchronize(raw: dict, cfg) -> dict:
    topics = [cfg.joint_topic, raw["base_topic"], raw["wrist_topic"]]
    for t in topics:
        if len(raw[t]) < 1:
            raise EpisodeError(f"{t} hat keine Nachrichten")
    # ganze Länge der Gelenkdaten; fehlende Kamerabilder werden mit dem nächsten vorhandenen aufgefüllt
    start, end = raw[cfg.joint_topic][0], raw[cfg.joint_topic][-1]
    grid = np.arange(start, end, 1.0 / cfg.fps)
    if len(grid) < MIN_SECONDS * cfg.fps:
        t0 = min(raw[t][0] for t in topics)
        ranges = ", ".join(f"{t.split('/')[2] if t.count('/') > 2 else t}: "
                           f"{raw[t][0] - t0:.1f}-{raw[t][-1] - t0:.1f} s ({len(raw[t])} Nachr.)"
                           for t in topics)
        raise EpisodeError(f"nur {max(len(grid), 0)} gemeinsame Frames. Zeitbereiche: {ranges}")
    last = {t: np.clip(np.searchsorted(raw[t], grid, side="right") - 1, 0, len(raw[t]) - 1) for t in topics}

    # Anteil der Frames, in denen eine Kamera kein aktuelles Bild hat (> 0,5 s außerhalb ihres Bereichs)
    stale = {}
    for key in ("base_topic", "wrist_topic"):
        t = raw[key]
        out_of_range = (grid < raw[t][0] - 0.5) | (grid > raw[t][-1] + 0.5)
        if out_of_range.any():
            stale[t] = float(out_of_range.mean())

    q = raw["q"][last[cfg.joint_topic]]
    grip = np.clip((q[:, 6] - cfg.gripper_open) / (cfg.gripper_closed - cfg.gripper_open), 0, 1)
    state = np.column_stack([q[:, :6], grip]).astype(np.float32)

    # Aktion t = Zustand t+1; Greifer-Spalte über die ganze Episode berechnet, dann verschoben
    grip_action_full = gripper_action(state[:, 6], cfg)
    action = state[1:].copy()
    action[:, 6] = grip_action_full[1:]

    keep = idle_keep_mask(state, grip_action_full, cfg)
    return {
        "state": state[:-1][keep],
        "action": action[keep],
        "base_idx": last[raw["base_topic"]][:-1][keep],
        "wrist_idx": last[raw["wrist_topic"]][:-1][keep],
        "removed": int((~keep).sum()),
        "stale": stale,
    }


def create_dataset(cfg):
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    root = HF_LEROBOT_HOME / cfg.repo_id
    if root.exists():
        if not cfg.overwrite:
            raise SystemExit(f"Datensatz existiert schon: {root}\n"
                             f"Zum Ersetzen --overwrite anhängen, oder eine andere --repo-id wählen.")
        shutil.rmtree(root)
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
    p.add_argument("--overwrite", action="store_true", help="vorhandenen Datensatz mit derselben --repo-id ersetzen")
    p.add_argument("--exclude", nargs="+", default=[], metavar="EPISODE",
                   help="Bags weglassen, z.B. --exclude episode_151 episode_152")
    for k, v in DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    cfg = p.parse_args()

    bags = [b for path in cfg.bags for b in find_bags(path.expanduser())]
    unknown = set(cfg.exclude) - {b.name for b in bags}
    if unknown:
        raise SystemExit(f"--exclude: unbekannte Bags {sorted(unknown)}")
    if cfg.exclude:
        bags = [b for b in bags if b.name not in cfg.exclude]
        print(f"Weggelassen: {', '.join(sorted(cfg.exclude))}")
    print(f"{len(bags)} Bag(s) gefunden. Greifer-Aktion: {cfg.gripper_mode}\n")
    dataset = None if cfg.dry_run else create_dataset(cfg)

    skipped = []
    for bag in bags:
        task = get_task(bag, cfg.task)
        try:
            raw = read_bag(bag, cfg)
            ep = synchronize(raw, cfg)
        except EpisodeError as e:
            print(f"⚠ {bag.name}: ÜBERSPRUNGEN – {e}")
            skipped.append(bag.name)
            continue
        g = raw["q"][:, 6]
        ga = ep["action"][:, 6]
        kind = "jpeg" if raw["base_topic"].endswith("/compressed") else "roh"
        print(f"{bag.name}: {len(ep['state'])} Frames (-{ep['removed'] / cfg.fps:.1f} s Pause) | Bilder {kind} | "
              f"Greifer roh {g.min():.3f}..{g.max():.3f} | "
              f"Aktion max {ga.max():.2f}, {(ga > 0.05).mean():.0%} der Frames geschlossen | \"{task}\"")
        for cam, frac in ep["stale"].items():
            print(f"   ⚠ {cam}: in {frac:.0%} der Frames kein aktuelles Bild – letztes/erstes Bild wird wiederholt")
        if dataset is None:
            continue

        base = load_images(bag, raw["base_topic"], ep["base_idx"], cfg.image_size)
        wrist = load_images(bag, raw["wrist_topic"], ep["wrist_idx"], cfg.image_size)
        for s, a, bi, wi in zip(ep["state"], ep["action"], ep["base_idx"], ep["wrist_idx"]):
            dataset.add_frame({
                "image": base[bi], "wrist_image": wrist[wi],
                "state": s, "actions": a, "task": task,
            })
        dataset.save_episode()

    print(f"\n{len(bags) - len(skipped)} von {len(bags)} Episoden verwendet.")
    if skipped:
        print(f"Übersprungen ({len(skipped)}): {', '.join(skipped)}")
    if dataset is not None:
        print(f"Fertig: {dataset.root}")


if __name__ == "__main__":
    main()