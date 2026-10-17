"""
rosbag_to_lerobot.py – ROS 2 Bags  ->  LeRobot-Datensatz (Format v2.1, wie openpi ihn erwartet)

Gedacht zum Verstehen UND Benutzen. Das Skript ist in 6 Schritte gegliedert:

    SCHRITT 1  Einstellungen       Welche Topics, welche Frequenz, welche Bildgröße?
    SCHRITT 2  Bag lesen           Gelenke + Bilder mit Zeitstempel aus der Bag holen
    SCHRITT 3  Synchronisieren     Alles auf ein festes Zeitraster (z. B. 10 Hz) bringen
    SCHRITT 4  Zustand & Aktion    state = Roboter jetzt, action = Roboter im nächsten Schritt
    SCHRITT 5  Datensatz anlegen   "Spalten" (Features) des LeRobot-Datensatzes definieren
    SCHRITT 6  Schreiben           Frame für Frame hinzufügen, Episode abschließen

Grundidee:  1 Bag = 1 Episode = 1 Demonstration.

Ordnerstruktur der Eingabe (Beispiel):
    ~/demos/pick_cube/episode_000/   <- Bag-Ordner (metadata.yaml + .mcap oder .db3)
    ~/demos/pick_cube/episode_001/
    ...
Optional pro Bag eine meta.json mit {"prompt": "put the red cube in the bowl"}.

Aufruf (im openpi-Ordner, damit die von openpi festgelegte LeRobot-Version genutzt wird):
    uv pip install rosbags
    uv run rosbag_to_lerobot.py --bags ~/demos/pick_cube --task "put the red cube in the bowl"

Nur anschauen, was passieren würde (schreibt nichts):
    uv run rosbag_to_lerobot.py --bags ~/demos/pick_cube --task "..." --dry-run
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


# =============================================================================
# SCHRITT 1 – Einstellungen
# =============================================================================
# Alles, was zu DEINEM Aufbau gehört, steht hier. Per Kommandozeile überschreibbar.

DEFAULTS = dict(
    joint_topic="/joint_states",
    base_cam_topic="/camera/D415/color/image_raw",    # feste Kamera auf den Tisch
    wrist_cam_topic="/camera/D405/color/image_raw",   # Kamera am Handgelenk
    fps=10,                 # Frequenz des Datensatzes. openpi-Beispiele nutzen 10–15 Hz.
    image_size=224,         # Bilder werden quadratisch (mit schwarzen Rändern) verkleinert.
    gripper_open=0.0,       # Rohwert des Greifergelenks, wenn offen
    gripper_closed=0.8,     # Rohwert, wenn geschlossen (Robotiq 2F-85: ca. 0.8 rad)
)

# Die 6 UR-Gelenke in DER Reihenfolge, die wir im Datensatz haben wollen.
# /joint_states liefert sie beim UR-Treiber alphabetisch sortiert – also NICHT in dieser
# Reihenfolge! Deshalb suchen wir sie unten per Name.
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
# Woran wir das Greifergelenk in /joint_states erkennen:
GRIPPER_HINTS = ("knuckle", "finger", "gripper")


# =============================================================================
# SCHRITT 2 – Bag lesen
# =============================================================================
# Eine Bag ist eine Liste von Nachrichten:  (Topic, Empfangszeit, Inhalt).
# Wir holen uns:
#   - alle /joint_states   -> Zeiten + Gelenkwinkel
#   - alle Kamerabilder    -> Zeiten (die Bilder selbst erst später, siehe unten)
#
# Warum Empfangszeit (logtime) statt header.stamp?
#   logtime stammt immer von der Systemuhr des PCs. header.stamp kommt vom Sender –
#   bei deiner D405 von der Kamerauhr, die nicht zur PC-Uhr passt.

TYPESTORE = get_typestore(Stores.ROS2_JAZZY)


def find_joint_indices(names: list[str]) -> tuple[list[int], int]:
    """Gibt die Positionen der 6 Armgelenke und des Greifers in msg.name zurück."""
    arm = []
    for joint in ARM_JOINTS:
        hits = [i for i, n in enumerate(names) if n.endswith(joint)]  # Präfix wie 'ur5e_' egal
        if len(hits) != 1:
            raise SystemExit(f"Gelenk '{joint}' nicht eindeutig in {names}")
        arm.append(hits[0])
    grip = [i for i, n in enumerate(names) if any(h in n.lower() for h in GRIPPER_HINTS)]
    if not grip:
        raise SystemExit(f"Kein Greifergelenk in /joint_states gefunden: {names}")
    return arm, grip[0]


def read_bag(bag: pathlib.Path, cfg) -> dict:
    joint_t, joint_q = [], []
    image_t = {cfg.base_cam_topic: [], cfg.wrist_cam_topic: []}
    gripper_name = None

    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        wanted = [cfg.joint_topic, cfg.base_cam_topic, cfg.wrist_cam_topic]
        available = {c.topic for c in reader.connections}
        missing = [t for t in wanted if t not in available]
        if missing:
            raise SystemExit(f"{bag.name}: Topics fehlen: {missing}\n  vorhanden: {sorted(available)}")

        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, logtime_ns, raw in reader.messages(connections=conns):
            t = logtime_ns * 1e-9  # Nanosekunden -> Sekunden

            if conn.topic == cfg.joint_topic:
                msg = reader.deserialize(raw, conn.msgtype)
                arm_idx, grip_idx = find_joint_indices(list(msg.name))
                gripper_name = msg.name[grip_idx]
                q = np.asarray(msg.position)
                joint_t.append(t)
                joint_q.append(np.append(q[arm_idx], q[grip_idx]))  # 6 Arm + 1 Greifer
            else:
                # Bilder sind groß. Wir merken uns hier nur, WANN eins kam.
                image_t[conn.topic].append(t)

    return {
        "joint_t": np.array(joint_t),
        "joint_q": np.array(joint_q),        # Form (N, 7)
        "base_t": np.array(image_t[cfg.base_cam_topic]),
        "wrist_t": np.array(image_t[cfg.wrist_cam_topic]),
        "gripper_name": gripper_name,
    }


def ros_image_to_rgb(msg) -> np.ndarray:
    """sensor_msgs/Image (roh) oder sensor_msgs/CompressedImage (JPEG) -> RGB-Array (H, W, 3)."""
    if hasattr(msg, "format"):  # CompressedImage
        return np.asarray(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB"))
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    img = data[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding.lower() == "bgr8":
        img = img[:, :, ::-1]  # BGR -> RGB
    elif msg.encoding.lower() != "rgb8":
        raise SystemExit(f"Bild-Encoding {msg.encoding} nicht unterstützt (nur rgb8/bgr8).")
    return img.copy()


def resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Seitenverhältnis behalten, auf size×size verkleinern, Rest schwarz auffüllen."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    small = Image.fromarray(img).resize((round(w * s), round(h * s)), Image.BILINEAR)
    out = Image.new("RGB", (size, size))
    out.paste(small, ((size - small.width) // 2, (size - small.height) // 2))
    return np.asarray(out)


def load_images(bag: pathlib.Path, topic: str, indices: np.ndarray, size: int) -> dict[int, np.ndarray]:
    """Zweiter Durchlauf durch die Bag: nur die Bilder dekodieren, die wir wirklich brauchen."""
    need = set(indices.tolist())
    out = {}
    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for i, (conn, _, raw) in enumerate(reader.messages(connections=conns)):
            if i in need:
                out[i] = resize_with_pad(ros_image_to_rgb(reader.deserialize(raw, conn.msgtype)), size)
    return out


# =============================================================================
# SCHRITT 3 – Synchronisieren
# =============================================================================
# Die Topics kommen mit unterschiedlichen Raten:  Gelenke 500 Hz, Kameras 30 Hz.
# Ein Datensatz-Frame braucht aber "ein Bild + einen Gelenkzustand zum selben Zeitpunkt".
#
# Lösung: Wir legen ein festes Zeitraster an (z. B. alle 0,1 s bei 10 Hz) und nehmen für
# jeden Rasterpunkt die jeweils LETZTE Nachricht davor. ("Was wusste der Roboter zu diesem
# Zeitpunkt?" – keine Information aus der Zukunft.)


def last_before(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(times, grid, side="right") - 1
    return np.clip(idx, 0, len(times) - 1)


def synchronize(raw: dict, fps: int) -> dict:
    start = max(raw["joint_t"][0], raw["base_t"][0], raw["wrist_t"][0])  # erst wenn ALLES da ist
    end = min(raw["joint_t"][-1], raw["base_t"][-1], raw["wrist_t"][-1])
    grid = np.arange(start, end, 1.0 / fps)
    return {
        "q": raw["joint_q"][last_before(raw["joint_t"], grid)],  # (T, 7)
        "base_idx": last_before(raw["base_t"], grid),             # welches Bild pro Frame
        "wrist_idx": last_before(raw["wrist_t"], grid),
    }


# =============================================================================
# SCHRITT 4 – Zustand (state) und Aktion (action)
# =============================================================================
# state[t]  = wo der Roboter JETZT ist        (6 Gelenke + Greifer)
# action[t] = wohin er als NÄCHSTES geht      = state[t+1]
#
# Das Modell lernt also: "Bei diesem Bild und diesem Zustand -> fahre dorthin."
# openpi rechnet die absoluten Ziele später selbst in Deltas um (DeltaActions in der Config).
#
# Der Greifer wird auf 0 (offen) .. 1 (geschlossen) normiert, damit er unabhängig von
# der Treiber-Einheit ist.


def make_state_and_action(q: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    grip = (q[:, 6] - cfg.gripper_open) / (cfg.gripper_closed - cfg.gripper_open)
    state = np.column_stack([q[:, :6], np.clip(grip, 0, 1)]).astype(np.float32)
    return state[:-1], state[1:]  # letzter Frame hat kein "nächstes", fällt weg


# =============================================================================
# SCHRITT 5 – Datensatz anlegen
# =============================================================================
# Ein LeRobot-Datensatz ist im Kern eine Tabelle: 1 Zeile = 1 Frame.
# "features" legt die Spalten fest – Name, Datentyp, Form.
#
# Unsere Spalten – exakt die Namen aus openpi/examples/ur5/README.md:
#   image        Basiskamera                  (224, 224, 3) uint8   -> "base_rgb"
#   wrist_image  Handgelenkkamera             (224, 224, 3) uint8   -> "wrist_rgb"
#   joints       6 Gelenkwinkel [rad]         (6,) float32
#   gripper      Greifer 0 (offen) .. 1 (zu)  (1,) float32
#   actions      Zustand im nächsten Schritt  (7,) float32 (absolut, 6 Gelenke + Greifer)
#   task         Sprachanweisung (wird zu "prompt", wenn prompt_from_task=True)
# Spalten, die LeRobot selbst ergänzt: timestamp, frame_index, episode_index, index, task_index
#
# ACHTUNG beim Abtippen der Config aus dem README:
# LeRobot speichert eine Spalte der Form (1,) als einzelne Zahl. "gripper" kommt beim Laden
# also als 0-dimensionaler Wert zurück. In UR5Inputs deshalb:
#     state = np.concatenate([data["joints"], np.atleast_1d(data["gripper"])])


def create_dataset(repo_id: str, fps: int, size: int):
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    root = HF_LEROBOT_HOME / repo_id
    if root.exists():
        shutil.rmtree(root)  # neu anfangen

    image = {"dtype": "image", "shape": (size, size, 3), "names": ["height", "width", "channel"]}
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="ur5e",
        fps=fps,
        features={
            "image": image,                                                         # Basiskamera
            "wrist_image": image,                                                   # Handgelenkkamera
            "joints": {"dtype": "float32", "shape": (6,), "names": ARM_JOINTS},
            "gripper": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ARM_JOINTS + ["gripper"]},
        },
        image_writer_threads=4,
    )


# =============================================================================
# SCHRITT 6 – Schreiben
# =============================================================================
# add_frame()     hängt eine Zeile an die aktuelle Episode an
# save_episode()  schließt die Episode ab: schreibt die Parquet-Datei, aktualisiert
#                 meta/episodes.jsonl, meta/tasks.jsonl, meta/info.json und die Statistiken


def load_paraphrases(path: pathlib.Path | None) -> dict[str, list[str]]:
    """JSON: {"<Original-Prompt aus meta.json>": ["Umformulierung 1", "Umformulierung 2", ...], ...}"""
    if path is None:
        return {}
    data = json.loads(path.expanduser().read_text())
    for canonical, variants in data.items():
        if not isinstance(variants, list) or not variants:
            raise SystemExit(f"Paraphrasen für '{canonical}' müssen eine nicht-leere Liste sein.")
    return data


def get_task(bag: pathlib.Path, fallback: str | None) -> str:
    meta = bag / "meta.json"
    if meta.exists() and json.loads(meta.read_text()).get("prompt"):
        return json.loads(meta.read_text())["prompt"]
    if fallback:
        return fallback
    raise SystemExit(f"{bag.name}: keine Sprachanweisung (meta.json fehlt, --task nicht gesetzt)")


def find_bags(path: pathlib.Path) -> list[pathlib.Path]:
    if (path / "metadata.yaml").exists():
        return [path]  # eine einzelne Bag
    bags = sorted(p for p in path.iterdir() if (p / "metadata.yaml").exists())
    if not bags:
        raise SystemExit(f"Keine Bags (Ordner mit metadata.yaml) in {path}")
    return bags


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bags", type=pathlib.Path, nargs="+", required=True,
                   help="ein oder mehrere Bag-Ordner bzw. Ordner mit vielen Bags")
    p.add_argument("--repo-id", default="local/ur5e_dataset", help="Name des Datensatzes")
    p.add_argument("--task", default=None, help="Sprachanweisung, falls keine meta.json existiert")
    p.add_argument("--dry-run", action="store_true", help="nur lesen und Statistik zeigen")
    p.add_argument("--paraphrases", type=pathlib.Path, default=None,
                   help="JSON mit Umformulierungen pro Original-Prompt. Jede Episode bekommt reihum "
                        "eine davon (für die Bedingung 'vielfältige Sprache').")
    for k, v in DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    cfg = p.parse_args()

    bags = [b for path in cfg.bags for b in find_bags(path.expanduser())]
    print(f"{len(bags)} Bag(s) gefunden.\n")
    paraphrases = load_paraphrases(cfg.paraphrases)
    dataset = None if cfg.dry_run else create_dataset(cfg.repo_id, cfg.fps, cfg.image_size)
    canonical_log = []          # welche Episode zu welcher Original-Aufgabe gehört (für die Auswertung)
    used_count: dict[str, int] = {}

    for bag in bags:
        canonical = get_task(bag, cfg.task)
        task = canonical
        if paraphrases:
            if canonical not in paraphrases:
                raise SystemExit(f"{bag.name}: keine Paraphrasen für '{canonical}' in {cfg.paraphrases}")
            k = used_count.get(canonical, 0)            # reihum -> jede Variante gleich oft
            task = paraphrases[canonical][k % len(paraphrases[canonical])]
            used_count[canonical] = k + 1
        canonical_log.append({"bag": str(bag), "canonical_task": canonical, "task": task})
        raw = read_bag(bag, cfg)                                   # Schritt 2
        sync = synchronize(raw, cfg.fps)                           # Schritt 3
        state, action = make_state_and_action(sync["q"], cfg)      # Schritt 4
        n = len(state)

        g = raw["joint_q"][:, 6]
        print(f"{bag.parent.name}/{bag.name}: {n} Frames @ {cfg.fps} Hz | Greifer '{raw['gripper_name']}' "
              f"roh {g.min():.3f}..{g.max():.3f} | \"{task}\"")
        if dataset is None:
            continue

        base = load_images(bag, cfg.base_cam_topic, sync["base_idx"][:n], cfg.image_size)
        wrist = load_images(bag, cfg.wrist_cam_topic, sync["wrist_idx"][:n], cfg.image_size)
        for i in range(n):                                         # Schritt 6
            dataset.add_frame({
                "image": base[sync["base_idx"][i]],
                "wrist_image": wrist[sync["wrist_idx"][i]],
                "joints": state[i, :6],
                "gripper": state[i, 6:],
                "actions": action[i],
                "task": task,
            })
        dataset.save_episode()

    if dataset is not None:
        # Zusatzdatei (von LeRobot ignoriert): Episode -> Original-Aufgabe, für die spätere Auswertung.
        with open(dataset.root / "meta" / "canonical_tasks.jsonl", "w") as f:
            for i, row in enumerate(canonical_log):
                f.write(json.dumps({"episode_index": i, **row}) + "\n")
        print(f"\nFertig: {dataset.root}")
        print("Anschauen mit:  uv run inspect_lerobot_dataset.py --repo-id", cfg.repo_id)


if __name__ == "__main__":
    main()
