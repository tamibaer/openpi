"""diagnose_jumps.py – woher kommen die Gelenksprünge am Episodenanfang?

Liest für jede angegebene Episode den Bag direkt (ohne Konverter) und zeigt die ersten Sekunden:
  - Zeitlücken zwischen /joint_states-Nachrichten (Lücke = Daten fehlen -> Sprung)
  - Gelenkgeschwindigkeit (schnelle, aber glatte Fahrt = z.B. Homing-Bewegung am Start)

    cd ~/openpi
    uv run --with rosbags python examples/ur5e/diagnose_jumps.py ~/ros2_ws/dataset_pi0-FAST episode_156 episode_220
"""
import pathlib
import sys

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
SECONDS = 4.0


def main():
    root = pathlib.Path(sys.argv[1]).expanduser()
    store = get_typestore(Stores.ROS2_JAZZY)
    for name in sys.argv[2:]:
        t, h, q = [], [], []
        with AnyReader([root / name], default_typestore=store) as r:
            conns = [c for c in r.connections if c.topic == "/joint_states"]
            for conn, ns, raw in r.messages(connections=conns):
                m = r.deserialize(raw, conn.msgtype)
                pos = dict(zip(m.name, m.position))
                try:
                    q.append([pos[next(n for n in pos if n.endswith(j))] for j in ARM])
                except StopIteration:
                    continue
                t.append(ns * 1e-9)
                h.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        t, h, q = np.array(t), np.array(h), np.array(q)
        t -= t[0]
        h -= h[0]
        dt = np.diff(t)
        step = np.abs(np.diff(q, axis=0)).max(axis=1)
        speed = step / np.maximum(dt, 1e-6)
        print(f"\n=== {name}: {len(t)} joint_states, Rate Median {1 / np.median(dt):.0f} Hz")
        m = t[1:] < SECONDS
        gaps = np.flatnonzero(m & (dt > 5 * np.median(dt)))
        print(f"  Zeitlücken in den ersten {SECONDS:g} s: " +
              (", ".join(f"{t[i]:.2f} s: {dt[i] * 1000:.0f} ms, Sprung {step[i]:.3f} rad" for i in gaps) or "keine"))
        print(f"  max. Gelenkgeschwindigkeit in den ersten {SECONDS:g} s: {speed[m].max():.2f} rad/s "
              f"(bei {t[1:][m][speed[m].argmax()]:.2f} s) | Rest der Episode: {np.percentile(speed[~m], 99.9):.2f} rad/s")
        # Verlauf in 0,1-s-Schritten (wie im Datensatz)
        grid = np.arange(0, SECONDS, 0.1)
        idx = np.searchsorted(t, grid, side="right") - 1
        d = np.abs(np.diff(q[idx], axis=0)).max(axis=1)
        print("  Schritt pro 0,1 s, Bag-Zeit    [rad]: " + " ".join(f"{x:.3f}" for x in d))
        idx = np.searchsorted(h, grid, side="right") - 1
        d = np.abs(np.diff(q[idx], axis=0)).max(axis=1)
        print("  Schritt pro 0,1 s, Header-Zeit [rad]: " + " ".join(f"{x:.3f}" for x in d))


if __name__ == "__main__":
    main()