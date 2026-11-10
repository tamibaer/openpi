"""ur5e_policy_client.py – minimale Ansteuerung des UR5e mit dem openpi-Policy-Server.

Schleife: Gelenke + 2 Kamerabilder lesen -> an den Server schicken -> die ersten EXEC_STEPS
Aktionen als Trajektorie fahren -> Greifer setzen -> von vorn.

Aufruf: python ur5e_policy_client.py --prompt "pick up the cube"
"""
import argparse
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from openpi_client import websocket_client_policy
from PIL import Image
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage, JointState
from trajectory_msgs.msg import JointTrajectoryPoint

HOST, PORT = "localhost", 8000
EXEC_STEPS = 6      # so viele Aktionen pro Chunk ausführen
DT = 0.2            # s pro Aktion (Training: 10 Hz, hier 2x langsamer)
MAX_STEP = 0.06     # max. Gelenkänderung pro Aktion in rad
GRIP_OPEN, GRIP_CLOSED, GRIP_EFFORT = 0.098, 0.79, 20.0
ARM = ["shoulder_pan_joint", 
       "shoulder_lift_joint", 
       "elbow_joint",
       "wrist_1_joint", 
       "wrist_2_joint", 
       "wrist_3_joint"]

latest = {}  # neueste ROS-Nachrichten: "joints", "base", "wrist"


def to_rgb(msg):
    img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    img = img[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    return (img[:, :, ::-1] if msg.encoding.lower() == "bgr8" else img).copy()


def resize_with_pad(img, size=224):
    h, w = img.shape[:2]
    s = size / max(h, w)
    small = Image.fromarray(img).resize((round(w * s), round(h * s)), Image.BILINEAR)
    out = Image.new("RGB", (size, size))
    out.paste(small, ((size - small.width) // 2, (size - small.height) // 2))
    return np.asarray(out)


def read_state():
    """Armgelenke (Reihenfolge wie ARM), Greifer 0..1 und die echten Gelenknamen."""
    j = latest["joints"]
    names = [next(n for n in j.name if n.endswith(a)) for a in ARM]
    pos = dict(zip(j.name, j.position))
    graw = next(v for n, v in pos.items() if any(h in n.lower() for h in ("knuckle", "finger", "gripper")))
    g = np.clip((graw - GRIP_OPEN) / (GRIP_CLOSED - GRIP_OPEN), 0.0, 1.0)
    return np.array([pos[n] for n in names]), g, names


def wait(fut):
    while not fut.done():
        time.sleep(0.01)
    return fut.result()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    prompt = p.parse_args().prompt

    rclpy.init()
    node = rclpy.create_node("ur5e_policy_client")
    node.create_subscription(JointState, "/joint_states", lambda m: latest.__setitem__("joints", m), 10)

    for key, topic in (("base", "/camera/D415/color/image_raw"), ("wrist", "/camera/D405/color/image_raw")):
        node.create_subscription(RosImage, topic, lambda m, k=key: latest.__setitem__(k, m), qos_profile_sensor_data)

    traj = ActionClient(node, FollowJointTrajectory, "/scaled_joint_trajectory_controller/follow_joint_trajectory")
    grip = ActionClient(node, GripperCommand, "/robotiq_gripper_controller/gripper_cmd")

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    threading.Thread(target=executor.spin, daemon=True).start()

    print("Warte auf Controller und Daten ...", flush=True)

    for name, client in (("Trajektorien-Controller", traj), ("Greifer-Controller", grip)):
        while not client.wait_for_server(timeout_sec=2.0):
            print(f"  ... {name} noch nicht erreichbar", flush=True)
        print(f"  {name} ok", flush=True)
    while len(latest) < 3:
        missing = [k for k in ("joints", "base", "wrist") if k not in latest]
        print(f"  ... keine Daten von: {', '.join(missing)}", flush=True)
        time.sleep(2.0)

    print("  Daten ok. Verbinde mit Policy-Server ...", flush=True)
    policy = websocket_client_policy.WebsocketClientPolicy(HOST, PORT)
    print("Verbunden. Erste Inferenz (Kompilieren) kann einige Minuten dauern ...", flush=True)

    g_last = None
    try:
        while True:
            # ---- 1. Beobachtung ----
            q, g, names = read_state()
            obs = {
                "base_rgb":  resize_with_pad(to_rgb(latest["base"])),
                "wrist_rgb": resize_with_pad(to_rgb(latest["wrist"])),
                "joints":    q.astype(np.float32),
                "gripper":   np.array([g], dtype=np.float32),
                "prompt":    prompt,
            }

            # ---- 2. Inferenz (Arm fährt währenddessen den alten Plan weiter) ----
            t0 = time.time()
            chunk = np.asarray(policy.infer(obs)["actions"])
            latency = time.time() - t0

            # ---- 3. Schon gefahrene Aktionen überspringen ----
            q = read_state()[0]
            half = max(1, len(chunk) // 2)
            skip = int(np.argmin(np.linalg.norm(chunk[:half, :6] - q, axis=1))) + 1
            chunk = chunk[skip:]

            # ---- 4. Gelenkziele: jede Aktion max. MAX_STEP vom vorherigen Ziel ----
            targets, prev = [], q
            for a in chunk:
                prev = prev + np.clip(a[:6] - prev, -MAX_STEP, MAX_STEP)
                targets.append(prev)
            vel = np.gradient(np.vstack([q, targets]), DT, axis=0)[1:]
            vel[-1] = 0.0  # am Ende stehen bleiben

            # ---- 5. Trajektorie senden (ersetzt den laufenden Plan) ----
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = names
            for k, (pos, v) in enumerate(zip(targets, vel), start=1):
                t = k * DT
                goal.trajectory.points.append(JointTrajectoryPoint(
                    positions=pos.tolist(),
                    velocities=v.tolist(),
                    time_from_start=Duration(sec=int(t), nanosec=int(t % 1 * 1e9)),
                ))
            handle = wait(traj.send_goal_async(goal))
            if not handle.accepted:
                print("Trajektorie abgelehnt – Abbruch.")
                break

            # ---- 6. Greifer: Wert bei Aktion EXEC_STEPS, auf 0/1 gerundet, nur bei Änderung ----
            g_new = 1.0 if chunk[min(EXEC_STEPS, len(chunk)) - 1, 6] > 0.5 else 0.0
            if g_new != g_last:
                gg = GripperCommand.Goal()
                gg.command.position = GRIP_OPEN + g_new * (GRIP_CLOSED - GRIP_OPEN)
                gg.command.max_effort = GRIP_EFFORT
                grip.send_goal_async(gg)
                g_last = g_new

            print(f"Inferenz {latency * 1000:4.0f} ms | Start bei Aktion {skip} | Greifer {g_new:.0f}")

            # ---- 7. Warten, bis neu geplant werden muss ----
            time.sleep(max(0.0, EXEC_STEPS * DT - latency))

    except KeyboardInterrupt:
        print("\nAbgebrochen.")

    finally:
        # Arm anhalten, sonst fährt er den Rest des Chunks zu Ende
        q, _, names = read_state()
        stop = FollowJointTrajectory.Goal()
        stop.trajectory.joint_names = names
        stop.trajectory.points = [JointTrajectoryPoint(
            positions=q.tolist(),
            time_from_start=Duration(sec=0, nanosec=500_000_000),
        )]
        wait(traj.send_goal_async(stop))
        executor.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()