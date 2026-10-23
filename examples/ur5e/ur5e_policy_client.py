"""
ur5e_policy_client.py – verbindet den openpi-Policy-Server mit dem UR5e (robot_table).

Ablauf pro Durchgang:
  1. letzte Daten aus ROS holen: /joint_states, Basis- und Handgelenkkamera
  2. als Beobachtung an den Policy-Server schicken (gleiches Format wie im Training)
  3. Aktions-Chunk zurückbekommen: absolute Gelenkziele + Greifer (0..1)
  4. Sicherheitsgrenzen anwenden und – je nach Modus – anzeigen oder ausführen

Modi:
  --mode dry    nur anzeigen, der Arm bewegt sich NICHT (immer zuerst!)
  --mode step   vor jedem Chunk anzeigen und mit Enter bestätigen
  --mode auto   läuft selbstständig, bis --max-chunks erreicht ist (Arm hält zwischen Chunks kurz an)
  --mode stream wie auto, aber OHNE Anhalten: der nächste Chunk wird berechnet, während der Arm
                noch fährt, und nahtlos übergeben. Funktioniert mit jeder Speed-Slider-Stellung.
                Empfohlen: --slowdown 1 --exec-steps 8
  --mode jointtest  ohne Modell: jedes Gelenk einzeln +0.1 rad und zurück (Anbindung prüfen)

Sicherheitsgrenzen (alle per Kommandozeile änderbar):
  --exec-steps    nur die ersten N Aktionen eines Chunks ausführen (Standard 3)
  --max-step      max. Gelenkänderung pro Aktion in rad (Standard 0.03)
  --max-total     max. Abweichung von der Startpose in rad (Standard 0.3)
  --slowdown      Faktor auf die Zeit pro Aktion (Standard 3 -> 0.3 s statt 0.1 s)
  --no-gripper    Greiferbefehle nie senden

Voraussetzungen:
  - Roboter läuft (bringup_real_robot.launch.py), scaled_joint_trajectory_controller aktiv
  - beide Kameras laufen
  - Policy-Server läuft (uv run scripts/serve_policy.py ...)
  - Python mit rclpy UND openpi-client (siehe Anleitung)
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import pathlib
import sys
import threading
import time

import numpy as np
from PIL import Image

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_HINTS = ("knuckle", "finger", "gripper")


# =============================================================================
# Reine Hilfsfunktionen (ohne ROS, einzeln testbar)
# =============================================================================


def ros_image_to_rgb(msg) -> np.ndarray:
    """sensor_msgs/Image (rgb8/bgr8) oder sensor_msgs/CompressedImage -> RGB uint8 (H, W, 3)."""
    if hasattr(msg, "format"):
        return np.asarray(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB"))
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    img = data[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding.lower() == "bgr8":
        img = img[:, :, ::-1]
    elif msg.encoding.lower() != "rgb8":
        raise ValueError(f"Bild-Encoding {msg.encoding} nicht unterstützt (nur rgb8/bgr8).")
    return img.copy()


def resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Exakt wie im Konverter: Seitenverhältnis behalten, schwarz auffüllen."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    small = Image.fromarray(img).resize((round(w * s), round(h * s)), Image.BILINEAR)
    out = Image.new("RGB", (size, size))
    out.paste(small, ((size - small.width) // 2, (size - small.height) // 2))
    return np.asarray(out)


def joint_indices(names: list[str]) -> tuple[list[int], int]:
    arm = []
    for joint in ARM_JOINTS:
        hits = [i for i, n in enumerate(names) if n.endswith(joint)]
        if len(hits) != 1:
            raise ValueError(f"Gelenk '{joint}' nicht eindeutig in {names}")
        arm.append(hits[0])
    grip = [i for i, n in enumerate(names) if any(h in n.lower() for h in GRIPPER_HINTS)]
    if not grip:
        raise ValueError(f"Kein Greifergelenk in /joint_states: {names}")
    return arm, grip[0]


@dataclasses.dataclass
class Limits:
    max_step: float      # rad pro Aktion
    max_total: float     # rad Abweichung von der Startpose
    exec_steps: int      # wie viele Aktionen eines Chunks


def make_safe_targets(actions: np.ndarray, current: np.ndarray, start: np.ndarray, lim: Limits):
    """Aus dem Modell-Chunk (H, 7) sichere Gelenkziele (N, 6) + Greiferziel machen.

    - nur die ersten lim.exec_steps Aktionen
    - jede Aktion höchstens lim.max_step rad vom vorherigen Ziel entfernt
    - nie weiter als lim.max_total rad von der Startpose
    Gibt zusätzlich zurück, ob begrenzt werden musste.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"Unerwartete Aktionsform {actions.shape}, erwartet (H, 7)")
    chunk = actions[: lim.exec_steps]
    targets, prev, clipped = [], current.copy(), False
    for a in chunk:
        step = np.clip(a[:6] - prev, -lim.max_step, lim.max_step)
        nxt = np.clip(prev + step, start - lim.max_total, start + lim.max_total)
        clipped |= not np.allclose(nxt, a[:6], atol=1e-6)
        targets.append(nxt)
        prev = nxt
    gripper = float(np.clip(chunk[-1, 6], 0.0, 1.0))
    return np.array(targets), gripper, clipped


def take_chunk(actions: np.ndarray, skip: int, n: int) -> np.ndarray:
    """Die nächsten n Aktionen ab Index skip (Aktionen, deren Zeitpunkt wegen der Inferenzzeit
    schon vorbei ist, werden übersprungen). Mindestens die letzte Aktion bleibt übrig."""
    actions = np.asarray(actions)
    skip = min(max(skip, 0), len(actions) - 1)
    return actions[skip : skip + n]


def clip_reasons(chunk: np.ndarray, targets: np.ndarray, start: np.ndarray, lim: Limits) -> str:
    """Welche Gelenke wurden begrenzt und warum: 'Gesamt' = --max-total erreicht, 'Schritt' = --max-step."""
    names = ["pan", "lift", "elbow", "w1", "w2", "w3"]
    out = []
    for i, n in enumerate(names):
        if np.allclose(chunk[: len(targets), i], targets[:, i], atol=1e-6):
            continue
        at_total = np.any(np.abs(targets[:, i] - start[i]) >= lim.max_total - 1e-6)
        out.append(f"{n}({'Gesamt' if at_total else 'Schritt'})")
    return ", ".join(out)


def gripper_to_raw(g: float, g_open: float, g_closed: float) -> float:
    return g_open + float(np.clip(g, 0.0, 1.0)) * (g_closed - g_open)


def gripper_from_raw(raw: float, g_open: float, g_closed: float) -> float:
    return float(np.clip((raw - g_open) / (g_closed - g_open), 0.0, 1.0))


def format_table(current: np.ndarray, targets: np.ndarray, g_now: float, g_target: float) -> str:
    rows = [f"  {'Gelenk':22s} {'jetzt':>8s} {'Ziel(end)':>10s} {'Δ':>8s}"]
    for i, name in enumerate(ARM_JOINTS):
        d = targets[-1, i] - current[i]
        rows.append(f"  {name:22s} {current[i]:8.3f} {targets[-1, i]:10.3f} {d:+8.3f}")
    rows.append(f"  {'gripper (0=offen)':22s} {g_now:8.2f} {g_target:10.2f} {g_target - g_now:+8.2f}")
    return "\n".join(rows)


def import_openpi_client():
    """openpi_client laden; falls nicht installiert, direkt aus diesem Repo (packages/openpi-client/src)."""
    try:
        from openpi_client import websocket_client_policy
    except ModuleNotFoundError as e:
        if e.name != "openpi_client":
            raise SystemExit(f"openpi_client braucht das Paket '{e.name}', das hier fehlt. "
                             "Siehe Anleitung: eigene Umgebung mit openpi-client anlegen.") from e
        src = pathlib.Path(__file__).resolve().parents[2] / "packages" / "openpi-client" / "src"
        sys.path.insert(0, str(src))
        try:
            from openpi_client import websocket_client_policy
        except ModuleNotFoundError as e2:
            raise SystemExit(f"openpi_client gefunden ({src}), aber das Paket '{e2.name}' fehlt. "
                             "Siehe Anleitung: eigene Umgebung mit openpi-client anlegen.") from e2
    return websocket_client_policy


# =============================================================================
# ROS-Node
# =============================================================================


def run(args):
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory, GripperCommand
    from sensor_msgs.msg import CompressedImage, Image as RosImage, JointState
    from trajectory_msgs.msg import JointTrajectoryPoint

    class PolicyClient(Node):
        def __init__(self):
            super().__init__("ur5e_policy_client")
            self.lock = threading.Lock()
            self.joint_msg = None
            self.images = {}
            self.create_subscription(JointState, args.joint_topic, self._on_joints, 10)
            for key, topic in (("base", args.base_cam_topic), ("wrist", args.wrist_cam_topic)):
                mtype = CompressedImage if topic.endswith("/compressed") else RosImage
                self.create_subscription(mtype, topic, lambda m, k=key: self._on_image(k, m), qos_profile_sensor_data)
            self.traj = ActionClient(self, FollowJointTrajectory, args.trajectory_action)
            self.grip = ActionClient(self, GripperCommand, args.gripper_action)

        def _on_joints(self, msg):
            with self.lock:
                self.joint_msg = msg

        def _on_image(self, key, msg):
            with self.lock:
                self.images[key] = msg

        def snapshot(self):
            with self.lock:
                return self.joint_msg, self.images.get("base"), self.images.get("wrist")

    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    node = PolicyClient()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    log = node.get_logger()

    def wait_for_data(timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            j, b, w = node.snapshot()
            if j is not None and b is not None and w is not None:
                return j, b, w
            time.sleep(0.1)
        j, b, w = node.snapshot()
        missing = [n for n, v in (("joint_states", j), ("Basiskamera", b), ("Handgelenkkamera", w)) if v is None]
        raise RuntimeError(f"Keine Daten von: {', '.join(missing)} – Topics prüfen.")

    def read_state(jmsg):
        arm_idx, g_idx = joint_indices(list(jmsg.name))
        pos = np.asarray(jmsg.position, dtype=np.float64)
        return pos[arm_idx], gripper_from_raw(pos[g_idx], args.gripper_open, args.gripper_closed), \
            [jmsg.name[i] for i in arm_idx]

    print("Warte auf Roboter- und Kameradaten ...")
    jmsg, _, _ = wait_for_data()
    start_q, _, ros_joint_names = read_state(jmsg)
    print("Startpose (rad):", np.round(start_q, 3))

    def make_goal(targets, dt, q_start=None):
        """Trajektorie aus Gelenkzielen. Mit q_start werden zusätzlich Geschwindigkeiten gesetzt
        (zentrale Differenzen, letzter Punkt 0) -> der Controller fährt glatte Kurven statt
        an jedem Punkt abzubremsen, und Übergänge zwischen Chunks bleiben ruckfrei."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ros_joint_names
        targets = np.asarray(targets, dtype=np.float64)
        vel = None
        if q_start is not None:
            pts = np.vstack([q_start, targets])
            vel = np.zeros_like(targets)
            for k in range(len(targets) - 1):
                vel[k] = (pts[k + 2] - pts[k]) / (2 * dt)
        for k, tq in enumerate(targets, start=1):
            p = JointTrajectoryPoint()
            p.positions = [float(x) for x in tq]
            if vel is not None:
                p.velocities = [float(v) for v in vel[k - 1]]
            t = k * dt
            p.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            goal.trajectory.points.append(p)
        return goal

    def send_trajectory(targets, dt):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ros_joint_names
        for k, tq in enumerate(targets, start=1):
            p = JointTrajectoryPoint()
            p.positions = [float(x) for x in tq]
            t = k * dt
            p.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            goal.trajectory.points.append(p)
        fut = node.traj.send_goal_async(goal)
        while not fut.done():
            time.sleep(0.01)
        handle = fut.result()
        if not handle.accepted:
            return False
        res_fut = handle.get_result_async()
        while not res_fut.done():
            time.sleep(0.01)
        return True

    if args.mode != "dry":
        print(f"Warte auf {args.trajectory_action} ...")
        if not node.traj.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Trajektorien-Controller nicht erreichbar. Ist scaled_joint_trajectory_controller aktiv?")
        if not args.no_gripper and not node.grip.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Greifer-Controller nicht erreichbar (oder --no-gripper verwenden).")

    if args.mode == "jointtest":
        # Ohne Modell: jedes Gelenk einzeln um +delta und zurück. Prüft Reihenfolge, Richtung, Controller.
        delta = min(args.test_delta, args.max_total)
        print(f"\nGelenktest: jedes Gelenk +{delta:.2f} rad ({np.degrees(delta):.0f}°) und zurück, je 2 s.")
        try:
            for i, name in enumerate(ARM_JOINTS):
                ans = input(f"\n[{i + 1}/6] {name} bewegen? [Enter = ja, s = überspringen, q = beenden] ").strip().lower()
                if ans == "q":
                    break
                if ans == "s":
                    continue
                jmsg, _, _ = wait_for_data()
                q, _, _ = read_state(jmsg)
                tgt = q.copy()
                tgt[i] += delta
                ok = send_trajectory([tgt], 2.0)
                jmsg, _, _ = wait_for_data()
                q_after, _, _ = read_state(jmsg)
                moved = q_after - q
                print(f"  Soll: {name} +{delta:.3f} | Ist-Änderung:",
                      " ".join(f"{n.split('_joint')[0]}={d:+.3f}" for n, d in zip(ARM_JOINTS, moved)),
                      "" if ok else " [ABGELEHNT]")
                send_trajectory([q], 2.0)
        except KeyboardInterrupt:
            print("\nAbgebrochen.")
        finally:
            rclpy.shutdown()
        return

    websocket_client_policy = import_openpi_client()
    print(f"Verbinde mit Policy-Server {args.host}:{args.port} ...")
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("Verbunden. Prompt:", repr(args.prompt))

    lim = Limits(args.max_step, args.max_total, args.exec_steps)
    dt = args.slowdown / args.fps
    last_gripper_sent = None

    def get_obs():
        jmsg, bmsg, wmsg = wait_for_data()
        q, g_now, _ = read_state(jmsg)
        obs = {
            "base_rgb": resize_with_pad(ros_image_to_rgb(bmsg), args.image_size),
            "wrist_rgb": resize_with_pad(ros_image_to_rgb(wmsg), args.image_size),
            "joints": q.astype(np.float32),
            "gripper": np.array([g_now], dtype=np.float32),
            "prompt": args.prompt,
        }
        return obs, q, g_now

    def send_gripper(g_target):
        nonlocal last_gripper_sent
        if args.no_gripper or (last_gripper_sent is not None and abs(g_target - last_gripper_sent) <= 0.1):
            return
        gg = GripperCommand.Goal()
        gg.command.position = gripper_to_raw(g_target, args.gripper_open, args.gripper_closed)
        gg.command.max_effort = args.gripper_effort
        node.grip.send_goal_async(gg)
        last_gripper_sent = g_target

    if args.mode == "stream":
        # Ablauf: Chunk senden -> warten, bis der Arm (gemessen an seiner IST-Position) fast am Ende
        # ist -> neues Bild, Modell fragen -> neuen Chunk senden, der den alten nahtlos ersetzt.
        # Weil der Fortschritt an der Ist-Position gemessen wird statt an der Uhr, passt das auch
        # bei niedrigem Speed-Slider. Neue Chunks starten bei der Aktion, die der Ist-Position am
        # nächsten liegt (statt einer geschätzten Zeit), damit nichts vor- oder zurückspringt.
        print("Kompiliere Modell (erster Aufruf dauert einige Sekunden) ...")
        obs, _, _ = get_obs()
        t0 = time.time()
        policy.infer(obs)
        print(f"  fertig nach {time.time() - t0:.1f} s")
        print(f"Stream-Modus: bis zu {args.exec_steps} Aktionen à {dt:.2f} s pro Chunk, Neuplanung ohne Anhalten.")

        latency_est = 0.3
        goal_fut, result_fut = None, None

        def current_q():
            j, _, _ = node.snapshot()
            return read_state(j)[0]

        def goal_finished():
            nonlocal result_fut
            if goal_fut is None or not goal_fut.done():
                return False
            handle = goal_fut.result()
            if not handle.accepted:
                return True
            if result_fut is None:
                result_fut = handle.get_result_async()
            return result_fut.done()

        try:
            for chunk_no in range(1, args.max_chunks + 1):
                obs, _, _ = get_obs()
                t_obs = time.time()
                actions = np.asarray(policy.infer(obs)["actions"])
                latency = time.time() - t_obs
                latency_est = 0.7 * latency_est + 0.3 * latency

                if goal_fut is not None and goal_fut.done() and not goal_fut.result().accepted:
                    print("Trajektorie vom Controller abgelehnt – Abbruch.")
                    break

                # Startpunkt im neuen Chunk: Aktion, die der aktuellen Ist-Position am nächsten ist.
                q_now = current_q()
                search = actions[: max(1, len(actions) // 2), :6]
                skip = int(np.argmin(np.linalg.norm(search - q_now, axis=1))) + 1
                chunk = take_chunk(actions, skip, args.exec_steps)
                targets, g_target, clipped = make_safe_targets(
                    chunk, q_now, start_q, Limits(lim.max_step, lim.max_total, len(chunk)))

                goal_fut, result_fut = node.traj.send_goal_async(make_goal(targets, dt, q_start=q_now)), None
                send_gripper(g_target)
                why = clip_reasons(np.asarray(chunk, dtype=np.float64), targets, start_q, lim) if clipped else ""
                print(f"Chunk {chunk_no:3d}: Inferenz {latency * 1000:4.0f} ms, Start bei Aktion {skip}, "
                      f"führe {len(targets)} aus, Greifer {g_target:.2f}"
                      f"{'  [BEGRENZT: ' + why + ']' if clipped else ''}")

                # Warten, bis der Arm so weit ist, dass die nächste Antwort rechtzeitig vor dem Ende kommt.
                lead = int(np.ceil(latency_est / dt)) + 1
                trigger = max(0, len(targets) - 1 - lead)
                deadline = time.time() + 20 * len(targets) * dt  # Notbremse, falls der Arm nicht vorankommt
                while time.time() < deadline:
                    reached = int(np.argmin(np.linalg.norm(targets - current_q(), axis=1)))
                    if reached >= trigger or goal_finished():
                        break
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nAbgebrochen (Strg+C).")
        finally:
            # Arm sanft an der aktuellen Position anhalten, dann ROS sauber beenden.
            try:
                q_stop = current_q()
                node.traj.send_goal_async(make_goal([q_stop], 0.5))
                time.sleep(0.6)
            except Exception:
                pass
            executor.shutdown()
            rclpy.shutdown()
        return

    try:
        for chunk_no in range(1, args.max_chunks + 1):
            jmsg, bmsg, wmsg = wait_for_data()
            q, g_now, _ = read_state(jmsg)
            obs = {
                "base_rgb": resize_with_pad(ros_image_to_rgb(bmsg), args.image_size),
                "wrist_rgb": resize_with_pad(ros_image_to_rgb(wmsg), args.image_size),
                "joints": q.astype(np.float32),
                "gripper": np.array([g_now], dtype=np.float32),
                "prompt": args.prompt,
            }
            t0 = time.time()
            result = policy.infer(obs)
            latency = time.time() - t0
            actions = np.asarray(result["actions"])
            targets, g_target, clipped = make_safe_targets(actions, q, start_q, lim)

            print(f"\n=== Chunk {chunk_no}: Inferenz {latency * 1000:.0f} ms, Modell-Chunk {actions.shape}, "
                  f"führe {len(targets)} Aktionen in {len(targets) * dt:.1f} s aus"
                  f"{'  [BEGRENZT]' if clipped else ''}")
            print(format_table(q, targets, g_now, g_target))

            if args.mode == "dry":
                time.sleep(args.dry_pause)
                continue
            if args.mode == "step":
                ans = input("Ausführen? [Enter = ja, s = überspringen, q = beenden] ").strip().lower()
                if ans == "q":
                    break
                if ans == "s":
                    continue

            if not send_trajectory(targets, dt):
                print("Trajektorie vom Controller abgelehnt – Abbruch.")
                break

            if not args.no_gripper and (last_gripper_sent is None or abs(g_target - last_gripper_sent) > 0.1):
                gg = GripperCommand.Goal()
                gg.command.position = gripper_to_raw(g_target, args.gripper_open, args.gripper_closed)
                gg.command.max_effort = args.gripper_effort
                node.grip.send_goal_async(gg)
                last_gripper_sent = g_target
    except KeyboardInterrupt:
        print("\nAbgebrochen (Strg+C).")
    finally:
        rclpy.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["dry", "step", "auto", "stream", "jointtest"], default="dry")
    p.add_argument("--prompt", default="", help="Sprachanweisung (nicht nötig bei --mode jointtest)")
    p.add_argument("--test-delta", type=float, default=0.1, help="Gelenktest: Auslenkung in rad")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--joint-topic", default="/joint_states")
    p.add_argument("--base-cam-topic", default="/camera/D415/color/image_raw")
    p.add_argument("--wrist-cam-topic", default="/camera/D405/color/image_raw")
    p.add_argument("--trajectory-action", default="/scaled_joint_trajectory_controller/follow_joint_trajectory")
    p.add_argument("--gripper-action", default="/robotiq_gripper_controller/gripper_cmd")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--fps", type=float, default=10.0, help="Frequenz des Trainingsdatensatzes")
    p.add_argument("--gripper-open", type=float, default=0.098)
    p.add_argument("--gripper-closed", type=float, default=0.79)  # wie rosbag_to_lerobot.py
    p.add_argument("--gripper-effort", type=float, default=20.0)
    p.add_argument("--no-gripper", action="store_true")
    p.add_argument("--exec-steps", type=int, default=3)
    p.add_argument("--max-step", type=float, default=0.03)
    p.add_argument("--max-total", type=float, default=0.3)
    p.add_argument("--slowdown", type=float, default=3.0)
    p.add_argument("--max-chunks", type=int, default=20)
    p.add_argument("--dry-pause", type=float, default=1.0, help="Pause zwischen Chunks im dry-Modus [s]")
    run(p.parse_args())


if __name__ == "__main__":
    main()