"""policy_client_pi0_fast.py – minimale Ansteuerung des UR5e mit dem openpi-Policy-Server.

Schleife: Gelenke + 2 Kamerabilder lesen -> an den Server schicken -> die ersten EXEC_STEPS
Aktionen als Trajektorie fahren -> Greifer setzen -> von vorn.

Greifer: stufenlose Position wie im Datensatz (rohe Greiferwerte).
  - Schließen: dem Plan folgen – der höchste Wert in den nächsten GRIP_LOOKAHEAD Aktionen wird
    geschickt, auch kleine Schritte. Das Modell schließt oft schrittweise (0.05 -> 0.1 -> ...),
    weil es die gemessene Position als Eingabe bekommt; eine Schwelle würde das blockieren.
  - Öffnen: nur bei deutlicher Absicht (GRIP_OPEN_INTENT), dann aber sofort und ganz. Kleine
    Schwankungen nach unten würden sonst den Griff während des Transports lockern.
  - Nachdrücken (GRIP_SQUEEZE) nur, wenn der Greifer am Objekt anliegt (erreicht den letzten
    Befehl nicht). Sonst schaukelt es sich auf.
Tisch-Schutz (der Client läuft dabei weiter):
  1. Vorher: Für jeden geplanten Punkt wird per Vorwärtskinematik die TCP-Höhe berechnet. Die
     Trajektorie wird vor dem ersten Punkt unter TABLE_Z_MIN abgeschnitten – der Arm bleibt an der
     Grenze stehen, seitlich/nach oben und der Greifer funktionieren weiter.
  2. Notbremse: Fällt die gemessene TCP-Höhe trotzdem unter TABLE_Z_MIN - 5 mm, wird der Arm angehalten.
TABLE_Z_MIN einmal einmessen (siehe unten).
Kameras: komprimierte Topics wie bei der Aufnahme, damit das Modell dieselben (JPEG-)Bilder sieht
wie im Training.

Aufruf: python policy_client_pi0_fast.py --prompt "pick up the cube"
"""
import argparse
import io
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
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, JointState
from trajectory_msgs.msg import JointTrajectoryPoint

HOST, PORT = "localhost", 8000
EXEC_STEPS = 6      # so viele Aktionen pro Chunk ausführen
DT = 0.2            # s pro Aktion (Training: 10 Hz, hier 2x langsamer)
MAX_STEP = 0.06     # max. Gelenkänderung pro Aktion in rad
GRIP_OPEN, GRIP_CLOSED, GRIP_EFFORT = 0.098, 0.79, 20.0
GRIP_SQUEEZE = 0.02    # so viel weiter zu als vorhergesagt, sobald gegriffen wird (wie beim Teleop)
GRIP_MIN_CHANGE = 0.02 # neuen Greiferbefehl nur bei mindestens dieser Änderung schicken
GRIP_LOOKAHEAD = 6     # so viele Aktionen voraus nach einem geplanten Öffnen/Schließen suchen
GRIP_OPEN_INTENT = 0.08  # so weit unter der Messung muss der Plan liegen, damit geöffnet wird
GRIP_BLOCKED = 0.01    # Messung so viel unter dem letzten Befehl -> Greifer liegt am Objekt an
MAX_STATE_AGE = 0.2    # s: ältere Gelenkwerte gelten als veraltet -> warten statt planen
# Tisch-Schutz: minimale TCP-Höhe (Basis-Frame, m). Bewegungen darunter werden vorab abgeschnitten.
# Einmessen: Greiferfinger von Hand/Teach-Pendant knapp über die Tischplatte fahren, dann
#   ros2 topic echo /tcp_pose_broadcaster/pose --once     -> pose.position.z ablesen
# und TABLE_Z_MIN = dieser Wert + ca. 0.003 setzen (Bahn wird vorab begrenzt). None = Schutz aus (nicht empfohlen).
TABLE_Z_MIN = 0.013
TABLE_HARD_MARGIN = 0.005   # Notbremse so viel unter TABLE_Z_MIN
TCP_TOPIC = "/tcp_pose_broadcaster/pose"
CAMERAS = {"base": "/camera/D415/color/image_raw/compressed",
           "wrist": "/camera/D405/color/image_raw/compressed"}
ARM = ["shoulder_pan_joint", 
       "shoulder_lift_joint", 
       "elbow_joint",
       "wrist_1_joint", 
       "wrist_2_joint", 
       "wrist_3_joint"]

latest = {}       # neueste ROS-Nachrichten: "joints", "base", "wrist", "tcp"
received = {}     # Empfangszeit (time.time()) je Schlüssel


def store(key, msg):
    latest[key] = msg
    received[key] = time.time()


def to_rgb(msg):
    """CompressedImage (JPEG) -> RGB, genauso dekodiert wie im Konverter."""
    return np.asarray(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB"))


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


def gripper_command(g_chunk, g, g_last):
    """Greiferziel (0..1) aus den vorhergesagten Greiferwerten des Chunks.

    g_chunk: Greiferwerte der noch auszuführenden Aktionen, g: gemessene Position,
    g_last: zuletzt gesendeter Befehl (None am Anfang). Gibt (Ziel, Grund) zurück.
    """
    seq = np.clip(np.asarray(g_chunk[:GRIP_LOOKAHEAD], dtype=float), 0.0, 1.0)
    if len(seq) == 0:
        return (g_last if g_last is not None else g), "leer"
    if seq.min() < g - GRIP_OPEN_INTENT:                       # deutliches Öffnen geplant
        i = int(np.argmax(seq < g - GRIP_OPEN_INTENT))
        return float(seq.min()), f"öffnen (geplant bei Aktion {i + 1})"
    target = float(seq.max())                                   # Schließen/Halten: dem Plan folgen
    why = "schließen" if target > g + 0.01 else "halten"
    blocked = g_last is not None and (g_last - g) > GRIP_BLOCKED
    if target > 0.1 and blocked:
        return min(target + GRIP_SQUEEZE, 1.0), why + " + nachdrücken"
    return target, why


# UR5e, Standard-DH (a, d, alpha); nur die Höhe wird gebraucht (unabhängig von Drehung um z der Basis)
UR5E_DH = [(0.0, 0.1625, np.pi / 2), (-0.425, 0.0, 0.0), (-0.3922, 0.0, 0.0),
           (0.0, 0.1333, np.pi / 2), (0.0, 0.0997, -np.pi / 2), (0.0, 0.0996, 0.0)]


def flange_z_and_axis(q):
    """Höhe des Flanschs und z-Komponente der Werkzeugachse (Basis-Frame) für Gelenkwinkel q."""
    T = np.eye(4)
    for (a, d, al), th in zip(UR5E_DH, q):
        ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st],
                          [0, sa, ca, d], [0, 0, 0, 1]])
    return T[2, 3], T[2, 2]


class TcpHeight:
    """TCP-Höhe aus Gelenkwinkeln. Der Abstand Flansch->TCP entlang der Werkzeugachse wird beim Start
    aus der gemessenen TCP-Pose bestimmt (TCP-Offset vom Teach-Pendant)."""

    def __init__(self, q, tcp_z):
        fz, az = flange_z_and_axis(q)
        if abs(az) < 0.5:
            raise SystemExit("Zum Start den Greifer ungefähr senkrecht nach unten richten (für die TCP-Kalibrierung).")
        self.offset = (tcp_z - fz) / az
        print(f"  TCP-Abstand zum Flansch: {self.offset * 1000:.0f} mm", flush=True)

    def __call__(self, q):
        fz, az = flange_z_and_axis(q)
        return fz + self.offset * az


def hold_position(traj):
    """Arm sofort an der aktuellen Position anhalten."""
    q, _, names = read_state()
    stop = FollowJointTrajectory.Goal()
    stop.trajectory.joint_names = names
    stop.trajectory.points = [JointTrajectoryPoint(positions=q.tolist(),
                                                   time_from_start=Duration(sec=0, nanosec=200_000_000))]
    return traj.send_goal_async(stop)


def table_guard(traj, stop_event):
    """Notbremse (eigener Thread): gemessene TCP-Höhe unter TABLE_Z_MIN - TABLE_HARD_MARGIN -> anhalten."""
    last = 0.0
    while not stop_event.is_set():
        tcp = latest.get("tcp")
        if tcp is not None and tcp.pose.position.z < TABLE_Z_MIN - TABLE_HARD_MARGIN and time.time() - last > 0.5:
            hold_position(traj)
            print(f"!!! Notbremse: TCP bei z = {tcp.pose.position.z:.3f} m – Arm angehalten", flush=True)
            last = time.time()
        time.sleep(0.02)


def wait(fut):
    while not fut.done():
        time.sleep(0.01)
    return fut.result()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    prompt = p.parse_args().prompt

    rclpy.init()
    node = rclpy.create_node("policy_client_pi0_fast")
    node.create_subscription(JointState, "/joint_states", lambda m: store("joints", m), 10)

    for key, topic in CAMERAS.items():
        node.create_subscription(CompressedImage, topic, lambda m, k=key: store(k, m), qos_profile_sensor_data)
    node.create_subscription(PoseStamped, TCP_TOPIC, lambda m: store("tcp", m), 10)

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
    needed = ("joints", "base", "wrist") + (("tcp",) if TABLE_Z_MIN is not None else ())
    while not all(k in latest for k in needed):
        missing = [k for k in needed if k not in latest]
        print(f"  ... keine Daten von: {', '.join(missing)}", flush=True)
        time.sleep(2.0)

    stop_event = threading.Event()
    if TABLE_Z_MIN is None:
        print("  ⚠ Tisch-Schutz AUS (TABLE_Z_MIN = None) – Hand am Not-Aus!", flush=True)
    else:
        z = latest["tcp"].pose.position.z
        if z < TABLE_Z_MIN:
            raise SystemExit(f"Flansch steht schon unter TABLE_Z_MIN ({z:.3f} < {TABLE_Z_MIN:.3f} m) – erst hochfahren.")
        tcp_height = TcpHeight(read_state()[0], z)
        threading.Thread(target=table_guard, args=(traj, stop_event), daemon=True).start()
        print(f"  Tisch-Schutz aktiv: nicht unter z = {TABLE_Z_MIN:.3f} m (jetzt {z:.3f} m)", flush=True)

    print("  Daten ok. Verbinde mit Policy-Server ...", flush=True)
    policy = websocket_client_policy.WebsocketClientPolicy(HOST, PORT)
    print("Verbunden. Erste Inferenz (Kompilieren) kann einige Minuten dauern ...", flush=True)

    g_last = None
    try:
        while not stop_event.is_set():
            # ---- 1. Beobachtung (nur mit frischen Gelenkwerten) ----
            age = time.time() - received["joints"]
            if age > MAX_STATE_AGE:
                print(f"Gelenkwerte {age * 1000:.0f} ms alt – warte ...", flush=True)
                time.sleep(0.05)
                continue
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
            result = policy.infer(obs)
            latency = time.time() - t0
            if not result.get("decode_ok", True):
                # pi0-FAST konnte die Tokens nicht dekodieren -> Chunk wäre Unsinn. Alten Plan weiterfahren.
                print(f"Inferenz {latency * 1000:4.0f} ms | Decoding fehlgeschlagen – Chunk verworfen", flush=True)
                continue
            chunk = np.asarray(result["actions"])

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

            # ---- 4b. Tisch-Schutz: vor dem ersten Punkt unter TABLE_Z_MIN abschneiden ----
            guard_note = ""
            if TABLE_Z_MIN is not None:
                zs = [tcp_height(t) for t in targets]
                low = next((i for i, zz in enumerate(zs) if zz < TABLE_Z_MIN), None)
                if low is not None:
                    targets = targets[:low]
                    guard_note = f" | Tisch: unten begrenzt (Plan bis z = {min(zs):.3f} m)"
                if abs(tcp_height(q) - latest["tcp"].pose.position.z) > 0.01:
                    guard_note += " | ⚠ TCP-Höhe passt nicht zur Messung"
            if not targets:
                wait(hold_position(traj))
            else:
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

            # ---- 6. Greifer ----
            g_new, why = gripper_command(chunk[:, 6], g, g_last)
            if g_last is None or abs(g_new - g_last) > GRIP_MIN_CHANGE:
                gg = GripperCommand.Goal()
                gg.command.position = GRIP_OPEN + g_new * (GRIP_CLOSED - GRIP_OPEN)
                gg.command.max_effort = GRIP_EFFORT
                grip.send_goal_async(gg)
                g_last = g_new

            gs = np.clip(chunk[:GRIP_LOOKAHEAD, 6], 0.0, 1.0)
            print(f"Inferenz {latency * 1000:4.0f} ms | Start bei Aktion {skip} | "
                  f"Greifer {g:.2f} -> {g_new:.2f} ({why}) | Plan {gs.min():.2f}..{gs.max():.2f}{guard_note}", flush=True)

            # ---- 7. Warten, bis neu geplant werden muss (bricht sofort ab, wenn der Tisch-Schutz auslöst) ----
            stop_event.wait(max(0.0, EXEC_STEPS * DT - latency))

    except KeyboardInterrupt:
        print("\nAbgebrochen.")

    finally:
        # Arm anhalten, sonst fährt er den Rest des Chunks zu Ende
        stop_event.set()
        wait(hold_position(traj))
        executor.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()