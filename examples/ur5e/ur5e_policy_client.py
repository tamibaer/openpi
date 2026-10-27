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
  --mode stream wie auto, aber OHNE Anhalten (empfohlen): Der Arm bekommt immer den ganzen Rest des
                Chunks als Trajektorie. Nach --exec-steps Aktionen wird neu geplant, der neue Chunk
                wird in den laufenden weich eingeblendet. Empfohlen: --slowdown 1 --exec-steps 5
  --mode jointtest  ohne Modell: jedes Gelenk einzeln +0.1 rad und zurück (Anbindung prüfen)

Sicherheitsgrenzen (alle per Kommandozeile änderbar):
  --exec-steps    nach so vielen Aktionen neu planen (auto/step: nur so viele ausführen, Standard 3)
  --max-step      max. Gelenkänderung pro Aktion in rad (Standard 0.03)
  --max-total     max. Abweichung von der Startpose in rad (Standard 0.3)
  --slowdown      Faktor auf die Zeit pro Aktion (Standard 3 -> 0.3 s statt 0.1 s)
  --no-gripper    Greiferbefehle nie senden

Glatte Bewegung (stream):
  --blend-steps   so viele Aktionen lang wird vom alten in den neuen Chunk übergeblendet (Standard 3)

Weiches Greifen (nur mit aktivem admittance_controller):
  Solange der Greifer schliesst, wird die Admittanz in z auf ein Greifprofil umgeschaltet (weich und
  leicht, mit fester absoluter Dämpfung). Die Finger drücken dann kaum auf den Tisch und schrammen
  nicht. Der Wechsel wird weich übergeblendet, damit der Arm nicht hochhüpft.
  --grasp-z-stiffness  z-Steifigkeit beim Schliessen in N/m (Standard 12, 0 = Funktion aus)
  --grasp-z-mass       virtuelle Masse in z beim Schliessen in kg (Standard 1.0)
  --grasp-z-damping    absolute Dämpfung in z beim Schliessen in Ns/m (Standard 200)
  --grasp-filter       Filterkoeffizient beim Schliessen (Standard 0.07, glättet Kraftspitzen
                       des schliessenden Greifers)
  --grasp-ramp         Überblenden ins Greifprofil in s (Standard 0.3)
  --grasp-settle       danach warten, bis der Arm die Anpresskraft abgebaut hat, erst dann
                       schliessen, in s (Standard 0.35)
  --grasp-ramp-back    Überblenden zurück ins Normalprofil in s (Standard 0.8)
  --grasp-hold         so lange nach Ende des Schliessens im Greifprofil bleiben, in s (Standard 0.3)
  --grasp-max          spätestens nach dieser Zeit zurück ins normale Profil, in s (Standard 3.0)

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


def arm_order(names: list[str]) -> list[int]:
    """Indizes der Armgelenke in der Reihenfolge von ARM_JOINTS (ohne Greifer)."""
    return joint_indices(list(names) + ["gripper"])[0]


@dataclasses.dataclass
class Limits:
    max_step: float      # rad pro Aktion
    max_total: float     # rad Abweichung von der Startpose


def make_safe_targets(actions: np.ndarray, current: np.ndarray, start: np.ndarray, lim: Limits):
    """Aus Modell-Aktionen (N, 7) sichere Gelenkziele (N, 6) machen.

    - jede Aktion höchstens lim.max_step rad vom vorherigen Ziel entfernt
    - nie weiter als lim.max_total rad von der Startpose
    Gibt die Ziele und zurück, ob begrenzt werden musste.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"Unerwartete Aktionsform {actions.shape}, erwartet (N, 7)")
    targets, prev, clipped = [], np.asarray(current, dtype=np.float64).copy(), False
    for a in actions:
        step = np.clip(a[:6] - prev, -lim.max_step, lim.max_step)
        nxt = np.clip(prev + step, start - lim.max_total, start + lim.max_total)
        clipped |= not np.allclose(nxt, a[:6], atol=1e-6)
        targets.append(nxt)
        prev = nxt
    return np.array(targets), clipped


def gripper_target(actions: np.ndarray, n: int) -> float:
    """Greiferziel: Wert der n-ten Aktion (Ende des ausgeführten Teils), auf 0..1 begrenzt."""
    actions = np.asarray(actions)
    return float(np.clip(actions[min(n, len(actions)) - 1, 6], 0.0, 1.0))


def nearest_action(actions: np.ndarray, q: np.ndarray) -> int:
    """Index der Aktion in der ersten Chunk-Hälfte, die q am nächsten liegt.

    Aktionen davor sind wegen der Inferenzzeit schon vorbei, der neue Chunk startet danach.
    """
    search = np.asarray(actions)[: max(1, len(actions) // 2), :6]
    return int(np.argmin(np.linalg.norm(search - q, axis=1)))


def joint_velocities(q_start: np.ndarray, targets: np.ndarray, dt: float, stop_at_end: bool = True) -> np.ndarray:
    """Geschwindigkeiten an den Zielpunkten (zentrale Differenzen über q_start und die Ziele).

    Damit interpoliert der Trajektorien-Controller kubisch statt linear: kein Abbremsen an
    jedem Punkt, stetige Geschwindigkeit.
    """
    pts = np.vstack([q_start, targets])
    vel = np.gradient(pts, dt, axis=0)[1:]
    if stop_at_end:
        vel[-1] = 0.0
    return vel


@dataclasses.dataclass
class Plan:
    """Zuletzt gesendete Trajektorie mit Startzeit, um neue Chunks daran anzuschliessen."""
    t_start: float           # time.monotonic() beim Senden
    dt: float
    q_start: np.ndarray      # Startpunkt (Zeit 0)
    targets: np.ndarray      # Punkte bei dt, 2*dt, ...

    def sample(self, t: float) -> np.ndarray | None:
        """Position des Plans zur Zeit t (linear zwischen den Punkten), None nach dem Ende."""
        s = (t - self.t_start) / self.dt
        if s < 0 or s > len(self.targets):
            return None
        pts = np.vstack([self.q_start, self.targets])
        i = min(int(np.floor(s)), len(pts) - 2)
        a = s - i
        return (1 - a) * pts[i] + a * pts[i + 1]


def blend_into_plan(new_targets: np.ndarray, old: Plan | None, t_start: float, dt: float, steps: int) -> np.ndarray:
    """Die ersten `steps` neuen Ziele weich aus dem alten Plan in den neuen Chunk überblenden.

    Gewicht des neuen Chunks steigt linear von 1/(steps+1) auf 1. Dort, wo der alte Plan schon
    zu Ende ist, werden die neuen Ziele unverändert übernommen.
    """
    out = np.array(new_targets, dtype=np.float64, copy=True)
    if old is None or steps <= 0:
        return out
    for i in range(min(steps, len(out))):
        old_q = old.sample(t_start + (i + 1) * dt)
        if old_q is None:
            break
        w = (i + 1) / (steps + 1)
        out[i] = (1 - w) * old_q + w * out[i]
    return out


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


def wait_future(fut, timeout: float | None = None):
    """Auf ein rclpy-Future warten, das vom Executor-Thread erfüllt wird. None bei Timeout."""
    t_end = None if timeout is None else time.time() + timeout
    while not fut.done():
        if t_end is not None and time.time() > t_end:
            return None
        time.sleep(0.005)
    return fut.result()


# =============================================================================
# ROS-Anbindung
# =============================================================================


class Robot:
    """Kapselt den ROS-Knoten: Zustand, Kameras, Trajektorie und Greifer."""

    def __init__(self, args):
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        from control_msgs.msg import JointTrajectoryControllerState
        from sensor_msgs.msg import CompressedImage, Image as RosImage, JointState

        self.args = args
        self._rclpy = rclpy
        self._FollowJointTrajectory = FollowJointTrajectory
        self._GripperCommand = GripperCommand
        try:
            from rclpy.signals import SignalHandlerOptions
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        except (ImportError, TypeError):
            rclpy.init()

        self.node = Node("ur5e_policy_client")
        self.lock = threading.Lock()
        self.joint_msg = None
        self.ref_msg = None
        self.images = {}
        n = self.node
        n.create_subscription(JointState, args.joint_topic, self._store("joint_msg"), 10)
        n.create_subscription(JointTrajectoryControllerState, args.controller_state_topic, self._store("ref_msg"), 10)
        for key, topic in (("base", args.base_cam_topic), ("wrist", args.wrist_cam_topic)):
            mtype = CompressedImage if topic.endswith("/compressed") else RosImage
            n.create_subscription(mtype, topic, lambda m, k=key: self._on_image(k, m), qos_profile_sensor_data)
        self.traj = ActionClient(n, FollowJointTrajectory, args.trajectory_action)
        self.grip = ActionClient(n, GripperCommand, args.gripper_action)

        self.executor = MultiThreadedExecutor()
        self.executor.add_node(n)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        self.joint_names = None

    def _store(self, attr):
        def cb(msg):
            with self.lock:
                setattr(self, attr, msg)
        return cb

    def _on_image(self, key, msg):
        with self.lock:
            self.images[key] = msg

    def shutdown(self):
        try:
            self.executor.shutdown()
        finally:
            self._rclpy.try_shutdown()

    # ---- Zustand --------------------------------------------------------- #

    def wait_for_data(self, timeout=10.0):
        t0 = time.time()
        while True:
            with self.lock:
                j, b, w = self.joint_msg, self.images.get("base"), self.images.get("wrist")
            if j is not None and b is not None and w is not None:
                return j, b, w
            if time.time() - t0 > timeout:
                missing = [n for n, v in (("joint_states", j), ("Basiskamera", b), ("Handgelenkkamera", w)) if v is None]
                raise RuntimeError(f"Keine Daten von: {', '.join(missing)} – Topics prüfen.")
            time.sleep(0.05)

    def read_state(self, jmsg):
        """Gemessene Armgelenke (ARM_JOINTS-Reihenfolge) und Greifer 0..1."""
        arm_idx, g_idx = joint_indices(list(jmsg.name))
        pos = np.asarray(jmsg.position, dtype=np.float64)
        if self.joint_names is None:
            self.joint_names = [jmsg.name[i] for i in arm_idx]
        g = gripper_from_raw(pos[g_idx], self.args.gripper_open, self.args.gripper_closed)
        return pos[arm_idx], g

    def measured_q(self):
        with self.lock:
            j = self.joint_msg
        return self.read_state(j)[0]

    def reference_q(self):
        """Aktuelle Referenz des Trajektorien-Controllers, ersatzweise die gemessene Position.

        Mit Admittanz weicht die gemessene Position (z. B. bei Tischkontakt) von der Referenz ab.
        Neue Trajektorien setzen deshalb an der Referenz an, sonst würde jeder Chunk den
        Ausweich-Offset der Admittanz übernehmen und der Arm würde wandern.
        """
        with self.lock:
            r = self.ref_msg
        if r is None or len(r.reference.positions) != len(r.joint_names):
            return self.measured_q()
        try:
            idx = arm_order(list(r.joint_names))
        except ValueError:
            return self.measured_q()
        return np.asarray(r.reference.positions, dtype=np.float64)[idx]

    def observation(self):
        """Beobachtung im Trainingsformat (gemessener Zustand, wie in den Demos)."""
        a = self.args
        jmsg, bmsg, wmsg = self.wait_for_data()
        q, g = self.read_state(jmsg)
        obs = {
            "base_rgb": resize_with_pad(ros_image_to_rgb(bmsg), a.image_size),
            "wrist_rgb": resize_with_pad(ros_image_to_rgb(wmsg), a.image_size),
            "joints": q.astype(np.float32),
            "gripper": np.array([g], dtype=np.float32),
            "prompt": a.prompt,
        }
        return obs, q, g

    # ---- Bewegung -------------------------------------------------------- #

    def wait_for_servers(self):
        print(f"Warte auf {self.args.trajectory_action} ...")
        if not self.traj.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Trajektorien-Controller nicht erreichbar. Ist scaled_joint_trajectory_controller aktiv?")
        if not self.args.no_gripper and not self.grip.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Greifer-Controller nicht erreichbar (oder --no-gripper verwenden).")

    def make_goal(self, targets, dt, q_start, stop_at_end=True):
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint

        goal = self._FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        targets = np.asarray(targets, dtype=np.float64)
        vel = joint_velocities(np.asarray(q_start, dtype=np.float64), targets, dt, stop_at_end)
        for k, (tq, v) in enumerate(zip(targets, vel), start=1):
            p = JointTrajectoryPoint()
            p.positions = [float(x) for x in tq]
            p.velocities = [float(x) for x in v]
            t = k * dt
            p.time_from_start = Duration(sec=int(t), nanosec=int(round((t % 1) * 1e9)))
            goal.trajectory.points.append(p)
        return goal

    def send_trajectory(self, targets, dt, q_start):
        """Trajektorie senden, ohne zu warten. Gibt das Goal-Future zurück."""
        return self.traj.send_goal_async(self.make_goal(targets, dt, q_start))

    def run_trajectory(self, targets, dt, q_start):
        """Trajektorie senden und bis zum Ende warten. False, wenn abgelehnt."""
        handle = wait_future(self.send_trajectory(targets, dt, q_start))
        if handle is None or not handle.accepted:
            return False
        wait_future(handle.get_result_async())
        return True

    def hold(self, duration=0.5):
        """Arm sanft an der aktuellen Referenz anhalten und warten, bis er steht."""
        q = self.reference_q()
        handle = wait_future(self.send_trajectory([q], duration, q), 2.0)
        if handle is not None and handle.accepted:
            wait_future(handle.get_result_async(), duration + 2.0)

    def gripper_goal(self, g):
        goal = self._GripperCommand.Goal()
        goal.command.position = gripper_to_raw(g, self.args.gripper_open, self.args.gripper_closed)
        goal.command.max_effort = self.args.gripper_effort
        return goal


class GraspProfile:
    """Schaltet die Admittanz beim Schliessen des Greifers in z auf ein Greifprofil.

    Normalprofil: steif und gut gedämpft, damit der Arm der Policy genau folgt.
    Greifprofil:  in z weich und leicht. Die Anpresskraft der Finger auf den Tisch
                  (~ k_z * Eindringtiefe) wird klein, und der Arm gibt dem Fingerbogen des 2F-85
                  nach, statt die Spitzen über die Platte zu ziehen (Quietschen = Reibung unter Last).

    Gegen Hochhüpfen:
      - Die Dämpfung wird als absoluter Wert D in Ns/m vorgegeben, nicht als Dämpfungsgrad.
        Sonst sinkt sie mit der Steifigkeit mit (D = 2*zeta*sqrt(M*K)) und die beim Aufdrücken
        gespeicherte Kontaktkraft schleudert den Arm nach oben.
      - Zwischen den Profilen wird über --grasp-ramp bzw. --grasp-ramp-back weich übergeblendet,
        statt die Kontaktkraft schlagartig freizugeben.
    Das Normalprofil wird beim Start aus dem Controller gelesen und danach wiederhergestellt.
    """

    PARAMS = ("admittance.stiffness", "admittance.mass", "admittance.damping_ratio", "ft_sensor.filter_coefficient")
    RAMP_DT = 0.025  # s zwischen zwei Zwischenschritten beim Überblenden

    def __init__(self, robot: Robot, args):
        from rclpy.parameter import Parameter
        from rclpy.parameter_client import AsyncParameterClient

        self.robot = robot
        self.args = args
        self._Parameter = Parameter
        self.enabled = False
        self.active = False
        self.generation = 0
        self.lock = threading.Lock()
        if args.mode in ("dry", "jointtest") or args.no_gripper or args.grasp_z_stiffness <= 0:
            return

        self.client = AsyncParameterClient(robot.node, args.admittance_node)
        if not self.client.wait_for_services(timeout_sec=2.0):
            print(f"Weiches Greifen aus: {args.admittance_node} nicht gefunden (Admittanz nicht aktiv?).")
            return
        res = wait_future(self.client.get_parameters(list(self.PARAMS)), 2.0)
        if res is None or len(res.values) != len(self.PARAMS):
            print("Weiches Greifen aus: Admittanz-Parameter nicht lesbar.")
            return
        stiffness, mass, ratio = (np.array(v.double_array_value, dtype=np.float64) for v in res.values[:3])
        filt = res.values[3].double_value
        if not (len(stiffness) == len(mass) == len(ratio) == 6):
            print("Weiches Greifen aus: unerwartete Admittanz-Parameter.")
            return

        # Profile intern als (K, M, D absolut, Filter)
        self.normal = (stiffness, mass, 2 * ratio * np.sqrt(mass * stiffness), filt)
        k, m, d = self.normal[0].copy(), self.normal[1].copy(), self.normal[2].copy()
        k[2], m[2], d[2] = args.grasp_z_stiffness, args.grasp_z_mass, args.grasp_z_damping
        self.grasp = (k, m, d, filt if args.grasp_filter is None else args.grasp_filter)
        self.current = self.normal
        self.enabled = True
        print(f"Weiches Greifen an: z beim Schliessen K {self.normal[0][2]:.0f}->{k[2]:.0f} N/m, "
              f"M {self.normal[1][2]:.1f}->{m[2]:.1f} kg, D {self.normal[2][2]:.0f}->{d[2]:.0f} Ns/m, "
              f"Filter {filt:.2f}->{self.grasp[3]:.2f}, Überblenden {args.grasp_ramp:.2f}/{args.grasp_ramp_back:.2f} s, "
              f"Entlasten {args.grasp_settle:.2f} s")

    # ---- Parameter setzen ------------------------------------------------ #

    def _apply(self, profile, wait):
        """Profil (K, M, D absolut, Filter) setzen. D wird in den Dämpfungsgrad umgerechnet."""
        P = self._Parameter
        k, m, d, filt = profile
        ratio = d / (2 * np.sqrt(m * np.maximum(k, 1e-9)))
        params = [
            P("admittance.stiffness", P.Type.DOUBLE_ARRAY, [float(x) for x in k]),
            P("admittance.mass", P.Type.DOUBLE_ARRAY, [float(x) for x in m]),
            P("admittance.damping_ratio", P.Type.DOUBLE_ARRAY, [float(x) for x in ratio]),
            P("ft_sensor.filter_coefficient", P.Type.DOUBLE, float(filt)),
        ]
        fut = self.client.set_parameters_atomically(params)
        if not wait:
            return True
        res = wait_future(fut, 1.0)
        ok = res is not None and res.result.successful
        if not ok:
            print("Warnung: Admittanz-Profil konnte nicht gesetzt werden.")
        return ok

    @staticmethod
    def _mix(a, b, s):
        """Zwischenprofil: Steifigkeit geometrisch (weicher Übergang über Grössenordnungen),
        Masse, Dämpfung und Filter linear."""
        def lin(x, y):
            return np.where(x == y, x, (1 - s) * x + s * y)

        k = np.where(a[0] == b[0], a[0], a[0] ** (1 - s) * b[0] ** s)
        return (k, lin(a[1], b[1]), lin(a[2], b[2]), float(lin(np.float64(a[3]), np.float64(b[3]))))

    def _ramp(self, target, duration, gen):
        """Vom aktuellen Profil in `duration` Sekunden zum Zielprofil überblenden.

        Bricht ab, wenn inzwischen ein neuerer Greifbefehl kam (andere Generation).
        """
        start = self.current
        steps = max(1, int(round(duration / self.RAMP_DT)))
        for i in range(1, steps + 1):
            with self.lock:
                if gen != self.generation:
                    return False
                prof = self._mix(start, target, i / steps)
                self._apply(prof, wait=(i == steps))
                self.current = prof
            if i < steps:
                time.sleep(self.RAMP_DT)
        return True

    # ---- Ablauf ------------------------------------------------------------ #

    def close_gripper(self, goal):
        """Weich ins Greifprofil, Greifer schliessen, nach Ende (+ --grasp-hold) zurück."""
        if not self.enabled:
            self.robot.grip.send_goal_async(goal)
            return
        with self.lock:
            self.generation += 1
            gen = self.generation
            self.active = True
        # Erst weich werden und den Arm die Anpresskraft abbauen lassen, dann die Finger bewegen.
        if self._ramp(self.grasp, self.args.grasp_ramp, gen) and self.args.grasp_settle > 0:
            time.sleep(self.args.grasp_settle)
        fut = self.robot.grip.send_goal_async(goal)

        def on_goal(f):
            handle = f.result()
            if handle is None or not handle.accepted:
                self._restore(gen)
                return
            handle.get_result_async().add_done_callback(
                lambda _: threading.Timer(self.args.grasp_hold, self._restore, (gen,)).start())

        fut.add_done_callback(on_goal)
        threading.Timer(self.args.grasp_max, self._restore, (gen,)).start()  # Notbremse

    def _restore(self, gen):
        with self.lock:
            if gen != self.generation or not self.active:
                return
            self.active = False
        threading.Thread(target=self._ramp, args=(self.normal, self.args.grasp_ramp_back, gen), daemon=True).start()

    def restore_now(self):
        """Sofort (ohne Überblenden) ins Normalprofil, z. B. beim Öffnen oder Beenden."""
        with self.lock:
            self.generation += 1
            if self.enabled and self.current is not self.normal:
                self.active = False
                self._apply(self.normal, wait=True)
                self.current = self.normal


class Gripper:
    """Sendet Greiferbefehle nur bei Änderung > 0.1 und nutzt beim Schliessen das Greifprofil."""

    def __init__(self, robot: Robot, profile: GraspProfile, args):
        self.robot = robot
        self.profile = profile
        self.args = args
        self.last = None

    def command(self, g_target: float):
        if self.args.no_gripper or (self.last is not None and abs(g_target - self.last) <= 0.1):
            return
        closing = g_target > (self.last if self.last is not None else 0.0)
        goal = self.robot.gripper_goal(g_target)
        if closing:
            self.profile.close_gripper(goal)
        else:
            self.profile.restore_now()  # beim Öffnen sofort zurück ins Normalprofil
            self.robot.grip.send_goal_async(goal)
        self.last = g_target


# =============================================================================
# Modi
# =============================================================================


def run_jointtest(robot: Robot, args, start_q):
    """Ohne Modell: jedes Gelenk einzeln um +delta und zurück. Prüft Reihenfolge, Richtung, Controller."""
    delta = min(args.test_delta, args.max_total)
    print(f"\nGelenktest: jedes Gelenk +{delta:.2f} rad ({np.degrees(delta):.0f}°) und zurück, je 2 s.")
    for i, name in enumerate(ARM_JOINTS):
        ans = input(f"\n[{i + 1}/6] {name} bewegen? [Enter = ja, s = überspringen, q = beenden] ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        q = robot.measured_q()
        tgt = q.copy()
        tgt[i] += delta
        ok = robot.run_trajectory([tgt], 2.0, robot.reference_q())
        moved = robot.measured_q() - q
        print(f"  Soll: {name} +{delta:.3f} | Ist-Änderung:",
              " ".join(f"{n.split('_joint')[0]}={d:+.3f}" for n, d in zip(ARM_JOINTS, moved)),
              "" if ok else " [ABGELEHNT]")
        robot.run_trajectory([q], 2.0, robot.reference_q())


def run_chunked(robot: Robot, policy, gripper: Gripper, args, start_q):
    """dry / step / auto: pro Chunk die ersten --exec-steps Aktionen ausführen, dann anhalten."""
    lim = Limits(args.max_step, args.max_total)
    dt = args.slowdown / args.fps
    for chunk_no in range(1, args.max_chunks + 1):
        obs, q, g_now = robot.observation()
        t0 = time.time()
        actions = np.asarray(policy.infer(obs)["actions"])
        latency = time.time() - t0
        q_ref = robot.reference_q() if args.mode != "dry" else q
        chunk = actions[: args.exec_steps]
        targets, clipped = make_safe_targets(chunk, q_ref, start_q, lim)
        g_target = gripper_target(chunk, len(chunk))

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

        if not robot.run_trajectory(targets, dt, q_ref):
            print("Trajektorie vom Controller abgelehnt – Abbruch.")
            break
        gripper.command(g_target)


def run_stream(robot: Robot, policy, gripper: Gripper, args, start_q):
    """Ohne Anhalten: immer den ganzen Rest des Chunks senden, nach --exec-steps neu planen.

    Glätte entsteht aus vier Dingen:
      - Geschwindigkeiten an jedem Punkt -> kubische statt lineare Interpolation
      - der Arm hat immer mehr Trajektorie vor sich, als bis zur Neuplanung nötig ist -> kein
        Abbremsen am Chunk-Ende, auch wenn die Inferenz mal länger dauert
      - der neue Chunk setzt an der Controller-Referenz an (nicht an der gemessenen Position)
      - die ersten --blend-steps Aktionen werden aus dem alten Plan übergeblendet
    """
    lim = Limits(args.max_step, args.max_total)
    dt = args.slowdown / args.fps

    print("Kompiliere Modell (erster Aufruf dauert einige Sekunden) ...")
    obs, _, _ = robot.observation()
    t0 = time.time()
    policy.infer(obs)
    print(f"  fertig nach {time.time() - t0:.1f} s")
    print(f"Stream-Modus: {dt:.2f} s pro Aktion, neu planen nach {args.exec_steps} Aktionen, "
          f"Überblenden über {args.blend_steps} Aktionen.")

    latency_est = 0.3
    plan: Plan | None = None
    goal_fut = None

    for chunk_no in range(1, args.max_chunks + 1):
        obs, _, _ = robot.observation()
        t_obs = time.time()
        actions = np.asarray(policy.infer(obs)["actions"])
        latency = time.time() - t_obs
        latency_est = 0.7 * latency_est + 0.3 * latency

        if goal_fut is not None and goal_fut.done() and not goal_fut.result().accepted:
            print("Trajektorie vom Controller abgelehnt – Abbruch.")
            break

        q_ref = robot.reference_q()
        skip = nearest_action(actions, q_ref) + 1
        chunk = actions[min(skip, len(actions) - 1):]
        targets, clipped = make_safe_targets(chunk, q_ref, start_q, lim)
        why = clip_reasons(np.asarray(chunk, dtype=np.float64), targets, start_q, lim) if clipped else ""
        t_send = time.monotonic()
        targets = blend_into_plan(targets, plan, t_send, dt, args.blend_steps)

        goal_fut = robot.send_trajectory(targets, dt, q_ref)
        plan = Plan(t_send, dt, q_ref, targets)
        n_exec = min(args.exec_steps, len(targets))
        g_target = gripper_target(chunk, n_exec)
        gripper.command(g_target)

        print(f"Chunk {chunk_no:3d}: Inferenz {latency * 1000:4.0f} ms, Start bei Aktion {skip}, "
              f"{len(targets)} gesendet, neu nach {n_exec}, Greifer {g_target:.2f}"
              f"{'  [BEGRENZT: ' + why + ']' if clipped else ''}")

        # Neu planen, sobald die Referenz so weit ist, dass die nächste Antwort vor Aktion n_exec da ist.
        lead = int(np.ceil(latency_est / dt))
        trigger = max(0, n_exec - 1 - lead)
        deadline = time.time() + 20 * n_exec * dt  # Notbremse, falls der Arm nicht vorankommt
        while time.time() < deadline:
            if goal_fut.done() and not goal_fut.result().accepted:
                break
            reached = int(np.argmin(np.linalg.norm(targets - robot.reference_q(), axis=1)))
            if reached >= trigger:
                break
            time.sleep(0.01)


# =============================================================================
# Hauptprogramm
# =============================================================================


def run(args):
    robot = Robot(args)
    profile = None
    try:
        print("Warte auf Roboter- und Kameradaten ...")
        jmsg, _, _ = robot.wait_for_data()
        start_q, _ = robot.read_state(jmsg)
        print("Startpose (rad):", np.round(start_q, 3))

        if args.mode != "dry":
            robot.wait_for_servers()
        if args.mode == "jointtest":
            run_jointtest(robot, args, start_q)
            return

        profile = GraspProfile(robot, args)
        gripper = Gripper(robot, profile, args)

        websocket_client_policy = import_openpi_client()
        print(f"Verbinde mit Policy-Server {args.host}:{args.port} ...")
        policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        print("Verbunden. Prompt:", repr(args.prompt))

        if args.mode == "stream":
            run_stream(robot, policy, gripper, args, start_q)
        else:
            run_chunked(robot, policy, gripper, args, start_q)
    except KeyboardInterrupt:
        print("\nAbgebrochen (Strg+C).")
    finally:
        if args.mode in ("stream", "auto", "step", "jointtest"):
            try:
                robot.hold()
            except Exception:
                pass
        if profile is not None:
            profile.restore_now()
        robot.shutdown()


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
    p.add_argument("--controller-state-topic", default="/scaled_joint_trajectory_controller/controller_state")
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
    p.add_argument("--blend-steps", type=int, default=3, help="Stream: Überblenden alter -> neuer Chunk")
    p.add_argument("--max-chunks", type=int, default=20)
    p.add_argument("--dry-pause", type=float, default=1.0, help="Pause zwischen Chunks im dry-Modus [s]")
    p.add_argument("--grasp-z-stiffness", type=float, default=12.0,
                   help="z-Steifigkeit der Admittanz beim Schliessen [N/m], 0 = weiches Greifen aus")
    p.add_argument("--grasp-z-mass", type=float, default=1.0, help="virtuelle Masse in z beim Schliessen [kg]")
    p.add_argument("--grasp-z-damping", type=float, default=200.0,
                   help="absolute Dämpfung in z beim Schliessen [Ns/m]")
    p.add_argument("--grasp-filter", type=float, default=0.07,
                   help="Filterkoeffizient Kraftmessung beim Schliessen (kleiner = Kraftspitzen beim "
                        "Schliessen werden stärker geglättet)")
    p.add_argument("--grasp-ramp", type=float, default=0.3, help="Überblenden ins Greifprofil [s]")
    p.add_argument("--grasp-settle", type=float, default=0.35,
                   help="nach dem Überblenden warten, bis der Arm entlastet hat, erst dann schliessen [s]")
    p.add_argument("--grasp-ramp-back", type=float, default=0.8, help="Überblenden zurück ins Normalprofil [s]")
    p.add_argument("--grasp-hold", type=float, default=0.3, help="nach Ende des Schliessens im Greifprofil bleiben [s]")
    p.add_argument("--grasp-max", type=float, default=3.0, help="spätestens nach dieser Zeit zurück [s]")
    p.add_argument("--admittance-node", default="/admittance_controller")
    run(p.parse_args())


if __name__ == "__main__":
    main()