"""check_dataset.py – Plausibilitätsprüfung eines fertigen LeRobot-Datensatzes (UR5e, openpi).

Liest die Parquet-Dateien direkt (ohne LeRobot) und prüft:
  1. Metadaten:   Episoden, Tasks, fps, Frames pro Task
  2. Zeit:        Zeitstempel lückenlos in 1/fps-Schritten
  3. Aktionen:    Aktion[t] passt zum Zustand[t+1], keine Sprünge in den Gelenken, keine NaNs
  4. Greifer:     Werte in 0..1, mindestens ein Griff pro Episode, keine harten Sprünge,
                  gestuftes Schließen (erst leicht zu, dann ganz zu) bleibt in der Aktion erhalten,
                  Greifer-Aktion läuft dem gemessenen Greifer nicht weit voraus
  5. Bilder:      richtige Größe, nicht schwarz, Kamera nicht eingefroren
  6. Länge:       auffällig kurze oder lange Episoden
Ausgabe: Übersicht + Liste auffälliger Episoden im Terminal, dazu
  dataset_check_sheet.png   Kontaktbogen: Start-, Mitte- und Endbild zufälliger Episoden (beide Kameras)
  dataset_check_stats.png   Verteilungen: Episodenlänge, Greifbreite, Gelenkschritte
  dataset_check_gripper.png Greifer gemessen vs. Aktion für auffällige bzw. gestufte Episoden

Mit --fast zusätzlich die pi0-FAST-Prüfung über den GESAMTEN Datensatz:
  Jeder Action-Chunk (Horizont 10) wird so aufbereitet wie in openpi (Gelenke als Delta zum
  aktuellen Zustand, Greifer absolut, Quantil-Normalisierung q01/q99) und mit dem echten
  FAST-Tokenizer kodiert und wieder dekodiert. Geprüft wird:
    - Tokens pro Chunk passen ins Budget (max_token_len minus Platz für Prompt und Zustand)
    - Rekonstruktionsfehler nach Kodieren/Dekodieren ist klein
  Braucht einmalig Internet für den Tokenizer von Hugging Face (physical-intelligence/fast).

Aufruf (in der openpi-Umgebung):
    uv run python examples/ur5e/check_dataset.py ~/ros2_ws/lerobot_datasets/local/ur5e_250
    uv run python examples/ur5e/check_dataset.py ~/ros2_ws/lerobot_datasets/local/ur5e_250 --fast
"""
import argparse
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

# Grenzwerte für "auffällig"
MAX_ACTION_STATE_GAP = 0.02   # rad: Aktion[t] vs. Zustand[t+1]
MAX_JOINT_STEP = 0.08         # rad pro Frame (Client begrenzt auf 0.06)
MAX_GRIP_STEP = 0.35          # Greifer-Aktion pro Frame
MIN_GRIP_CLOSED = 0.1         # mindestens so weit muss pro Episode gegriffen werden
FROZEN_SECONDS = 1.0          # identische Bilder länger als das = Kamera eingefroren
DARK_MEAN = 15                # mittlere Helligkeit darunter = schwarzes Bild
MIN_SECONDS, MAX_SECONDS = 8.0, 60.0

# Greifer-Stufen und Vorlauf
GRIP_STILL_EPS = 0.005        # Änderung pro Frame darunter = Greifer steht
GRIP_MIN_PLATEAU = 8          # so viele Frames still = Haltestufe (0.8 s bei 10 Hz)
GRIP_STAGE_THR = 0.08         # Stufen, die sich weniger unterscheiden, zählen als eine
GRIP_CLOSED_MIN = 0.05        # darüber gilt eine Stufe als "zu"
GRIP_LOOKAHEAD = 5            # Aktion[t] wird mit gemessenem Greifer[t+5] verglichen (Rampe + Verzögerung)
GRIP_LEAD_MARGIN = 0.10       # so viel weiter zu als der Greifer gleich sein wird ...
GRIP_LEAD_SECONDS = 1.0       # ... und das länger als 1 s = Aktion eilt voraus (Stufe verloren)

# pi0-FAST
FAST_HORIZON = 10             # action_horizon der Config
FAST_MAX_TOKEN_LEN = 180      # max_token_len der Config
FAST_PREFIX_RESERVE = 60      # Tokens für Prompt, diskretisierten Zustand und Formatierung
FAST_MAX_RECON_ERR = 0.05     # mittlerer Fehler (normalisiert, Bereich -1..1) nach Kodieren/Dekodieren
DELTA_DIMS = 6                # Gelenke als Delta, Greifer absolut (delta_actions=True)


def img_bytes(x):
    return x["bytes"] if isinstance(x, dict) else x


def decode(x):
    return np.asarray(Image.open(io.BytesIO(img_bytes(x))).convert("RGB"))


def longest_identical_run(col):
    """Längste Folge identischer aufeinanderfolgender Bilder (über Byte-Hash)."""
    best = run = 1
    prev = None
    for x in col:
        h = hashlib.md5(img_bytes(x)).digest()
        run = run + 1 if h == prev else 1
        best = max(best, run)
        prev = h
    return best


def grip_stages(gs):
    """Haltestufen des gemessenen Greifers: Liste (start_frame, level), benachbarte ähnliche zusammengefasst."""
    still = np.r_[True, np.abs(np.diff(gs)) < GRIP_STILL_EPS]
    stages, t = [], 0
    while t < len(gs):
        if not still[t]:
            t += 1
            continue
        u = t
        while u < len(gs) and still[u]:
            u += 1
        if u - t >= GRIP_MIN_PLATEAU:
            level = float(np.median(gs[t:u]))
            if stages and abs(level - stages[-1][1]) < GRIP_STAGE_THR:
                stages[-1] = (stages[-1][0], level)
            else:
                stages.append((t, level))
        t = u
    return stages


def stepped_closings(stages):
    """Anzahl Stellen, an denen von einer Zu-Stufe direkt auf eine deutlich engere Stufe weitergeschlossen wird."""
    return sum(1 for (_, a), (_, b) in zip(stages, stages[1:])
               if a > GRIP_CLOSED_MIN and b > a + GRIP_STAGE_THR)


def grip_lead_run(ga, gs):
    """Längste Strecke (Frames), in der die Aktion deutlich weiter zu ist als der Greifer kurz danach."""
    n = len(ga)
    future = gs[np.minimum(np.arange(n) + GRIP_LOOKAHEAD, n - 1)]
    ahead = ga - future > GRIP_LEAD_MARGIN
    best = run = 0
    best_end = 0
    for t, v in enumerate(ahead):
        run = run + 1 if v else 0
        if run > best:
            best, best_end = run, t
    return best, best_end - best + 1


def fast_check(episodes, horizon, stride):
    """episodes: Liste (name, S, A). Gibt Bericht-Zeilen und auffällige Episoden zurück."""
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

    # Chunks wie in openpi: Delta der Gelenke zum Zustand zu Beginn des Chunks, Greifer absolut
    chunks, owner = [], []
    for name, S, A in episodes:
        for t in range(0, len(A) - horizon + 1, stride):
            c = A[t:t + horizon].copy()
            c[:, :DELTA_DIMS] -= S[t, :DELTA_DIMS]
            chunks.append(c)
            owner.append(name)
    chunks = np.stack(chunks)
    flat = chunks.reshape(-1, chunks.shape[-1])
    q01, q99 = np.percentile(flat, 1, axis=0), np.percentile(flat, 99, axis=0)
    norm = np.clip((chunks - q01) / (q99 - q01 + 1e-6) * 2 - 1, -1, 1)

    lens, errs = [], []
    for b in range(0, len(norm), 512):
        batch = norm[b:b + 512]
        tokens = tok(batch)
        lens.extend(len(x) for x in tokens)
        recon = np.asarray(tok.decode(tokens, time_horizon=horizon, action_dim=norm.shape[-1]))
        errs.extend(np.abs(recon - batch).mean(axis=(1, 2)).tolist())
    lens, errs = np.array(lens), np.array(errs)
    budget = FAST_MAX_TOKEN_LEN - FAST_PREFIX_RESERVE

    bad = defaultdict(list)
    for name, n_tok, err in zip(owner, lens, errs):
        if n_tok > budget:
            bad[name].append(f"Chunk mit {n_tok} FAST-Tokens > Budget {budget}")
        if err > FAST_MAX_RECON_ERR:
            bad[name].append(f"Chunk schlecht rekonstruiert (Fehler {err:.3f})")
    report = [
        f"Chunks geprüft: {len(lens)} (jeder {stride}. Startpunkt, Horizont {horizon})",
        f"FAST-Tokens pro Chunk: Median {np.median(lens):.0f}, 99 % {np.percentile(lens, 99):.0f}, "
        f"Max {lens.max()}  | Budget {budget} (max_token_len {FAST_MAX_TOKEN_LEN} - {FAST_PREFIX_RESERVE} Reserve)",
        f"Rekonstruktionsfehler: Median {np.median(errs):.4f}, 99 % {np.percentile(errs, 99):.4f}, "
        f"Max {errs.max():.4f}  | Grenze {FAST_MAX_RECON_ERR}",
        f"Chunks über Budget: {(lens > budget).sum()} | schlecht rekonstruiert: {(errs > FAST_MAX_RECON_ERR).sum()}",
    ]
    verdict = "geeignet" if (lens.max() <= budget and np.percentile(errs, 99) <= FAST_MAX_RECON_ERR) else "PRÜFEN"
    # Dedupliziert pro Episode: nur Anzahl melden
    bad_summary = {k: [f"{len(v)} auffällige FAST-Chunks, z.B.: {v[0]}"] for k, v in bad.items()}
    return report, verdict, bad_summary, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Datensatz-Ordner, z.B. .../lerobot_datasets/local/ur5e_250")
    ap.add_argument("--sheet-episodes", type=int, default=12, help="Episoden auf dem Kontaktbogen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="zusätzlich pi0-FAST-Tokenisierung prüfen")
    ap.add_argument("--fast-stride", type=int, default=1, help="nur jeden n-ten Chunk-Start prüfen (schneller)")
    args = ap.parse_args()
    root = args.root.expanduser()

    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    tasks = {}
    for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            tasks[t["task_index"]] = t["task"]
    files = sorted((root / "data").rglob("episode_*.parquet"))

    print(f"Datensatz: {root}")
    print(f"info.json: {info['total_episodes']} Episoden, {info['total_frames']} Frames, "
          f"{info['total_tasks']} Tasks, {fps} fps | Parquet-Dateien gefunden: {len(files)}")
    if len(files) != info["total_episodes"]:
        print("⚠ Anzahl Parquet-Dateien passt nicht zu info.json!")
    if not files:
        raise SystemExit("Der Datensatz enthält keine Episoden – Konvertierung abgebrochen oder Datensatz gelöscht?")
    if not files:
        raise SystemExit("Der Datensatz enthält keine Episoden – Konvertierung abgebrochen oder Datensatz gelöscht?")

    issues = defaultdict(list)
    per_task_eps, per_task_frames = Counter(), Counter()
    lengths, grip_max, joint_steps = [], [], []
    grip_curves = {}              # name -> (S_greifer, A_greifer, Grund) für die Greifer-Grafik
    n_stepped = n_lead = 0
    fast_episodes = []
    sheet_rows = []
    rng = random.Random(args.seed)
    sheet_pick = set(rng.sample(range(len(files)), min(args.sheet_episodes, len(files))))

    for i, f in enumerate(files):
        df = pq.read_table(f).to_pandas()
        ep = int(df["episode_index"].iloc[0])
        name = f"episode_{ep:03d}"
        task = tasks.get(int(df["task_index"].iloc[0]), "?")
        S = np.stack(df["state"].values).astype(float)
        A = np.stack(df["actions"].values).astype(float)
        n = len(df)
        if args.fast:
            fast_episodes.append((name, S, A))
        per_task_eps[task] += 1
        per_task_frames[task] += n
        lengths.append(n / fps)

        # 1. NaN / Form
        if not (np.isfinite(S).all() and np.isfinite(A).all()):
            issues[name].append("NaN/Inf in state oder actions")
        if S.shape[1] != 7 or A.shape[1] != 7:
            issues[name].append(f"falsche Dimension state {S.shape} actions {A.shape}")

        # 2. Zeit
        dt = np.diff(df["timestamp"].values)
        if len(dt) and (np.abs(dt - 1 / fps) > 1e-3).any():
            issues[name].append(f"Zeitstempel nicht gleichmäßig ({dt.min():.3f}..{dt.max():.3f} s)")

        # 3. Aktionen vs. Zustand, Gelenksprünge
        gap = np.abs(A[:-1, :6] - S[1:, :6]).max(axis=1) if n > 1 else np.zeros(1)
        if gap.max() > MAX_ACTION_STATE_GAP:
            issues[name].append(f"Aktion passt nicht zum nächsten Zustand (max {gap.max():.3f} rad, "
                                f"{(gap > MAX_ACTION_STATE_GAP).sum()} Frames)")
        step = np.abs(np.diff(S[:, :6], axis=0)).max(axis=1) if n > 1 else np.zeros(1)
        joint_steps.extend(step.tolist())
        if step.max() > MAX_JOINT_STEP:
            issues[name].append(f"Gelenksprung {step.max():.3f} rad in einem Frame "
                                f"(Frame {int(step.argmax())}, {step.argmax() / fps:.1f} s)")

        # 4. Greifer
        g = A[:, 6]
        grip_max.append(g.max())
        if g.min() < -0.01 or g.max() > 1.01:
            issues[name].append(f"Greifer-Aktion außerhalb 0..1 ({g.min():.2f}..{g.max():.2f})")
        if g.max() < MIN_GRIP_CLOSED:
            issues[name].append("Greifer schließt nie")
        if n > 1 and np.abs(np.diff(g)).max() > MAX_GRIP_STEP:
            issues[name].append(f"harter Greifersprung ({np.abs(np.diff(g)).max():.2f} pro Frame)")
        gs = S[:, 6]
        stepped = stepped_closings(grip_stages(gs))
        lead, lead_at = grip_lead_run(g, gs)
        if stepped:
            n_stepped += 1
        if lead / fps > GRIP_LEAD_SECONDS:
            n_lead += 1
            issues[name].append(f"Greifer-Aktion schließt {lead / fps:.1f} s zu früh (ab {lead_at / fps:.1f} s) – "
                                f"Zwischenstufe verloren? Mit neuer Glättung neu konvertieren")
            grip_curves[name] = (gs, g, f"Aktion {lead / fps:.1f} s zu früh")
        elif stepped:
            grip_curves.setdefault(name, (gs, g, f"{stepped}x gestuft geschlossen, Aktion folgt"))

        # 5. Bilder: eingefroren, schwarz, Größe
        for cam in ("image", "wrist_image"):
            run = longest_identical_run(df[cam].values)
            if run / fps > FROZEN_SECONDS:
                issues[name].append(f"{cam}: {run / fps:.1f} s eingefroren ({run / n:.0%} der Episode)")
            for k in (0, n // 2, n - 1):
                im = decode(df[cam].iloc[k])
                if im.shape != (224, 224, 3):
                    issues[name].append(f"{cam}: Bildgröße {im.shape}")
                    break
                if im.mean() < DARK_MEAN:
                    issues[name].append(f"{cam}: fast schwarzes Bild bei Frame {k}")
                    break

        # 6. Länge
        if n / fps < MIN_SECONDS or n / fps > MAX_SECONDS:
            issues[name].append(f"Länge {n / fps:.1f} s")

        if i in sheet_pick:
            row = [decode(df[cam].iloc[k]) for k in (0, n // 2, n - 1) for cam in ("image", "wrist_image")]
            sheet_rows.append((name, task, row))

    # ---------- Ausgabe ----------
    print(f"\nFrames gesamt: {sum(per_task_frames.values())} | "
          f"Episodenlänge: Median {np.median(lengths):.1f} s, {min(lengths):.1f}..{max(lengths):.1f} s")
    print("\nEpisoden / Frames pro Task:")
    for t, c in sorted(per_task_eps.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {c:4d} Ep. | {per_task_frames[t]:6d} Frames | {t}")

    print(f"\nGelenkschritt pro Frame: Median {np.median(joint_steps):.4f}, "
          f"99 % {np.percentile(joint_steps, 99):.4f}, Max {max(joint_steps):.4f} rad")

    print(f"\nGreifer: {n_stepped} Episoden mit gestuftem Schließen (erst leicht zu, dann ganz zu), "
          f"{n_lead} Episoden, in denen die Greifer-Aktion > {GRIP_LEAD_SECONDS:g} s vorausläuft"
          + ("  ✓" if n_lead == 0 else "  ⚠ → Datensatz mit neuer gripper_smoothing.py neu konvertieren"))

    if args.fast:
        print("\npi0-FAST-Prüfung (gesamter Datensatz):")
        report, verdict, fast_bad, _ = fast_check(fast_episodes, FAST_HORIZON, args.fast_stride)
        for line in report:
            print("  " + line)
        print(f"  Ergebnis: {verdict}")
        for k, v in fast_bad.items():
            issues[k].extend(v)

    print(f"\nAuffällige Episoden: {len(issues)} von {len(files)}")
    for name in sorted(issues):
        for msg in issues[name]:
            print(f"  {name}: {msg}")
    if not issues:
        print("  keine – alles plausibel.")

    # Kontaktbogen
    if sheet_rows:
        tile, pad, label_h = 112, 4, 18
        cols = 6
        W = cols * (tile + pad) + pad
        H = len(sheet_rows) * (tile + label_h + pad) + pad
        sheet = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(sheet)
        for r, (name, task, imgs) in enumerate(sheet_rows):
            y = pad + r * (tile + label_h + pad)
            draw.text((pad, y), f"{name}: {task}   [Start: Tisch | Hand  -  Mitte: Tisch | Hand  -  Ende: Tisch | Hand]",
                      fill="black")
            for c, im in enumerate(imgs):
                sheet.paste(Image.fromarray(im).resize((tile, tile)), (pad + c * (tile + pad), y + label_h))
        sheet.save("dataset_check_sheet.png")
        print("\nKontaktbogen gespeichert: dataset_check_sheet.png")

    # Verteilungen
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
        ax[0].hist(lengths, bins=30, color="#0072B2")
        ax[0].set_title("Episodenlänge [s]")
        ax[1].hist(grip_max, bins=30, color="#0072B2")
        ax[1].set_title("Max. Greifer-Aktion pro Episode")
        ax[2].hist(joint_steps, bins=60, color="#0072B2", log=True)
        ax[2].axvline(0.06, color="#D55E00", lw=1)
        ax[2].set_title("Gelenkschritt pro Frame [rad] (rot: Client-Limit)")
        for a in ax:
            a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig("dataset_check_stats.png", dpi=120)
        print("Verteilungen gespeichert: dataset_check_stats.png")
        # Greifer gemessen vs. Aktion: zuerst die mit vorlaufender Aktion, dann gestufte
        if grip_curves:
            names = sorted(grip_curves, key=lambda k: ("früh" not in grip_curves[k][2], k))[:12]
            rows = (len(names) + 1) // 2
            fig, axs = plt.subplots(rows, 2, figsize=(14, 2.2 * rows), squeeze=False)
            for a in axs.flat:
                a.set_visible(False)
            for a, k in zip(axs.flat, names):
                gs, ga, why = grip_curves[k]
                tt = np.arange(len(gs)) / fps
                a.set_visible(True)
                a.plot(tt, gs, color="#0072B2", lw=1.5, label="gemessen (state)")
                a.plot(tt, ga, color="#D55E00", lw=1.2, label="Aktion")
                a.set_title(f"{k}: {why}", fontsize=9)
                a.set_ylim(-0.05, 1.05)
                a.spines[["top", "right"]].set_visible(False)
            axs.flat[0].legend(fontsize=8, frameon=False)
            for a in axs[-1]:
                a.set_xlabel("Zeit [s]")
            fig.tight_layout()
            fig.savefig("dataset_check_gripper.png", dpi=110)
            print("Greifer-Verläufe gespeichert: dataset_check_gripper.png")
    except ImportError:
        print("(matplotlib fehlt – keine Verteilungsgrafik)")


if __name__ == "__main__":
    main()