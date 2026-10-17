"""Prüft, ob das Vortraining für den UR5e Delta- oder absolute Gelenkaktionen benutzt hat.

Aufruf (nach dem Download der Basismodelle):
    uv run examples/ur5e/check_ur5e_normstats.py
"""
import json
import pathlib

import numpy as np

base = pathlib.Path("~/.cache/openpi/openpi-assets/checkpoints").expanduser()
for model in ("pi0_base", "pi0_fast_base"):
    f = base / model / "assets" / "ur5e" / "norm_stats.json"
    if not f.exists():
        print(f"{model}: {f} nicht gefunden – erst herunterladen.")
        continue
    ns = json.loads(f.read_text())["norm_stats"]
    s_mean, a_mean = np.array(ns["state"]["mean"][:7]), np.array(ns["actions"]["mean"][:7])
    a_std = np.array(ns["actions"]["std"][:7])
    print(f"\n{model}  (Gelenke 1-6, dann Greifer)")
    print("  state   mean:", np.round(s_mean, 3))
    print("  actions mean:", np.round(a_mean, 3))
    print("  actions std :", np.round(a_std, 3))
    close_to_state = np.abs(a_mean[:6] - s_mean[:6]).mean()
    near_zero = np.abs(a_mean[:6]).mean()
    verdict = "ABSOLUT  -> *_zeroshot_abs verwenden" if close_to_state < near_zero else "DELTA   -> *_zeroshot verwenden"
    print(f"  Aktionen ähneln eher {verdict}")
    s_std = np.array(ns["state"]["std"][:7])
    print("  state   std :", np.round(s_std, 3))
    q01, q99 = ns["state"].get("q01"), ns["state"].get("q99")
    if q01 is not None and q99 is not None:
        print("  Greifer q01..q99:", round(q01[6], 3), "..", round(q99[6], 3))
    # Wie weit liegt eine typische Pose von dir vom Vortraining entfernt (in Standardabweichungen)?
    mine = np.array([1.25, -1.57, 2.16, -2.19, -1.56, -0.58])  # Mittelwerte aus deinem Datensatz
    print("  Abstand deiner Pose zum Vortraining [Std-Abw.]:", np.round((mine - s_mean[:6]) / np.maximum(s_std[:6], 1e-6), 1))
