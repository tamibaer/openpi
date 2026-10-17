"""Legt einen Zero-Shot-Checkpoint für pi0-FAST + UR5e an.

Problem: Die UR5e-Normstats im Basis-Checkpoint haben 32 Dimensionen. pi0-FAST füllt den
Zustand aber nicht auf 32 auf (im Gegensatz zu pi0), deshalb scheitert das Zurückrechnen.
Lösung: Statistik auf die ersten 7 Dimensionen kürzen (6 Gelenke + Greifer) – genau das,
was openpi beim Hinrechnen ohnehin benutzt.

Ergebnis: ~/openpi/checkpoints/pi0_fast_base_ur5e/
    params  -> Verweis auf die Gewichte des Basismodells (keine Kopie, kein Speicherplatz)
    assets/ur5e/norm_stats.json  (gekürzte Statistik)

Aufruf (im openpi-Ordner):
    uv run examples/ur5e/make_fast_zeroshot_ckpt.py
Danach:
    uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config=pi0_fast_ur5e_zeroshot --policy.dir=checkpoints/pi0_fast_base_ur5e
"""

import json
import pathlib

DIMS = 7
base = pathlib.Path("~/.cache/openpi/openpi-assets/checkpoints/pi0_fast_base").expanduser()
out = pathlib.Path("checkpoints/pi0_fast_base_ur5e")

src_stats = base / "assets" / "ur5e" / "norm_stats.json"
if not src_stats.exists():
    raise SystemExit(f"{src_stats} fehlt – erst pi0_fast_base herunterladen.")

data = json.loads(src_stats.read_text())
for key, stats in data["norm_stats"].items():
    for field, values in stats.items():
        if isinstance(values, list):
            stats[field] = values[:DIMS]
    print(f"{key}: auf {DIMS} Dimensionen gekürzt")

(out / "assets" / "ur5e").mkdir(parents=True, exist_ok=True)
(out / "assets" / "ur5e" / "norm_stats.json").write_text(json.dumps(data, indent=2))

link = out / "params"
if not link.exists():
    link.symlink_to(base / "params", target_is_directory=True)
print(f"Fertig: {out.resolve()}")
