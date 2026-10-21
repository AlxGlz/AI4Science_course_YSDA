"""Print the задано/получено table used in report.md."""

import json
from pathlib import Path

SETS = Path(__file__).resolve().parent / "sets"
rows = []
for d in sorted(SETS.iterdir()):
    if not d.is_dir():
        continue
    s = json.loads((d / "params.json").read_text(encoding="utf-8"))
    c, m = s["config"], s["measured"]
    rows.append((d.name, c["duration_s"], c["hr_bpm"], m["hr_bpm"],
                 c["noise_hand"], m["noise_hand_std"], c["noise_foot"], m["noise_foot_std"],
                 c["amp_hand"], m["amp_hand_median"], s["n_beats"], s["n_missed_hand"],
                 c["n_missed_hand"], s["n_missed_foot"], c["n_missed_foot_only"],
                 max(c["n_attenuation"], 1), m["delay_ms_median"]))
print("| набор | T, с | HR задан/факт | шум рука | шум стопа | ампл. руки | пропуски рука / стопа |")
for r in rows:
    print(f"| {r[0]} | {r[1]:.0f} | {r[2]:.0f} / {r[3]:.1f} | {r[4]:.0f} / {r[5]:.1f} | "
          f"{r[6]:.0f} / {r[7]:.1f} | {r[8]:.0f} / {r[9]:.0f} | "
          f"{r[11]} из {r[12]} / {r[13]} |")
for r in rows:
    print(r[0], "peak delay", r[15], "ms")
