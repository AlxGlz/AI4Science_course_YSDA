"""Сводка «задано / фактическое» для отчёта (вспомогательный скрипт прогона)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SETS = Path(__file__).resolve().parents[1] / "sets"

for name in ("set_clean", "set_baseline", "set_hard"):
    d = SETS / name
    p = json.loads((d / "params.json").read_text(encoding="utf-8"))["config"]
    with open(d / "reference.csv", encoding="utf-8", newline="") as fh:
        ref = list(csv.DictReader(fh))
    ah = [float(r["amp_hand"]) for r in ref if r["missed_hand"] == "0"]
    af = [float(r["amp_foot"]) for r in ref if r["missed_foot"] == "0"]
    gn = [float(r["foot_gain"]) for r in ref if r["missed_foot"] == "0"]
    print(f"{name}: длительность {p['duration_s']:.0f} с, ударов {len(ref)}, "
          f"W {p['design']['width_s']:.3f} с, min IBI {p['design']['min_ibi_s']:.3f} с, "
          f"rise {p['rise_hand_s']:.3f}/{p['rise_foot_s']:.3f} с, "
          f"задержка {p['foot_delay_s']:.3f} с, "
          f"ампл. руки мед. {np.median(ah):.0f} (мин {min(ah):.0f}), "
          f"стопы мед. {np.median(af):.0f} (мин {min(af):.0f}), "
          f"gain мед. {np.median(gn):.3f} мин {min(gn):.3f}, "
          f"пропусков рука/стопа {p['design']['n_missed_hand']}/"
          f"{p['design']['n_missed_foot']}")
