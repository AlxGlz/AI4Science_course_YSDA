"""Numeric verification of what the figures show (no vision available)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

FS = 320.0
SETS = Path(__file__).resolve().parent / "sets"


def load(name):
    data = np.loadtxt(SETS / name / "signal.csv", delimiter=",", skiprows=1)
    ref = list(csv.DictReader(open(SETS / name / "reference.csv", encoding="utf-8")))
    return data[:, 0], data[:, 1], data[:, 2], ref


def markers_ok(t, x, ref, t0, t1, key_t, key_missed):
    sel = [b for b in ref if t0 <= float(b["onset_s"]) <= t1 and b[key_missed] == "0"]
    errs = []
    for b in sel:
        pk = float(b[key_t])
        i = int(round(pk * FS))
        seg = x[max(0, i - 16):i + 17]
        errs.append(np.abs(seg - x[i]).min() / max(x[i], 1))
    return len(sel), (max(errs) if errs else 0.0)


t, hand, foot, ref = load("set_clean")
n1, e1 = markers_ok(t, hand, ref, 40, 68, "peak_s", "missed_hand")
n2, e2 = markers_ok(t, foot, ref, 40, 68, "peak_foot_s", "missed_foot")
print(f"fig1 window 40-68 s: {n1} hand peaks, {n2} foot peaks, "
      f"max |peak - sample| {1000 * max(e1, e2):.0f} ms")

t, hand, foot, ref = load("set_baseline")
gains = np.array([float(b["foot_gain"]) for b in ref])
ons = np.array([float(b["onset_s"]) for b in ref])
c = ons[int(np.argmin(gains))]
lo, hi = c - 20, c + 20
sel_in = [b for b in ref if abs(float(b["onset_s"]) - c) < 4]
sel_out = [b for b in ref if 8 < abs(float(b["onset_s"]) - c) < 18]
print(f"fig2 window {lo:.0f}-{hi:.0f} s around t={c:.1f} s")
print(f"   median amp_foot inside +-4 s: {np.median([float(b['amp_foot']) for b in sel_in]):.0f}, "
      f"outside: {np.median([float(b['amp_foot']) for b in sel_out]):.0f}")

t, hand, foot, ref = load("set_hard")
cand = [b for b in ref if b["missed_hand"] == "1" and b["missed_foot"] == "1"]
b = cand[len(cand) // 2]
print(f"fig3: missed beat {b['beat_id']} at t={float(b['onset_s']):.2f} s "
      f"(both channels), {len(cand)} such beats in the set")
i = int(round(float(b["onset_s"]) * FS))
seg = hand[i:i + int(0.45 * FS)]
print(f"   hand max inside the missed beat - neighbourhood level: "
      f"{seg.max() - np.median(hand[i - 150:i]):.0f} counts")
