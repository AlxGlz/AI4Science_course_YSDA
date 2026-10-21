"""Inspect the worst 'missed foot beat' on set_hard."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

FS = 320.0
d = Path("runs/t1-w1-r1/sets/set_hard")
data = np.loadtxt(d / "signal.csv", delimiter=",", skiprows=1)
foot_raw = data[:, 2]
foot = foot_raw - median_filter(foot_raw, 321, mode="nearest")
fs_ = uniform_filter1d(foot, 13, mode="nearest")
ref = list(csv.DictReader(open(d / "reference.csv", encoding="utf-8")))
sig = float(np.std(np.diff(foot_raw, n=2)) / np.sqrt(6))

rows = []
for b in ref:
    if b["missed_foot"] != "1":
        continue
    on = float(b["onset_s"])
    lo = on + 0.082 + 0.02
    a = int(round(lo * FS))
    c = int(round((lo + 0.45) * FS))
    i = a + int(np.argmax(fs_[a:c]))
    v = float(np.max(foot[max(0, i - 6):i + 7]))
    near = [float(x["amp_foot"]) for x in ref
            if x["missed_foot"] == "0" and abs(float(x["onset_s"]) - on) < 5]
    refa = max(float(np.median(near)), 3 * sig) if near else 3 * sig
    rows.append((v / refa, int(b["beat_id"]), on, v, refa))
rows.sort(reverse=True)
print(f"noise sigma {sig:.1f}; worst 5 of {len(rows)} missed foot beats:")
for ratio, bid, on, v, refa in rows[:5]:
    print(f" beat {bid} onset={on:.3f} bump={v:.0f} ref={refa:.0f} ratio={ratio:.2f}")
    k = bid
    for x in ref:
        if abs(int(x["beat_id"]) - k) <= 1 and int(x["beat_id"]) != k:
            print(f"    nb {x['beat_id']} onset={float(x['onset_s']):.3f} "
                  f"amp_f={x['amp_foot']} gain={x['foot_gain']} missed_f={x['missed_foot']}")
    a = int(round((on + 0.082 - 0.05) * FS))
    c = a + int(0.55 * FS)
    print("    foot:", " ".join(f"{foot[a + j]:.0f}" for j in range(0, c - a, int(0.03 * FS))))
