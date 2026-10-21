"""Where does the 13% amplitude deficit on set_hard come from?"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

FS = 320.0
SETS = Path("runs/t1-w1-r1/sets")


def run(set_name: str, channel: str) -> None:
    d = SETS / set_name
    data = np.loadtxt(d / "signal.csv", delimiter=",", skiprows=1)
    raw = data[:, 1] if channel == "hand" else data[:, 2]
    ped = median_filter(raw, size=int(1.0 * FS) | 1, mode="nearest")
    x = raw - ped
    xs = uniform_filter1d(x, int(0.04 * FS) | 1, mode="nearest")
    ref = list(csv.DictReader(open(d / "reference.csv", encoding="utf-8")))
    key_amp = "amp_hand" if channel == "hand" else "amp_foot"
    key_pk = "peak_s" if channel == "hand" else "peak_foot_s"
    key_missed = "missed_hand" if channel == "hand" else "missed_foot"
    r_max, r_smooth, r_plateau, dts = [], [], [], []
    for b in ref:
        if int(b[key_missed]):
            continue
        on, pk, amp = float(b["onset_s"]), float(b[key_pk]), float(b[key_amp])
        lo = max(0, int(round((on - 0.05) * FS)))
        hi = min(x.size, int(round((on + 0.45) * FS)))
        r_max.append(x[lo:hi].max() / amp)
        i = lo + int(np.argmax(xs[lo:hi]))
        r_smooth.append(xs[i] / amp)
        r_plateau.append(np.mean(x[max(0, i - 3):i + 4]) / amp)
        dts.append((i / FS - pk) * 1000)
    print(f"{set_name:13s} {channel:4s} max/decl {np.median(r_max):.3f}  "
          f"smoothed/decl {np.median(r_smooth):.3f}  plateau/decl {np.median(r_plateau):.3f}  "
          f"argmax dt(ms) median {np.median(dts):+.1f} p95 {np.percentile(np.abs(dts), 95):.1f}")


for s in ("set_clean", "set_baseline", "set_hard"):
    run(s, "hand")
    run(s, "foot")
