"""Compare two drift-removal schemes for amplitude measurement."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

FS = 320.0
SETS = Path("runs/t1-w1-r1/sets")


def lowpass(x: np.ndarray, fc: float = 0.4) -> np.ndarray:
    return sosfiltfilt(butter(4, fc / (FS / 2), btype="low", output="sos"), x)


def report(tag: str, est_amp, decl, meas) -> None:
    decl, meas = np.array(decl), np.array(meas)
    err = meas / decl - 1
    slope = np.polyfit(decl, meas, 1)[0]
    print(f"  {tag:26s} bias {100 * np.median(err):+6.1f}%  "
          f"p90|err| {100 * np.percentile(np.abs(err), 90):5.1f}%  slope {slope:.3f}")


for set_name in ("set_clean", "set_baseline", "set_hard"):
    d = SETS / set_name
    data = np.loadtxt(d / "signal.csv", delimiter=",", skiprows=1)
    ref = list(csv.DictReader(open(d / "reference.csv", encoding="utf-8")))
    print(f"=== {set_name}")
    for channel in ("hand", "foot"):
        raw = data[:, 1] if channel == "hand" else data[:, 2]
        key_amp = "amp_hand" if channel == "hand" else "amp_foot"
        key_pk = "peak_s" if channel == "hand" else "peak_foot_s"
        key_missed = "missed_hand" if channel == "hand" else "missed_foot"

        x1 = raw - median_filter(raw, 321, mode="nearest")
        x2 = raw - lowpass(raw)

        decl1, m1, decl2, m2, decl3, m3 = [], [], [], [], [], []
        for b in ref:
            if int(b[key_missed]):
                continue
            on, pk, amp = float(b["onset_s"]), float(b[key_pk]), float(b[key_amp])
            lo = int(round((on - 0.05) * FS))
            hi = min(raw.size, int(round((on + 0.45) * FS)))
            decl1.append(amp); m1.append(float(x1[lo:hi].max()))
            seg = x2[lo:hi]
            decl2.append(amp); m2.append(float(np.percentile(seg, 99.5) - np.percentile(seg, 5)))
            decl3.append(amp); m3.append(float(seg.max() - np.percentile(seg, 5)))
        report(f"{channel} median1s + max", None, decl1, m1)
        report(f"{channel} lp0.4 + p99.5-p5", None, decl2, m2)
        report(f"{channel} lp0.4 + max-p5", None, decl3, m3)
