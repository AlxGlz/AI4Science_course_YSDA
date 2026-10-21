"""Compare baseline estimators against the modulator's own markup (fixed per-channel base)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

FS = 320.0
SETS = Path("runs/t1-w1-r1/sets")


def evaluate(set_dir: Path, name: str, fh, ff) -> None:
    data = np.loadtxt(set_dir / "signal.csv", delimiter=",", skiprows=1)
    hand_raw, foot_raw = data[:, 1], data[:, 2]
    bh, bf = fh(hand_raw), ff(foot_raw)
    ref = list(csv.DictReader(open(set_dir / "reference.csv", encoding="utf-8")))
    errs_h, errs_f = [], []
    for r in ref:
        on = float(r["onset_s"])
        if not int(r["missed_hand"]):
            lo = max(0, int(round(on * FS)))
            hi = min(hand_raw.size, int(round((on + 0.45) * FS)))
            seg = hand_raw[lo:hi] - bh[lo:hi]
            errs_h.append((seg.max() - float(r["amp_hand"])) / float(r["amp_hand"]))
        if not int(r["missed_foot"]):
            pf = float(r["peak_foot_s"])
            lo = max(0, int(round((pf - 0.2) * FS)))
            hi = min(foot_raw.size, int(round((pf + 0.2) * FS)))
            seg = foot_raw[lo:hi] - bf[lo:hi]
            errs_f.append((seg.max() - float(r["amp_foot"])) / float(r["amp_foot"]))
    e_h, e_f = np.array(errs_h), np.array(errs_f)
    print(f"{name:16s} hand bias {100 * np.median(e_h):+6.1f}% p90|err| {100 * np.percentile(np.abs(e_h), 90):5.1f}%"
          f" | foot bias {100 * np.median(e_f):+6.1f}% p90|err| {100 * np.percentile(np.abs(e_f), 90):5.1f}%")


def med(win_s):
    return lambda x: median_filter(x, size=int(win_s * FS) | 1, mode="nearest")


for set_dir in sorted(SETS.iterdir()):
    if not set_dir.is_dir():
        continue
    print(f"\n=== {set_dir.name}")
    for w in (1.0, 1.2, 1.5, 2.0, 3.0):
        evaluate(set_dir, f"median {w}s", med(w), med(w))
