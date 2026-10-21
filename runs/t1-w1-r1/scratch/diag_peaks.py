"""Diagnostic: beats where the declared peak is far below the local maximum."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

FS = 320.0
SETS = Path("runs/t1-w1-r1/sets")


def analyse(set_name: str, channel: str, only_worst: int = 4) -> None:
    set_dir = SETS / set_name
    data = np.loadtxt(set_dir / "signal.csv", delimiter=",", skiprows=1)
    raw = data[:, 1] if channel == "hand" else data[:, 2]
    x = raw - median_filter(raw, size=int(1.0 * FS) | 1, mode="nearest")
    ref = list(csv.DictReader(open(set_dir / "reference.csv", encoding="utf-8")))
    bad = []
    for r in ref:
        missed = int(r["missed_hand"] if channel == "hand" else r["missed_foot"])
        if missed:
            continue
        pk = float(r["peak_s"] if channel == "hand" else r["peak_foot_s"])
        on = float(r["onset_s"])
        amp = float(r["amp_hand"] if channel == "hand" else r["amp_foot"])
        lo = max(0, int(round((on - 0.1) * FS)))
        hi = min(x.size, int(round((on + 0.8) * FS)))
        seg = x[lo:hi]
        i = lo + int(np.argmax(seg))
        lvl = x[int(round(pk * FS))] / seg.max() if seg.max() > 0 else 0
        bad.append((lvl, r["beat_id"], on, pk, amp, i / FS, float(seg.max())))
    bad.sort()
    print(f"\n=== {set_name} / {channel}: {sum(1 for b in bad if b[0] < 0.97)} beats below 97%")
    for lvl, bid, on, pk, amp, tmax, vmax in bad[:only_worst]:
        k = int(bid)
        nb = [r for r in ref if abs(int(r["beat_id"]) - k) <= 1 and int(r["beat_id"]) != k]
        print(f" beat {bid} level={lvl:.2f} onset={on:.3f} peak_decl={pk:.3f} amp={amp:.0f} "
              f"argmax_t={tmax:.3f} max={vmax:.0f}")
        for r in nb:
            print(f"    nb {r['beat_id']} onset={float(r['onset_s']):.3f} "
                  f"amp_h={float(r['amp_hand']):.0f} amp_f={float(r['amp_foot']):.0f} "
                  f"missed_h={r['missed_hand']} missed_f={r['missed_foot']}")
        lo = max(0, int(round((on - 0.05) * FS)))
        hi = min(x.size, int(round((on + 0.8) * FS)))
        print("    ", " ".join(f"{(lo + j) / FS - on:+.2f}:{x[lo + j]:.0f}"
                               for j in range(0, hi - lo, int(0.05 * FS))))


for s in ("set_baseline", "set_clean", "set_hard"):
    analyse(s, "hand")
    analyse(s, "foot")
