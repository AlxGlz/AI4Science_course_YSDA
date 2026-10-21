"""Is the amplitude deficit a property of the estimator or of the markup?

Build the same signal with and without noise and compare the estimators.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

sys.path.insert(0, "runs/t1-w1-r1")
import modulator  # noqa: E402

FS = 320.0


def measure(x, xs, lo_s, hi_s):
    lo = max(0, int(round(lo_s * FS)))
    hi = min(x.size, int(round(hi_s * FS)))
    i = lo + int(np.argmax(xs[lo:hi]))
    a, b = max(0, i - 5), min(x.size, i + 6)
    tt = (np.arange(a, b) - i) / FS
    coef = np.polyfit(tt, x[a:b], 2)
    if coef[0] < 0:
        tv = float(np.clip(-coef[1] / (2 * coef[0]), tt[0], tt[-1]))
        return (i + tv * FS) / FS, float(np.polyval(coef, tv)), i / FS, float(x[i]), float(x[a:b].max())
    return i / FS, float(x[i]), i / FS, float(x[i]), float(x[a:b].max())


for noise in (0.0, 45.0):
    cfg = modulator.Config(name="t", seed=303, duration_s=120.0, noise_hand=noise,
                           noise_foot=noise, n_attenuation=0, n_missed_hand=0,
                           n_missed_foot_only=0)
    d = modulator.build(cfg)
    raw = d["hand"].astype(float)
    ped = median_filter(raw, 321, mode="nearest")
    x = raw - ped
    xs = uniform_filter1d(x, 13, mode="nearest")
    ratios = {"parabola": [], "at_smoothed": [], "max11": []}
    for b in d["beats"]:
        if b.amp_hand < 100:
            continue
        _, vp, _, vs, vm = measure(x, xs, b.onset_s - 0.05, b.peak_s + 0.2)
        ratios["parabola"].append(vp / b.amp_hand)
        ratios["at_smoothed"].append(vs / b.amp_hand)
        ratios["max11"].append(vm / b.amp_hand)
    print(f"noise={noise:.0f}: " + "  ".join(
        f"{k} {np.median(v):.3f} (p10 {np.percentile(v, 10):.3f})" for k, v in ratios.items()))
