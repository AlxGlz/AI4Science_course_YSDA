from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

FS = 320.0
d = Path("runs/t1-w1-r1/sets/set_hard")
data = np.loadtxt(d / "signal.csv", delimiter=",", skiprows=1)
raw = data[:, 1]
ped = median_filter(raw, size=321, mode="nearest")
x = raw - ped
xs = uniform_filter1d(x, 13, mode="nearest")
ref = [l.split(",") for l in (d / "reference.csv").read_text(encoding="utf-8").splitlines()[1:]]
shown = 0
for r in ref:
    if r[6] == "1":
        continue
    on, pk, amp = float(r[1]), float(r[2]), float(r[4])
    lo = int(round((on - 0.05) * FS))
    hi = int(round((on + 0.45) * FS))
    imax = lo + int(np.argmax(x[lo:hi]))
    ism = lo + int(np.argmax(xs[lo:hi]))
    val = float(np.mean(x[ism - 3:ism + 4]))
    if val / amp < 0.9 and shown < 3:
        shown += 1
        print(f"beat {r[0]} on={on:.3f} decl_peak={pk:.3f} decl_amp={amp:.0f}")
        print(f"   true max {x[imax]:.0f} at t={imax / FS:.3f} ({(imax / FS - on) * 1000:.0f} ms)")
        print(f"   smooth argmax t={ism / FS:.3f} ({(ism / FS - on) * 1000:.0f} ms) "
              f"raw there {x[ism]:.0f}, mean+-3 {val:.0f}")
        print("   raw (4-sample grid, 12.5 ms):",
              " ".join(f"{x[lo + j]:.0f}" for j in range(0, hi - lo, 4)))
