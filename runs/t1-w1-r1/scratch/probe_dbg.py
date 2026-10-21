from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

FS = 320.0
p = Path("data/private/ГАА_1")
rows = [l.split("\t") for l in p.read_text(encoding="cp1251").splitlines() if l.strip()]
a = np.array([[float(v.replace(",", ".")) for v in r] for r in rows])


def lowpass(x, fc):
    return sosfiltfilt(butter(4, fc / (FS / 2), btype="low", output="sos"), x)


def band(x, lo, hi):
    return sosfiltfilt(butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band", output="sos"), x)


for name, col in (("hand", 0), ("foot", 6)):
    x = a[:, col]
    pulse = x - lowpass(x, 0.05)
    y = band(x, 0.5, 8.0)
    pk, _ = find_peaks(y, distance=int(0.33 * FS), prominence=0.4 * np.std(y))
    pre, post = int(0.4 * FS), int(0.7 * FS)
    pk = pk[(pk > pre) & (pk < x.size - post)]
    segs = np.array([pulse[i - pre:i + post] for i in pk])
    segs = segs - np.median(segs[:, :12], axis=1, keepdims=True)
    mb = np.median(segs, axis=0)
    amp = mb.max()
    print(f"--- {name}: amp={amp:.0f}  t_peak={(np.argmax(mb) - pre) / FS:+.3f}")
    print(" ".join(f"{(i - pre) / FS:+.2f}:{mb[i] / amp:+.3f}" for i in np.arange(0, mb.size, 8)))
