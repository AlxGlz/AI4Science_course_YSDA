"""Fourth read-only probe: ensemble-averaged beat morphology + 2-Gaussian fit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, find_peaks, sosfiltfilt

FS = 320.0
ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "data" / "private"


def load(path: Path) -> np.ndarray:
    rows = [ln.split("\t") for ln in path.read_text(encoding="cp1251").splitlines() if ln.strip()]
    return np.array([[float(v.replace(",", ".")) for v in r] for r in rows])


def band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    s = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band", output="sos")
    return sosfiltfilt(s, x)


def two_gauss(t, a1, t1, s1, a2, t2, s2):
    return a1 * np.exp(-0.5 * ((t - t1) / s1) ** 2) + a2 * np.exp(-0.5 * ((t - t2) / s2) ** 2)


def mean_beat(x: np.ndarray) -> tuple[np.ndarray, int]:
    y = band(x, 0.5, 8.0)
    p, _ = find_peaks(y, distance=int(0.33 * FS), prominence=0.4 * np.std(y))
    pre, post = int(0.35 * FS), int(0.65 * FS)
    p = p[(p > pre) & (p < y.size - post)]
    segs = np.array([y[i - pre:i + post] for i in p])
    segs = segs - segs[:, :10].mean(axis=1, keepdims=True)
    return np.median(segs, axis=0), pre


def main() -> None:
    out = []
    t = (np.arange(-int(0.35 * FS), int(0.65 * FS))) / FS
    for path in sorted(PRIVATE.iterdir()):
        if not path.is_file():
            continue
        arr = load(path)
        rec = {"record": path.name}
        for name, col in (("hand", 0), ("foot", 6)):
            mb, _ = mean_beat(arr[:, col])
            pk = int(np.argmax(mb))
            amp = mb[pk] - mb[:10].mean()
            base = mb[:10].mean()
            # onset: first crossing of 10 % of amp, walking back from peak
            i = pk
            while i > 0 and mb[i] - base > 0.1 * amp:
                i -= 1
            onset = t[i]
            # dicrotic: largest local max after the peak within 0.6 s
            tail = mb[pk:pk + int(0.6 * FS)]
            tp, _ = find_peaks(tail, prominence=0.03 * amp)
            dict_amp = (tail[tp].max() - base) / amp if tp.size else float("nan")
            dict_time = t[pk + tp[np.argmax(tail[tp])]] - t[pk] if tp.size else float("nan")
            p0 = [amp, t[pk], 0.06, 0.35 * amp, t[pk] + 0.25, 0.12]
            try:
                popt, _ = curve_fit(two_gauss, t, mb - base, p0=p0, maxfev=20000)
                popt = [round(float(v), 4) for v in popt]
            except RuntimeError:
                popt = None
            rec[name] = {
                "onset_to_peak_s": round(float(t[pk] - onset), 3),
                "peak_amp": round(float(amp), 1),
                "dicrotic_rel_amp": round(float(dict_amp), 3),
                "dicrotic_delay_s": round(float(dict_time), 3),
                "fit_a1_t1_s1_a2_t2_s2": popt,
            }
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    (Path(__file__).parent / "real_stats4.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
