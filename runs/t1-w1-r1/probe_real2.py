"""Second read-only probe: baseline / pulse / noise decomposition of real records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, medfilt, welch

FS = 320.0
ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "data" / "private"


def load(path: Path) -> np.ndarray:
    rows = [ln.split("\t") for ln in path.read_text(encoding="cp1251").splitlines() if ln.strip()]
    return np.array([[float(v.replace(",", ".")) for v in r] for r in rows])


def bp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    b, a = butter(2, [lo / (FS / 2), min(hi, 0.95 * FS / 2) / (FS / 2)], btype="band")
    return filtfilt(b, a, x)


def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = medfilt(x, kernel_size=int(1.5 * FS) | 1)
    pulse = x - base
    noise = x - bp(x, 0.3, 12.0)
    return base, pulse, noise


def delay_ms(hand: np.ndarray, foot: np.ndarray, max_lag_s: float = 1.0) -> float:
    a = bp(hand, 0.5, 8.0)
    b = bp(foot, 0.5, 8.0)
    n = int(max_lag_s * FS)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    lags = range(-n, n + 1, 1)
    cc = [float(np.corrcoef(a[max(0, l):a.size + min(0, l)], b[max(0, -l):b.size + min(0, -l)])[0, 1])
          for l in lags]
    best = list(lags)[int(np.argmax(cc))]
    return round(1000.0 * best / FS, 1), round(max(cc), 3)


def main() -> None:
    out = []
    for path in sorted(PRIVATE.iterdir()):
        if not path.is_file():
            continue
        arr = load(path)
        rec = {"record": path.name}
        for name, col in (("hand", 0), ("foot", 6)):
            x = arr[:, col]
            base, pulse, noise = split(x)
            rms_pulse = float(np.std(bp(x, 0.5, 8.0)))
            f, pxx = welch(pulse, FS, nperseg=4096)
            band = (f > 0.7) & (f < 3.0)
            f0 = float(f[band][np.argmax(pxx[band])])
            # amplitude envelope of the pulse over 10 s windows -> attenuation episodes
            win = int(10 * FS)
            rms_win = np.array([np.std(bp(x[i:i + win], 0.5, 8.0)) for i in range(0, x.size - win, win)])
            rec[name] = {
                "dc": round(float(np.median(x)), 1),
                "baseline_std": round(float(np.std(base)), 1),
                "baseline_range_p1_p99": [round(float(np.percentile(base, 1)), 1),
                                          round(float(np.percentile(base, 99)), 1)],
                "pulse_rms": round(rms_pulse, 1),
                "pulse_peak_amp_median_p99": round(float(np.percentile(np.abs(pulse), 99.5)), 1),
                "pulse_to_dc_pct": round(100.0 * rms_pulse / float(np.median(x)), 2),
                "noise_rms_gt12hz": round(float(np.std(noise)), 1),
                "snr_db": round(20 * np.log10(rms_pulse / float(np.std(noise))), 1),
                "noise_rel_to_pulse_pct": round(100.0 * float(np.std(noise)) / rms_pulse, 1),
                "dominant_hz": round(f0, 3),
                "env_cv_10s": round(float(np.std(rms_win) / np.mean(rms_win)), 3),
                "env_min_over_max": round(float(rms_win.min() / rms_win.max()), 3),
            }
        rec["delay_ms"], rec["delay_cc"] = delay_ms(arr[:, 0], arr[:, 6])
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    (Path(__file__).parent / "real_stats2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
