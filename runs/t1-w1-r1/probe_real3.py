"""Third read-only probe: numerically stable (SOS) decomposition of real records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

FS = 320.0
ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "data" / "private"


def load(path: Path) -> np.ndarray:
    rows = [ln.split("\t") for ln in path.read_text(encoding="cp1251").splitlines() if ln.strip()]
    return np.array([[float(v.replace(",", ".")) for v in r] for r in rows])


def sos(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    s = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band", output="sos")
    return sosfiltfilt(s, x)


def main() -> None:
    out = []
    for path in sorted(PRIVATE.iterdir()):
        if not path.is_file():
            continue
        arr = load(path)
        rec = {"record": path.name}
        for name, col in (("hand", 0), ("foot", 6)):
            x = arr[:, col]
            base = sos(x, 0.01, 0.4)
            pulse = sos(x, 0.5, 8.0)
            noise = x - sos(x, 0.01, 12.0)
            f, pxx = welch(pulse, FS, nperseg=8192)
            band = (f > 0.7) & (f < 3.0)
            amp_env = np.abs(pulse)
            win = int(10 * FS)
            env = np.array([np.std(pulse[i:i + win]) for i in range(0, x.size - win, win)])
            rec[name] = {
                "dc": round(float(np.median(x)), 1),
                "drift_std": round(float(np.std(base)), 1),
                "drift_p1_p99": [round(float(np.percentile(base, 1)), 1),
                                 round(float(np.percentile(base, 99)), 1)],
                "pulse_rms": round(float(np.std(pulse)), 1),
                "pulse_p999": round(float(np.percentile(amp_env, 99.9)), 1),
                "pulse_to_dc_pct": round(100.0 * float(np.std(pulse)) / float(np.median(x)), 2),
                "noise_rms": round(float(np.std(noise)), 1),
                "snr_db": round(20 * np.log10(float(np.std(pulse)) / float(np.std(noise))), 1),
                "noise_over_pulse_pct": round(100.0 * float(np.std(noise)) / float(np.std(pulse)), 1),
                "dominant_hz": round(float(f[band][np.argmax(pxx[band])]), 3),
                "env_cv_10s": round(float(np.std(env) / np.mean(env)), 3),
                "env_min_over_max": round(float(env.min() / env.max()), 3),
            }
        # peak-based hand->foot delay
        h = sos(arr[:, 0], 0.5, 8.0)
        f_ = sos(arr[:, 6], 0.5, 8.0)
        ph, _ = find_peaks(h, distance=int(0.33 * FS), prominence=0.35 * np.std(h))
        pf, _ = find_peaks(f_, distance=int(0.33 * FS), prominence=0.35 * np.std(f_))
        offs = []
        for p in ph:
            cand = pf[np.abs(pf - p) < int(0.3 * FS)]
            if cand.size:
                offs.append((cand[int(np.argmax(f_[cand]))] - p) / FS)
        offs = np.array(offs)
        rec["delay_ms_median"] = round(1000.0 * float(np.median(offs)), 1)
        rec["delay_ms_iqr"] = [round(1000.0 * float(np.percentile(offs, 25)), 1),
                               round(1000.0 * float(np.percentile(offs, 75)), 1)]
        rec["pairs"] = int(offs.size)
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    (Path(__file__).parent / "real_stats3.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
