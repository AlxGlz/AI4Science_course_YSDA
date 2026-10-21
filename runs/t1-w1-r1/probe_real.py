"""Read-only probe of real records: statistics used to justify the modulator model.

Does not copy patient data anywhere; prints aggregates only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FS = 320.0
ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "data" / "private"


def load(path: Path) -> np.ndarray:
    with open(path, "r", encoding="cp1251", newline="") as fh:
        text = fh.read()
    rows = [line.split("\t") for line in text.splitlines() if line.strip()]
    arr = np.array([[float(v.replace(",", ".")) for v in row] for row in rows])
    return arr


def bandpass(x: np.ndarray, lo: float, hi: float, fs: float = FS) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def peaks(x: np.ndarray, fs: float = FS) -> np.ndarray:
    from scipy.signal import find_peaks

    y = bandpass(x, 0.5, 8.0, fs)
    p, _ = find_peaks(y, distance=int(0.33 * fs), prominence=0.3 * np.std(y))
    return p


def describe(name: str, hand: np.ndarray, foot: np.ndarray) -> dict:
    ph = peaks(hand)
    pf = peaks(foot)
    ibi = np.diff(ph) / FS
    amp_h = np.array([hand[p] - np.mean(hand[max(0, p - 40):p + 40]) for p in ph])
    amp_f = np.array([foot[p] - np.mean(foot[max(0, p - 40):p + 40]) for p in pf])

    hf = bandpass(hand, 0.5, 8.0)
    ff = bandpass(foot, 0.5, 8.0)

    return {
        "record": name,
        "n_samples": int(hand.size),
        "duration_s": round(hand.size / FS, 1),
        "hand_mean": round(float(np.mean(hand)), 1),
        "hand_std": round(float(np.std(hand)), 1),
        "foot_mean": round(float(np.mean(foot)), 1),
        "foot_std": round(float(np.std(foot)), 1),
        "hr_bpm_median": round(float(60.0 / np.median(ibi)), 1),
        "ibi_s_p05_p95": [round(float(np.percentile(ibi, 5)), 3), round(float(np.percentile(ibi, 95)), 3)],
        "n_peaks_hand": int(ph.size),
        "n_peaks_foot": int(pf.size),
        "peak_match_frac": round(float(np.mean([np.min(np.abs(pf - p)) < 0.15 * FS for p in ph])), 3),
        "amp_hand_median": round(float(np.median(np.abs(amp_h))), 1),
        "amp_foot_median": round(float(np.median(np.abs(amp_f))), 1),
        "foot_to_hand_amp": round(float(np.median(np.abs(amp_f)) / np.median(np.abs(amp_h))), 3),
        "noise_hand_std_hf": round(float(np.std(hand - bandpass(hand, 0.5, 8.0))), 2),
        "noise_foot_std_hf": round(float(np.std(foot - bandpass(foot, 0.5, 8.0))), 2),
        "snr_hand": round(float(np.std(hf) / np.std(hand - bandpass(hand, 0.5, 8.0))), 2),
        "snr_foot": round(float(np.std(ff) / np.std(foot - bandpass(foot, 0.5, 8.0))), 2),
    }


def wave_shape(x: np.ndarray, fs: float = FS) -> dict:
    y = bandpass(x, 0.5, 8.0, fs)
    p = peaks(x, fs)
    p = p[(p > int(fs)) & (p < y.size - int(fs))]
    rise, width, dicrotic = [], [], []
    for pk in p:
        lo = max(0, pk - int(0.4 * fs))
        seg = y[lo:pk]
        base = np.min(seg)
        above = np.where(seg >= base + 0.1 * (y[pk] - base))[0]
        if above.size:
            rise.append((pk - (lo + above[0])) / fs)
        left = pk
        while left > 0 and y[left] > base + 0.5 * (y[pk] - base):
            left -= 1
        right = pk
        while right < y.size - 1 and y[right] > base + 0.5 * (y[pk] - base):
            right += 1
        width.append((right - left) / fs)
        tail = y[pk:min(y.size, pk + int(0.6 * fs))]
        if tail.size > 5:
            dicrotic.append(tail[1:-1].max() / (y[pk] - base))
    return {
        "rise_time_s_median": round(float(np.median(rise)), 3),
        "half_width_s_median": round(float(np.median(width)), 3),
        "dicrotic_ratio_median": round(float(np.median(dicrotic)), 3),
    }


def main() -> None:
    out = []
    for path in sorted(PRIVATE.iterdir()):
        if not path.is_file():
            continue
        arr = load(path)
        hand = arr[:, 0]
        foot = arr[:, 6]
        rec = describe(path.name, hand, foot)
        rec["hand_shape"] = wave_shape(hand)
        rec["foot_shape"] = wave_shape(foot)
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    (Path(__file__).parent / "real_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
