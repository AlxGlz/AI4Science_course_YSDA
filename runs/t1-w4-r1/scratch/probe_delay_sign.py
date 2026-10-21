"""Проверка знака задержки независимым способом: по пикам каналов.

Кросс-корреляция в probe_real.py даёт -84...-75 мс при соглашении
"roll(yf, s)"; здесь задержка считается как медиана (пик стопы - пик руки)
по сопоставленным ударам.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

PROJECT = Path(__file__).resolve().parents[3]
FS = 320.0


def bandpass(x):
    b, a = butter(3, [0.5 / (FS / 2), 8.0 / (FS / 2)], btype="band")
    return filtfilt(b, a, x - np.median(x))


def main() -> None:
    for p in sorted((PROJECT / "data" / "private").iterdir()):
        if not p.is_file():
            continue
        raw = pd.read_csv(p, sep="\t", decimal=",", header=None, usecols=[0, 6],
                          dtype=float).to_numpy()
        yh, yf = bandpass(raw[:, 0]), bandpass(raw[:, 1])
        ih, _ = find_peaks(yh, distance=int(0.33 * FS), prominence=0.4 * np.std(yh))
        iff, _ = find_peaks(yf, distance=int(0.33 * FS), prominence=0.4 * np.std(yf))
        delays, pairs = [], 0
        for i in ih:
            j = int(np.argmin(np.abs(iff - i)))
            if abs(iff[j] - i) < 0.25 * FS:
                delays.append((iff[j] - i) / FS)
                pairs += 1
        d = np.array(delays)
        print(f"{p.name}: ударов {ih.size}/{iff.size}, сопоставлено {pairs}, "
              f"медиана (стопа-рука) {1000 * np.median(d):+.0f} мс, "
              f"IQR {1000 * (np.percentile(d, 75) - np.percentile(d, 25)):.0f} мс")


if __name__ == "__main__":
    main()
