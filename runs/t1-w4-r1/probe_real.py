"""Замеры по реальным записям data/private/ — обоснование параметров пресетов.

Читает файлы прибора по формату data/signals.md (12 столбцов, табуляция,
десятичная запятая, без заголовка, fs = 320 Гц), берёт столбец 1 (палец руки,
жёлтый) и столбец 7 (палец стопы, жёлтый) и считает:

  * длительность записи и уровень DC по каналам;
  * СКО белого шума по вторым разностям (та же оценка, что в checks/check_1.py);
  * ЧСС по пикам руки в полосе 0.5-8 Гц;
  * задержку стопы относительно руки по кросс-корреляции полосовых сигналов;
  * отношение амплитуд стопа/рука по высотам волн.

Записи не копируются и не изменяются: файлы только читаются.

    .venv\\Scripts\\python runs\\t1-w4-r1\\probe_real.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

PROJECT = Path(__file__).resolve().parents[2]
PRIVATE = PROJECT / "data" / "private"
OUT = Path(__file__).resolve().parent / "real_stats.json"
FS = 320.0


def load_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Столбцы 1 и 7 записи прибора (рука и палец стопы, жёлтый свет)."""
    raw = pd.read_csv(path, sep="\t", decimal=",", header=None,
                      usecols=[0, 6], dtype=float, engine="c").to_numpy()
    return raw[:, 0], raw[:, 1]


def noise_sigma(x: np.ndarray) -> float:
    """СКО белого шума по вторым разностям: var(d2) = 6 sigma^2."""
    return float(np.std(np.diff(x, n=2)) / np.sqrt(6.0))


def bandpass(x: np.ndarray, lo: float = 0.5, hi: float = 8.0) -> np.ndarray:
    b, a = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, x - np.median(x))


def pulse_stats(x: np.ndarray) -> dict:
    y = bandpass(x)
    idx, _ = find_peaks(y, distance=int(0.33 * FS), prominence=0.4 * np.std(y))
    if idx.size < 10:
        return {"beats": int(idx.size)}
    ibi = np.diff(idx) / FS
    return {
        "beats": int(idx.size),
        "hr_bpm": float(60.0 / np.median(ibi)),
        "ibi_cv": float(np.std(ibi) / np.mean(ibi)),
    }


def analyse(path: Path) -> dict:
    hand, foot = load_pair(path)
    n = hand.size
    yh, yf = bandpass(hand), bandpass(foot)

    # задержка стопы: максимум кросс-корреляции в пределах +-0.5 с
    max_lag = int(0.5 * FS)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([np.dot(yh[max_lag:-max_lag], np.roll(yf, s)[max_lag:-max_lag])
                     for s in lags])
    lag = float(lags[int(np.argmax(corr))] / FS)

    # отношение амплитуд по высотам волн
    ih, _ = find_peaks(yh, distance=int(0.33 * FS), prominence=0.4 * np.std(yh))
    ih = ih[(ih > int(FS)) & (ih < n - int(FS))]
    ratios = []
    for i in ih:
        a = max(0, i - int(0.3 * FS))
        b = min(n, i + int(0.3 * FS))
        hi = float(np.max(yh[a:b]) - np.min(yh[a:b]))
        fi = float(np.max(yf[a:b]) - np.min(yf[a:b]))
        if hi > 0:
            ratios.append(fi / hi)

    return {
        "file": path.name,
        "duration_s": n / FS,
        "dc_hand": float(np.median(hand)),
        "dc_foot": float(np.median(foot)),
        "noise_hand": noise_sigma(hand),
        "noise_foot": noise_sigma(foot),
        "foot_delay_s": lag,
        "amp_ratio_foot_hand": float(np.median(ratios)) if ratios else float("nan"),
        "hand": pulse_stats(hand),
        "foot": pulse_stats(foot),
    }


def main() -> None:
    files = sorted(p for p in PRIVATE.iterdir() if p.is_file())
    stats = [analyse(p) for p in files]
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    for s in stats:
        print(f"{s['file']}: {s['duration_s']:.0f} с, ЧСС руки {s['hand'].get('hr_bpm', float('nan')):.1f}, "
              f"шум {s['noise_hand']:.1f}/{s['noise_foot']:.1f}, "
              f"DC {s['dc_hand']:.0f}/{s['dc_foot']:.0f}, "
              f"задержка {1000 * s['foot_delay_s']:.0f} мс, "
              f"стопа/рука {s['amp_ratio_foot_hand']:.2f}")
    hrs = [s["hand"]["hr_bpm"] for s in stats if "hr_bpm" in s["hand"]]
    if hrs:
        print(f"сводка: ЧСС {min(hrs):.1f}-{max(hrs):.1f} уд/мин, "
              f"шум руки {min(s['noise_hand'] for s in stats):.1f}-"
              f"{max(s['noise_hand'] for s in stats):.1f}, "
              f"задержка {1000 * min(s['foot_delay_s'] for s in stats):.0f}-"
              f"{1000 * max(s['foot_delay_s'] for s in stats):.0f} мс")
    print(f"записано: {OUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
