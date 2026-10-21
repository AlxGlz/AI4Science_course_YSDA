"""Пробник реальных записей data/private/ — только агрегаты, данные не копируются.

Даёт числа для обоснования параметров синтетики: уровень АЦП, шум, ЧСС,
амплитуда волны, запаздывание стопа-рука, эпизоды затухания.

Запуск:  .venv\\Scripts\\python runs\\t1-w2-r1\\probe_real.py
Результат: runs/t1-w2-r1/real_stats.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

FS = 320.0
PROJECT = Path(__file__).resolve().parents[2]
PRIVATE = PROJECT / "data" / "private"
OUT = Path(__file__).resolve().parent / "real_stats.json"


def load(path: Path) -> np.ndarray:
    raw = path.read_text(encoding="utf-8", errors="replace")
    rows = [ln.split("\t") for ln in raw.splitlines() if ln.strip()]
    arr = np.array([[float(v.replace(",", ".")) for v in r[:9]] for r in rows])
    return arr


def ped(x: np.ndarray, win_s: float = 1.0) -> np.ndarray:
    return median_filter(x, size=int(win_s * FS) | 1, mode="nearest")


def sigma_noise(x: np.ndarray) -> float:
    return float(np.std(np.diff(x, n=2)) / np.sqrt(6.0))


def analyse(name: str) -> dict:
    a = load(PRIVATE / name)
    hand_raw, foot_raw = a[:, 0], a[:, 6]
    hand = hand_raw - ped(hand_raw)
    foot = foot_raw - ped(foot_raw)
    hs = uniform_filter1d(hand, 13, mode="nearest")
    fs_ = uniform_filter1d(foot, 13, mode="nearest")

    # пики рука/стопа: порог по амплитудному масштабу записи
    def peaks(xs: np.ndarray) -> np.ndarray:
        s = float(np.percentile(np.abs(xs), 99))
        idx, _ = find_peaks(xs, prominence=0.25 * s, distance=int(0.33 * FS))
        return idx

    ph, pf = peaks(hs), peaks(fs_)
    ibi = np.diff(ph) / FS

    def amps(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
        # амплитуда = пик минус уровень в начале подъёма (за 0.15 с до пика)
        return np.array([x[i] - x[max(0, i - int(0.15 * FS))] for i in idx])

    ah = amps(hs, ph)
    af = amps(fs_, pf)
    # окно наибольшей амплитуды — там нет окклюзии плеча
    win = slice(int(0.15 * hs.size), int(0.45 * hs.size))
    ah_w = amps(hs, ph[(ph > int(0.15 * hs.size)) & (ph < int(0.45 * hs.size))])
    af_w = amps(fs_, pf[(pf > int(0.15 * fs_.size)) & (pf < int(0.45 * fs_.size))])

    # запаздывание стопа: средняя задержка по кросс-корреляции в окне 0.05..0.6 с
    h_bp = hs - uniform_filter1d(hs, int(3 * FS) | 1, mode="nearest")
    f_bp = fs_ - uniform_filter1d(fs_, int(3 * FS) | 1, mode="nearest")
    seg = slice(int(0.15 * hs.size), int(0.45 * hs.size))
    lag = np.arange(0, int(0.6 * FS))
    cc = np.array([np.dot(h_bp[seg][: -lag[-1] - 1], f_bp[seg][l: l + h_bp[seg].size - lag[-1] - 1])
                   for l in lag])
    delay_s = float(lag[int(np.argmax(cc))] / FS)

    # ход амплитуды по записи (окно 10 с): видно фазы протокола
    def rolling(x: np.ndarray, idx: np.ndarray) -> list[float]:
        t = idx / FS
        edges = np.arange(0, t[-1] + 10, 10.0)
        return [float(np.median(x[(t >= lo) & (t < lo + 10)])) if np.any((t >= lo) & (t < lo + 10))
                else float("nan") for lo in edges[:-1]]

    return {
        "file": name,
        "duration_s": round(float(a.shape[0] / FS), 1),
        "adc_median_hand": float(np.median(hand_raw)),
        "adc_median_foot": float(np.median(foot_raw)),
        "adc_min_hand": float(hand_raw.min()),
        "adc_max_hand": float(hand_raw.max()),
        "integer_counts": bool(np.allclose(hand_raw, np.round(hand_raw))),
        "noise_sigma_hand": round(sigma_noise(hand_raw), 1),
        "noise_sigma_foot": round(sigma_noise(foot_raw), 1),
        "hr_bpm": round(60.0 / float(np.median(ibi)), 1),
        "ibi_iqr_s": [round(float(np.percentile(ibi, 25)), 3), round(float(np.percentile(ibi, 75)), 3)],
        "amp_hand_med": round(float(np.median(ah)), 1),
        "amp_hand_med_window": round(float(np.median(ah_w)), 1),
        "amp_foot_med_window": round(float(np.median(af_w)), 1),
        "amp_ratio_foot_hand": round(float(np.median(af_w) / np.median(ah_w)), 3),
        "snr_hand_db": round(float(20 * np.log10(np.median(ah) / sigma_noise(hand_raw))), 1),
        "snr_foot_db": round(float(20 * np.log10(np.median(af) / sigma_noise(foot_raw))), 1),
        "foot_vs_hand_delay_s": round(delay_s, 3),
        "amp_hand_roll10s": [round(v, 0) if v == v else None for v in rolling(ah, ph)],
        "amp_foot_roll10s": [round(v, 0) if v == v else None for v in rolling(af, pf)],
        "n_peaks": [int(ph.size), int(pf.size)],
    }


def main() -> None:
    stats = [analyse(p.name) for p in sorted(PRIVATE.iterdir()) if p.is_file()]
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    for s in stats:
        print(f"{s['file']}: {s['duration_s']} s, HR {s['hr_bpm']}, "
              f"amp {s['amp_hand_med_window']}/{s['amp_foot_med_window']} counts, "
              f"ratio {s['amp_ratio_foot_hand']}, noise {s['noise_sigma_hand']}/"
              f"{s['noise_sigma_foot']}, delay {s['foot_vs_hand_delay_s']} s, "
              f"SNR {s['snr_hand_db']}/{s['snr_foot_db']} dB")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
