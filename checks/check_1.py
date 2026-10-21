"""Приёмка модулятора синтетических ФПГ-пар (задача 1).

Запуск из корня проекта:

    .venv\\Scripts\\python checks\\check_1.py
    .venv\\Scripts\\python checks\\check_1.py --run runs/t1-w3-r1
    .venv\\Scripts\\python checks\\check_1.py --run runs/t1-w3-r1 --regenerate
    .venv\\Scripts\\python checks\\check_1.py --no-selftest

По умолчанию проверяется самый свежий каталог runs/t1-*/ с modulator.py.
Код 0 — все проверки прошли.

Проверка читает только выпущенные файлы (signal.csv, reference.csv,
params.json) и измеряет сигнал сама: модулятор не импортируется, поэтому
ошибка, общая для генератора и разметки, не может пройти незамеченной.

Контракт набора и обоснование допусков — в заметках:
    notes/Метод — контракт синтетического набора T1.md
    notes/Метод — приёмка модулятора.md

Два измерительных решения, от которых зависят допуски:
  * опорный уровень — медиана в окне 1 с (окно 3 с смещает амплитуду на
    5-9 %, потому что пульс занимает около половины сердечного цикла);
  * пик ищется на скользящем среднем 40 мс: на плоской вершине белый шум
    30 отсчётов смещает сырой argmax на 20-30 мс.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

PROJECT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- контракт

FS_NOMINAL = 320.0
SIGNAL_COLUMNS = ["t", "hand", "foot"]
REFERENCE_COLUMNS = [
    "beat_id", "onset_s", "peak_s", "peak_foot_s", "amp_hand", "amp_foot",
    "missed_hand", "missed_foot", "foot_gain",
]
REQUIRED_KEYS = [
    "name", "seed", "fs", "duration_s", "hr_bpm", "hr_variability",
    "dc_hand", "dc_foot", "drift_hand", "drift_foot",
    "noise_hand", "noise_foot", "amp_hand", "foot_delay_s",
    "rise_hand_s", "rise_foot_s", "foot_gain",
    "n_attenuation", "attenuation_depth", "n_missed_hand", "n_missed_foot",
]
MIN_SETS = 3
MIN_BEATS = 150
MIN_DURATION_S = 120.0

# ---------------------------------------------------------------- допуски

TOL = {
    "peak_median_s": 0.015,     # медиана |dt| пика
    "peak_p99_hand_s": 0.050,   # p99 |dt| руки
    "peak_p99_foot_s": 0.060,   # p99 |dt| стопы
    "peak_share_hand_s": 0.030, # доля ударов в пределах 30 мс
    "peak_share_foot_s": 0.050,
    "peak_share_min": 0.95,
    "amp_sigma_median": 3.0,    # медиана |ошибки| в сигмах шума
    "amp_sigma_p95": 5.0,
    "amp_ratio": (0.90, 1.12),  # медиана «измерено / объявлено»
    "amp_corr": 0.45,           # корреляция с объявленной модуляцией
    "delay_s": 0.025,           # медиана задержки против объявленной
    "delay_iqr_s": 0.060,       # разброс задержки по ударам
    "rise_diff_s": 0.030,       # прирост времени подъёма стопы
    "gain_cv": 0.20,            # CV(amp_foot / (amp_hand * foot_gain))
    "missed_hand_share": 0.40,  # остаток пропущенной волны руки
    "missed_foot_share": 0.55,  # остаток пропущенной волны стопы
    "noise_rel": 0.25,
    "hr_rel": 0.06,
    "beats_rel": 0.10,
    "ibi_cv_low": 0.40,         # измеренный CV / объявленная вариабельность
    "ibi_cv_high": 2.20,
    "markup_share": 0.98,
}

SUMMARY = {"sets": 0, "beats": 0, "checks": 0}


class Report:
    """Сбор результатов: печатает PASS/FAIL и хранит список."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.results: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.results.append((bool(ok), name, detail))
        if self.verbose:
            line = f"{'PASS' if ok else 'FAIL'}  {name}"
            if detail:
                line += f" — {detail}"
            print(line)
        return bool(ok)

    def note(self, text: str) -> None:
        if self.verbose:
            print(f"      {text}")

    @property
    def failed(self) -> list[tuple[bool, str, str]]:
        return [r for r in self.results if not r[0]]


# ---------------------------------------------------------------- измерение


def load_signal(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8", newline="") as fh:
        header = fh.readline().strip().lstrip("\ufeff")
    if header != ",".join(SIGNAL_COLUMNS):
        raise ValueError(f"{path.name}: заголовок {header!r}, ожидался "
                         f"{','.join(SIGNAL_COLUMNS)!r}")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"{path.name}: ожидалось 3 столбца, получено {data.shape}")
    return data[:, 0], data[:, 1], data[:, 2]


def load_reference(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if fields != REFERENCE_COLUMNS:
            raise ValueError(f"{path.name}: столбцы {fields}, ожидались "
                             f"{REFERENCE_COLUMNS}")
        rows = list(reader)
    out = []
    for r in rows:
        out.append({
            "beat_id": int(r["beat_id"]),
            "onset_s": float(r["onset_s"]),
            "peak_s": float(r["peak_s"]),
            "peak_foot_s": float(r["peak_foot_s"]),
            "amp_hand": float(r["amp_hand"]),
            "amp_foot": float(r["amp_foot"]),
            "missed_hand": bool(int(r["missed_hand"])),
            "missed_foot": bool(int(r["missed_foot"])),
            "foot_gain": float(r["foot_gain"]),
        })
    return out


def load_params(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"{path.name}: ожидался объект с ключом config")
    return payload


def pedestal(x: np.ndarray, fs: float, win_s: float = 1.0) -> np.ndarray:
    """Опорный уровень (DC + дрейф) как медиана в окне 1 с."""
    return median_filter(x, size=int(win_s * fs) | 1, mode="nearest")


def smooth(x: np.ndarray, fs: float, win_s: float = 0.04) -> np.ndarray:
    return uniform_filter1d(x, int(win_s * fs) | 1, mode="nearest")


def noise_sigma(x: np.ndarray) -> float:
    """СКО белого шума по вторым разностям: var(d2) = 6 sigma^2."""
    return float(np.std(np.diff(x, n=2)) / np.sqrt(6.0))


def measure_peak(x_search: np.ndarray, x_value: np.ndarray, lo_s: float, hi_s: float,
                 base_s: float, fs: float) -> tuple[float, float, int, float] | None:
    """Время и высота максимума в [lo_s, hi_s].

    Пик ищется по `x_search` (сигнал с вычтенной базовой линией), а значение
    берётся из `x_value` — того же сигнала без вычитания, но сглаженного.
    Так сделано намеренно: уровень в пике и уровень перед началом волны
    отстоят на десятые доли секунды, поэтому DC и медленный дрейф в разности
    сокращаются, а смещение медианного фильтра (он стоит выше диастолы на
    5-8 %, потому что пульс занимает больше половины окна) в амплитуду
    не попадает. На контрольном прогоне это дало отношение измеренного
    к объявленному 0.97-1.01 вместо 1.02-1.14 при вычитании медианного уровня.
    """
    n = x_search.size
    lo = max(0, int(round(lo_s * fs)))
    hi = min(n, int(round(hi_s * fs)))
    if hi - lo < 3:
        return None
    i = lo + int(np.argmax(x_search[lo:hi]))
    b0 = max(0, int(round((base_s - 0.05) * fs)))
    b1 = max(b0 + 1, min(n, int(round((base_s - 0.01) * fs))))
    base = float(np.median(x_value[b0:b1]))
    a, b = max(0, i - 5), min(n, i + 6)
    if b - a >= 5:
        tt = (np.arange(a, b) - i) / fs  # центрирование: иначе polyfit плохо обусловлен
        coef = np.polyfit(tt, x_search[a:b], 2)
        if coef[0] < 0:
            tv = float(np.clip(-coef[1] / (2 * coef[0]), tt[0], tt[-1]))
            return (i + tv * fs) / fs, float(x_value[i]) - base, i, base
    return i / fs, float(x_value[i]) - base, i, base


def rise_time(x: np.ndarray, i_peak: int, base: float, amp: float, fs: float) -> float:
    """Время от пересечения 15 % амплитуды до пика, с.

    `x` — сглаженный сигнал без вычитания базовой линии, `base` — её уровень:
    порог задаётся как base + 0.15 * amp, поэтому смещение базовой линии
    не сдвигает точку пересечения.
    """
    thr = base + 0.15 * max(amp, 1e-9)
    lo = max(0, i_peak - int(0.45 * fs))
    j = i_peak
    while j > lo and x[j] > thr:
        j -= 1
    if j >= i_peak:
        return float("nan")
    x0, x1 = float(x[j]), float(x[j + 1])
    frac = (thr - x0) / (x1 - x0) if x1 != x0 else 0.0
    return (i_peak - (j + frac)) / fs


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- проверка набора


def check_set(set_dir: Path, rep: Report) -> None:
    tag = f"[{set_dir.name}]"
    cfg = load_params(set_dir / "params.json")["config"]
    t, hand_raw, foot_raw = load_signal(set_dir / "signal.csv")
    ref = load_reference(set_dir / "reference.csv")
    fs = float(cfg["fs"])

    hand = hand_raw - pedestal(hand_raw, fs)
    foot = foot_raw - pedestal(foot_raw, fs)
    hand_s = smooth(hand, fs)
    foot_s = smooth(foot, fs)
    hand_rs = smooth(hand_raw, fs)
    foot_rs = smooth(foot_raw, fs)

    onsets = np.array([b["onset_s"] for b in ref])
    peaks = np.array([b["peak_s"] for b in ref])
    pfoot = np.array([b["peak_foot_s"] for b in ref])

    # 1. формат файлов -------------------------------------------------
    dt = np.diff(t)
    fs_meas = 1.0 / float(np.median(dt))
    rep.check(abs(fs - FS_NOMINAL) < 1e-6, f"{tag} формат: fs = 320 Гц",
              f"в params.json {fs:g}")
    rep.check(abs(fs_meas - fs) < 0.01, f"{tag} формат: шаг времени",
              f"измерено {fs_meas:.4f} Гц")
    rep.check(abs(t[0]) < 1e-3 and t.size == int(round(cfg["duration_s"] * fs)),
              f"{tag} формат: длительность {cfg['duration_s']:.0f} с",
              f"{t.size} отсчётов, последний t = {t[-1]:.3f} с")
    rep.check(bool(np.all(dt > 0)), f"{tag} формат: время строго растёт")
    rep.check(cfg["duration_s"] >= MIN_DURATION_S,
              f"{tag} формат: набор не короче {MIN_DURATION_S:.0f} с",
              f"{cfg['duration_s']:.0f} с")
    quant = bool(np.allclose(hand_raw, np.round(hand_raw))
                 and np.allclose(foot_raw, np.round(foot_raw)))
    rep.check(quant, f"{tag} формат: целые отсчёты АЦП")
    in_range = bool(hand_raw.min() >= 0 and hand_raw.max() <= 65535
                    and foot_raw.min() >= 0 and foot_raw.max() <= 65535)
    saturated = int(np.sum(hand_raw <= 0) + np.sum(hand_raw >= 65535)
                    + np.sum(foot_raw <= 0) + np.sum(foot_raw >= 65535))
    rep.check(in_range and saturated == 0, f"{tag} формат: нет насыщения АЦП",
              f"диапазон {int(min(hand_raw.min(), foot_raw.min()))}…"
              f"{int(max(hand_raw.max(), foot_raw.max()))}")

    # 2. структура разметки --------------------------------------------
    rep.check(len(ref) >= MIN_BEATS, f"{tag} разметка: не меньше {MIN_BEATS} ударов",
              f"{len(ref)} ударов")
    rep.check([b["beat_id"] for b in ref] == list(range(len(ref))),
              f"{tag} разметка: beat_id непрерывны")
    rep.check(bool(np.all(np.diff(onsets) > 0.3)),
              f"{tag} разметка: onset строго возрастают",
              f"min IBI {1000 * float(np.min(np.diff(onsets))):.0f} мс")
    ordered = bool(np.all(peaks > onsets) and np.all(pfoot > peaks))
    rep.check(ordered, f"{tag} разметка: onset < peak_hand < peak_foot")
    rep.check(bool(np.all(onsets >= 0) and np.all(onsets < cfg["duration_s"])),
              f"{tag} разметка: onset внутри записи")

    vis_f = [b for b in ref if not b["missed_foot"]]
    gain_err = max((abs(b["foot_gain"] - b["amp_foot"] / b["amp_hand"])
                    for b in vis_f if b["amp_hand"] > 0), default=0.0)
    rep.check(gain_err < 1e-4, f"{tag} разметка: foot_gain = amp_foot / amp_hand",
              f"макс. расхождение {gain_err:.1e}")
    rep.check(all(b["amp_hand"] > 0 and b["amp_foot"] > 0 for b in vis_f),
              f"{tag} разметка: амплитуды положительны")

    n_mh = sum(b["missed_hand"] for b in ref)
    n_mf = sum(b["missed_foot"] for b in ref)
    rep.check(n_mh == int(cfg["n_missed_hand"]) and n_mf == int(cfg["n_missed_foot"]),
              f"{tag} разметка: число пропусков совпадает с конфигурацией",
              f"{n_mh}/{cfg['n_missed_hand']} рука, {n_mf}/{cfg['n_missed_foot']} стопа")

    rise_h_decl = float(np.median(peaks - onsets))
    rise_f_decl = float(np.median(pfoot - onsets - cfg["foot_delay_s"]))
    rep.check(abs(rise_h_decl - cfg["rise_hand_s"]) < 0.010,
              f"{tag} разметка: время подъёма руки = rise_hand_s",
              f"разметка {1000 * rise_h_decl:.0f} мс, конфиг "
              f"{1000 * cfg['rise_hand_s']:.0f} мс")
    rep.check(abs(rise_f_decl - cfg["rise_foot_s"]) < 0.010,
              f"{tag} разметка: время подъёма стопы = rise_foot_s",
              f"разметка {1000 * rise_f_decl:.0f} мс, конфиг "
              f"{1000 * cfg['rise_foot_s']:.0f} мс")
    rep.check(cfg["rise_foot_s"] > cfg["rise_hand_s"],
              f"{tag} разметка: стопа поднимается медленнее руки",
              f"{1000 * cfg['rise_foot_s']:.0f} vs {1000 * cfg['rise_hand_s']:.0f} мс")
    rep.check(cfg["foot_delay_s"] > 0, f"{tag} разметка: задержка стопы > 0",
              f"{1000 * cfg['foot_delay_s']:.0f} мс")

    amp_med = float(np.median([b["amp_hand"] for b in ref]))
    rep.check(abs(amp_med - cfg["amp_hand"]) / cfg["amp_hand"] < 0.20,
              f"{tag} разметка: базовая амплитуда руки",
              f"медиана {amp_med:.0f}, конфиг {cfg['amp_hand']:.0f}")

    # 3. ритм ----------------------------------------------------------
    ibi = np.diff(onsets)
    hr_meas = 60.0 / float(np.median(ibi))
    rep.check(abs(hr_meas - cfg["hr_bpm"]) / cfg["hr_bpm"] < TOL["hr_rel"],
              f"{tag} ритм: медианная ЧСС {cfg['hr_bpm']:.0f} уд/мин",
              f"измерено {hr_meas:.1f}")
    n_expected = cfg["duration_s"] * cfg["hr_bpm"] / 60.0
    rep.check(abs(len(ref) - n_expected) / n_expected < TOL["beats_rel"],
              f"{tag} ритм: число ударов", f"{len(ref)}, ожидалось ~{n_expected:.0f}")
    cv = float(np.std(ibi) / np.mean(ibi))
    hi = cfg["hr_variability"] * TOL["ibi_cv_high"]
    lo = cfg["hr_variability"] * TOL["ibi_cv_low"]
    rep.check(cv <= max(hi, 0.01) and cv >= min(lo, 0.002) and cv < 0.25,
              f"{tag} ритм: вариабельность IBI",
              f"CV {100 * cv:.1f} %, объявлено {100 * cfg['hr_variability']:.1f} %")

    vis_h = [b for b in ref if not b["missed_hand"]]
    s_hand, s_foot = noise_sigma(hand_raw), noise_sigma(foot_raw)

    # 4. пики и амплитуды руки ----------------------------------------
    dt_h, pa_h, pm_h, rise_h = [], [], [], []
    for b in vis_h:
        got = measure_peak(hand_s, hand_rs, b["onset_s"] - 0.05, b["peak_s"] + 0.20,
                           b["onset_s"], fs)
        if got is None:
            continue
        pk_t, pk_v, i_pk, base_v = got
        dt_h.append(abs(pk_t - b["peak_s"]))
        pa_h.append(b["amp_hand"])
        pm_h.append(pk_v)
        rise_h.append(rise_time(hand_rs, i_pk, base_v, pk_v, fs))
    rep.check(float(np.median(dt_h)) < TOL["peak_median_s"]
              and np.percentile(dt_h, 99) < TOL["peak_p99_hand_s"],
              f"{tag} рука: время пика",
              f"медиана |dt| {1000 * float(np.median(dt_h)):.1f} мс, "
              f"p99 {1000 * float(np.percentile(dt_h, 99)):.0f} мс")
    share_h = float(np.mean(np.array(dt_h) < TOL["peak_share_hand_s"]))
    rep.check(share_h > TOL["peak_share_min"],
              f"{tag} рука: время пика в пределах 30 мс",
              f"{100 * share_h:.1f} % из {len(dt_h)}")

    err_h = np.abs(np.array(pm_h) - np.array(pa_h))
    rep.check(float(np.median(err_h)) < TOL["amp_sigma_median"] * s_hand
              and np.percentile(err_h, 95) < TOL["amp_sigma_p95"] * s_hand,
              f"{tag} рука: амплитуда",
              f"медиана |err| {np.median(err_h):.1f} ({100 * np.median(err_h) / np.median(pa_h):.1f} %), "
              f"p95 {np.percentile(err_h, 95):.1f}; sigma шума {s_hand:.1f}")
    scale_h = float(np.median(pm_h) / np.median(pa_h))
    lo_a, hi_a = TOL["amp_ratio"]
    rep.check(lo_a < scale_h < hi_a,
              f"{tag} рука: систематический масштаб амплитуды",
              f"медиана измерено/объявлено {scale_h:.3f}")
    corr_h = float(np.corrcoef(pa_h, pm_h)[0, 1])
    rep.check(corr_h > TOL["amp_corr"],
              f"{tag} рука: амплитуда следует объявленной модуляции",
              f"корреляция {corr_h:.2f} по {len(pa_h)} ударам")

    # 5. пики, амплитуды и затухание стопы ----------------------------
    dt_f, pa_f, pm_f, rise_f, delays = [], [], [], [], []
    pm_h_pair, gain_pair = [], []
    both = [b for b in ref if not (b["missed_hand"] or b["missed_foot"])]
    for b in both:
        if b["amp_foot"] < 3 * s_foot:
            continue
        got_f = measure_peak(foot_s, foot_rs, b["peak_foot_s"] - 0.25,
                             b["peak_foot_s"] + 0.20,
                             b["onset_s"] + cfg["foot_delay_s"], fs)
        got_h = measure_peak(hand_s, hand_rs, b["onset_s"] - 0.05,
                             b["peak_s"] + 0.20, b["onset_s"], fs)
        if got_f is None or got_h is None:
            continue
        dt_f.append(abs(got_f[0] - b["peak_foot_s"]))
        pa_f.append(b["amp_foot"])
        pm_f.append(got_f[1])
        pm_h_pair.append(got_h[1])
        gain_pair.append(b["foot_gain"])
        rise_f.append(rise_time(foot_rs, got_f[2], got_f[3], got_f[1], fs))
        delays.append(got_f[0] - got_h[0])

    weak_note = ""
    n_weak = len(both) - len(dt_f)
    if n_weak:
        weak_note = f"; {n_weak} ударов ниже 3 sigma шума пропущены"
    rep.check(float(np.median(dt_f)) < TOL["peak_median_s"]
              and np.percentile(dt_f, 99) < TOL["peak_p99_foot_s"],
              f"{tag} стопа: время пика",
              f"медиана |dt| {1000 * float(np.median(dt_f)):.1f} мс, "
              f"p99 {1000 * float(np.percentile(dt_f, 99)):.0f} мс{weak_note}")
    share_f = float(np.mean(np.array(dt_f) < TOL["peak_share_foot_s"]))
    rep.check(share_f > TOL["peak_share_min"],
              f"{tag} стопа: время пика в пределах 50 мс",
              f"{100 * share_f:.1f} % из {len(dt_f)}{weak_note}")
    err_f = np.abs(np.array(pm_f) - np.array(pa_f))
    rep.check(float(np.median(err_f)) < TOL["amp_sigma_median"] * s_foot
              and np.percentile(err_f, 95) < TOL["amp_sigma_p95"] * s_foot,
              f"{tag} стопа: амплитуда",
              f"медиана |err| {np.median(err_f):.1f} "
              f"({100 * np.median(err_f) / np.median(pa_f):.1f} %), "
              f"p95 {np.percentile(err_f, 95):.1f}; sigma шума {s_foot:.1f}")
    scale_f = float(np.median(pm_f) / np.median(pa_f))
    rep.check(lo_a < scale_f < hi_a,
              f"{tag} стопа: систематический масштаб амплитуды",
              f"медиана измерено/объявлено {scale_f:.3f}")
    corr_f = float(np.corrcoef(pa_f, pm_f)[0, 1])
    rep.check(corr_f > TOL["amp_corr"],
              f"{tag} стопа: амплитуда следует объявленной модуляции",
              f"корреляция {corr_f:.2f} по {len(pa_f)} ударам{weak_note}")

    rise_h_meas = float(np.nanmedian(rise_h))
    rise_f_meas = float(np.nanmedian(rise_f))
    expect_rise = cfg["rise_foot_s"] - cfg["rise_hand_s"]
    rep.check(abs((rise_f_meas - rise_h_meas) - expect_rise) < TOL["rise_diff_s"],
              f"{tag} стопа: затухание удлиняет подъём",
              f"прирост {1000 * (rise_f_meas - rise_h_meas):.0f} мс, "
              f"объявлено {1000 * expect_rise:.0f} мс")

    expect_peak = cfg["foot_delay_s"] + cfg["rise_foot_s"] - cfg["rise_hand_s"]
    d_markup = float(np.median(pfoot - peaks))
    rep.check(abs(d_markup - expect_peak) < 0.020,
              f"{tag} стопа: задержка пика в разметке",
              f"{1000 * d_markup:.0f} мс, ожидалось {1000 * expect_peak:.0f} мс")
    d_sig = float(np.median(delays))
    iqr = float(np.percentile(delays, 75) - np.percentile(delays, 25))
    rep.check(abs(d_sig - expect_peak) < TOL["delay_s"] and iqr < TOL["delay_iqr_s"],
              f"{tag} стопа: задержка пика в сигнале",
              f"{1000 * d_sig:.0f} мс (IQR {1000 * iqr:.0f} мс), "
              f"ожидалось {1000 * expect_peak:.0f} мс")

    gains = np.array([b["foot_gain"] for b in vis_f])
    ratio = np.array(pm_f) / (np.array(pm_h_pair) * np.array(gain_pair))
    rep.check(float(np.std(ratio) / np.mean(ratio)) < TOL["gain_cv"],
              f"{tag} стопа: амплитуда следует foot_gain",
              f"CV {100 * float(np.std(ratio) / np.mean(ratio)):.1f} % "
              f"по {len(ratio)} измеренным ударам")
    rep.check(float(gains.max()) <= 1.05,
              f"{tag} стопа: коэффициент передачи не усиливает",
              f"макс. foot_gain {gains.max():.2f}")

    n_att = int(cfg["n_attenuation"])
    if n_att > 0:
        depth = float(cfg["attenuation_depth"])
        base = float(cfg["foot_gain"])
        dips = gains < base * depth * 1.15
        idx = np.where(dips)[0]
        groups: list[list[int]] = []
        vis_onset = np.array([b["onset_s"] for b in vis_f])
        for i in idx:
            if groups and vis_onset[i] - vis_onset[groups[-1][-1]] < 5.0:
                groups[-1].append(i)
            else:
                groups.append([int(i)])
        rep.check(abs(float(gains.min()) / base - depth) < 0.12,
                  f"{tag} стопа: глубина эпизодов затухания",
                  f"мин. отношение {gains.min() / base:.2f}, объявлено {depth:.2f}")
        # перекрывающиеся эпизоды сливаются в один провал, поэтому число
        # наблюдаемых групп не больше объявленного, но не меньше одной
        rep.check(1 <= len(groups) <= n_att,
                  f"{tag} стопа: эпизоды затухания",
                  f"наблюдаемых провалов {len(groups)}, объявлено {n_att} "
                  f"(перекрытия сливаются)")

    # 6. пропуски ------------------------------------------------------
    def local_amp(beat: dict, amp_key: str, missed_key: str, span: float = 5.0) -> float:
        near = [b[amp_key] for b in ref
                if not b[missed_key] and abs(b["onset_s"] - beat["onset_s"]) < span]
        if not near:
            near = [b[amp_key] for b in ref if not b[missed_key]]
        return float(np.median(near))

    bumps_h = []
    for b in [b for b in ref if b["missed_hand"]]:
        got = measure_peak(hand_s, hand_rs, b["onset_s"] + 0.02, b["onset_s"] + 0.45,
                           b["onset_s"], fs)
        if got is not None:
            bumps_h.append(got[1] / max(local_amp(b, "amp_hand", "missed_hand"),
                                        3 * s_hand))
    rep.check(max(bumps_h, default=0.0) < TOL["missed_hand_share"],
              f"{tag} пропуски руки пусты",
              f"{len(bumps_h)} ударов, макс. остаток "
              f"{100 * max(bumps_h, default=0):.0f} % соседней волны")

    bumps_f = []
    floor_f = 6 * s_foot
    for b in [b for b in ref if b["missed_foot"]]:
        lo_f = b["onset_s"] + cfg["foot_delay_s"] + 0.02
        got = measure_peak(foot_s, foot_rs, lo_f, lo_f + 0.45, lo_f, fs)
        if got is not None:
            bumps_f.append(got[1] / max(local_amp(b, "amp_foot", "missed_foot"), floor_f))
    rep.check(max(bumps_f, default=0.0) < TOL["missed_foot_share"],
              f"{tag} пропуски стопы пусты",
              f"{len(bumps_f)} ударов, макс. остаток "
              f"{100 * max(bumps_f, default=0):.0f} % соседней волны или 6 sigma "
              f"({floor_f:.0f} отсчётов)")

    lost_foot = [b for b in ref if b["missed_hand"] and not b["missed_foot"]]
    if lost_foot:
        errs = []
        for b in lost_foot:
            got = measure_peak(foot_s, foot_rs, b["peak_foot_s"] - 0.25,
                               b["peak_foot_s"] + 0.20,
                               b["onset_s"] + cfg["foot_delay_s"], fs)
            if got is not None:
                errs.append(abs(got[1] - b["amp_foot"]) / max(b["amp_foot"], 1e-9))
        rep.check(bool(errs) and float(np.median(errs)) < 0.25,
                  f"{tag} пропуск руки не гасит волну стопы",
                  f"медиана отн. ошибки {100 * float(np.median(errs)):.1f} % "
                  f"на {len(errs)} ударах" if errs else "не нашлось измеримых ударов")

    # 7. полнота разметки независимым детектором ----------------------
    med_amp = float(np.median(pa_h)) if pa_h else 0.0
    prom = max(3 * s_hand, 0.25 * med_amp)
    idx, _ = find_peaks(hand_s, distance=int(0.3 * fs), prominence=prom,
                        height=0.5 * med_amp)
    det_t = idx / fs
    vis_t = np.array([b["peak_s"] for b in vis_h])
    matched = 0
    used = set()
    for tt_ in det_t:
        j = int(np.argmin(np.abs(vis_t - tt_)))
        if abs(vis_t[j] - tt_) < 0.06 and j not in used:
            used.add(j)
            matched += 1
    share_det = matched / max(len(det_t), 1)
    share_mark = matched / max(len(vis_t), 1)
    rep.check(share_det > TOL["markup_share"] and share_mark > TOL["markup_share"],
              f"{tag} полнота разметки: все волны руки размечены",
              f"найдено {len(det_t)} пиков, размечено {len(vis_t)}, "
              f"сопоставлено {matched} ({100 * share_det:.1f} % / "
              f"{100 * share_mark:.1f} %)")

    # 8. шум, DC, дрейф ------------------------------------------------
    rep.check(abs(s_hand - cfg["noise_hand"]) / cfg["noise_hand"] < TOL["noise_rel"],
              f"{tag} шум руки ~{cfg['noise_hand']:.0f} отсчётов",
              f"измерено {s_hand:.1f}")
    rep.check(abs(s_foot - cfg["noise_foot"]) / cfg["noise_foot"] < TOL["noise_rel"],
              f"{tag} шум стопы ~{cfg['noise_foot']:.0f} отсчётов",
              f"измерено {s_foot:.1f}")
    rep.check(s_foot >= 0.95 * s_hand, f"{tag} шум: стопа не тише руки",
              f"{s_foot:.1f} vs {s_hand:.1f}")
    rep.check(med_amp / s_hand > 5.0, f"{tag} SNR руки",
              f"{20 * np.log10(med_amp / s_hand):.1f} дБ")

    dc_h, dc_f = float(np.median(hand_raw)), float(np.median(foot_raw))
    rep.check(abs(dc_h - cfg["dc_hand"]) < 0.10 * cfg["dc_hand"]
              and abs(dc_f - cfg["dc_foot"]) < 0.10 * cfg["dc_foot"],
              f"{tag} DC: уровень в пределах 10 %",
              f"рука {dc_h:.0f} / {cfg['dc_hand']:.0f}, "
              f"стопа {dc_f:.0f} / {cfg['dc_foot']:.0f}")

    drift_h = float(np.std(pedestal(hand_raw, fs)))
    drift_f = float(np.std(pedestal(foot_raw, fs)))
    rep.check(2.0 < drift_h < cfg["drift_hand"] and 2.0 < drift_f < cfg["drift_foot"],
              f"{tag} базовая линия: дрейф есть и не превышает объявленный",
              f"СКО уровня {drift_h:.1f} / {drift_f:.1f} при объявленном "
              f"{cfg['drift_hand']:.0f} / {cfg['drift_foot']:.0f}")

    if rep.verbose:  # мутационные контроли считают копии, их в сводку не берём
        SUMMARY["sets"] += 1
        SUMMARY["beats"] += len(ref)

# ---------------------------------------------------------------- сценарии


def check_scenarios(found: list[tuple[Path, dict]], rep: Report) -> None:
    rep.check(len(found) >= MIN_SETS, "сценарии: наборов не меньше трёх",
              f"{len(found)} наборов")
    roles = {
        "чистый": lambda c: (c["n_missed_hand"] == 0 and c["n_missed_foot"] == 0
                             and c["n_attenuation"] == 0),
        "дрейф и шум": lambda c: (c["drift_hand"] > 0 and c["drift_foot"] > 0
                                  and c["noise_hand"] > 0 and c["noise_foot"] > 0),
        "пропуски в обоих каналах": lambda c: (c["n_missed_hand"] > 0
                                               and c["n_missed_foot"] > 0),
        "эпизоды затухания": lambda c: (c["n_attenuation"] > 0
                                        and 0 < c["attenuation_depth"] <= 0.6),
        "шумная стопа": lambda c: c["noise_foot"] >= 1.2 * c["noise_hand"],
    }
    covered, missing = [], []
    for role, pred in roles.items():
        if any(pred(c) for _, c in found):
            covered.append(role)
        else:
            missing.append(role)
    rep.check(not missing, "сценарии: покрытие ролей",
              "покрыто: " + ", ".join(covered)
              + (("; нет: " + ", ".join(missing)) if missing else ""))

    # согласованность параметров между наборами: где шум/дрейф различаются
    # заметно, измерение обязано различаться в ту же сторону
    pairs_noise = 0
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            ci, cj = found[i][1], found[j][1]
            if max(ci["noise_foot"], cj["noise_foot"]) >= 1.5 * min(ci["noise_foot"],
                                                                   cj["noise_foot"]):
                si = noise_sigma(load_signal(found[i][0] / "signal.csv")[2])
                sj = noise_sigma(load_signal(found[j][0] / "signal.csv")[2])
                lo, hi = (si, sj) if ci["noise_foot"] < cj["noise_foot"] else (sj, si)
                pairs_noise += 1
                rep.check(lo < hi, "сценарии: шум стопы растёт с объявленным",
                          f"{lo:.1f} vs {hi:.1f} отсчётов")
    if not pairs_noise:
        rep.note("сценарии: пары с разным шумом стопы не проверялись "
                 "(объявленный шум различается меньше чем в 1.5 раза)")


# ---------------------------------------------------------------- воспроизводимость


def check_reproducibility(run_dir: Path, rep: Report) -> None:
    modulator = run_dir / "modulator.py"
    with tempfile.TemporaryDirectory() as tmp:
        def build(seed: int, target: str) -> dict[str, Path]:
            cmd = [sys.executable, str(modulator), "--out", tmp, "--preset", "clean",
                   "--preset-name", target, "--seed", str(seed)]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=run_dir)
            if proc.returncode != 0:
                raise RuntimeError(f"modulator.py упал: {proc.stderr.strip()[:400]}")
            return {name: Path(tmp) / target / name
                    for name in ("signal.csv", "reference.csv", "params.json")}

        a = build(20261021, "s1")
        b = build(20261021, "s2")
        same = all(md5(a[n]) == md5(b[n]) for n in a)
        rep.check(same, "воспроизводимость: тот же seed — те же файлы",
                  md5(a["signal.csv"])[:12])
        c = build(20261022, "s3")
        rep.check(md5(a["signal.csv"]) != md5(c["signal.csv"]),
                  "воспроизводимость: другой seed — другой сигнал",
                  md5(c["signal.csv"])[:12])

        probe = load_params(a["params.json"])["config"]
        rep.check(probe["name"] == "clean" and abs(probe["fs"] - FS_NOMINAL) < 1e-6,
                  "воспроизводимость: CLI пишет params.json по контракту",
                  f"name={probe['name']}, fs={probe['fs']:g}")


# ---------------------------------------------------------------- мутационные контроли


def _write_set(dst: Path, t: np.ndarray, hand: np.ndarray, foot: np.ndarray,
               ref: list[dict], params_src: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([t, hand, foot])
    np.savetxt(dst / "signal.csv", data, fmt="%.6f,%d,%d", delimiter="",
               header="t,hand,foot", comments="", newline="\r\n")
    with open(dst / "reference.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REFERENCE_COLUMNS, lineterminator="\r\n")
        writer.writeheader()
        for r in ref:
            writer.writerow({
                "beat_id": int(r["beat_id"]),
                "onset_s": f"{r['onset_s']:.6f}",
                "peak_s": f"{r['peak_s']:.6f}",
                "peak_foot_s": f"{r['peak_foot_s']:.6f}",
                "amp_hand": f"{r['amp_hand']:.3f}",
                "amp_foot": f"{r['amp_foot']:.3f}",
                "missed_hand": int(r["missed_hand"]),
                "missed_foot": int(r["missed_foot"]),
                "foot_gain": f"{r['foot_gain']:.6f}",
            })
    shutil.copyfile(params_src, dst / "params.json")


def check_selftest(found: list[tuple[Path, dict]], rep: Report) -> None:
    """Проверка, которая не падает, бесполезна: убеждаемся, что падает."""
    if not found:
        return
    base_dir, base_cfg = found[0]
    t, hand, foot = load_signal(base_dir / "signal.csv")
    ref0 = load_reference(base_dir / "reference.csv")
    fs = float(base_cfg["fs"])
    hand_i = np.round(hand).astype(np.int64)
    foot_i = np.round(foot).astype(np.int64)
    rng = np.random.default_rng(7)
    s_hand = noise_sigma(hand)

    def run_case(label: str, src: Path, mutate) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "case"
            t2, h2, f2 = t.copy(), hand_i.copy(), foot_i.copy()
            ref2 = [dict(r) for r in load_reference(src / "reference.csv")]
            mutate(t2, h2, f2, ref2)
            _write_set(dst, t2, h2, f2, ref2, src / "params.json")
            sub = Report(verbose=False)
            try:
                check_set(dst, sub)
            except Exception as exc:  # noqa: BLE001 — исключение не считается срабатыванием
                rep.check(False, f"self-test: {label} → проверка упала с "
                                 f"{type(exc).__name__}", str(exc)[:200])
                return
            failed_names = [n for ok, n, _ in sub.results if not ok]
            want = {
                "сдвиг пиков": "рука: время пика",
                "амплитуда стопы в разметке": "стопа: систематический масштаб амплитуды",
                "пропуск заполнен волной": "пропуски руки пусты",
                "потеря седьмого удара": "полнота разметки",
                "шум руки x4": "шум руки",
                "стопа пришла раньше": "стопа: задержка пика",
            }[label]
            hit = any(want in n for n in failed_names)
            rep.check(hit, f"self-test: {label} → падает «{want}»",
                      "упало: " + "; ".join(failed_names[:4]) if failed_names
                      else "ни одна проверка не упала")

    # 1. пики сдвинуты на +80 мс
    def shift_peaks(tt, hh, ff, rr):
        for r in rr:
            r["peak_s"] += 0.08
            r["peak_foot_s"] += 0.08

    run_case("сдвиг пиков", base_dir, shift_peaks)

    # 2. амплитуда стопы в разметке завышена в 1.6 раза
    def inflate_amp(tt, hh, ff, rr):
        for r in rr:
            if not r["missed_foot"]:
                r["amp_foot"] *= 1.6

    run_case("амплитуда стопы в разметке", base_dir, inflate_amp)

    # 3. пропущенный удар заполнен волной
    with_missed = next(((d, c) for d, c in found
                        if any(r["missed_hand"] for r in load_reference(d / "reference.csv"))),
                       None)
    if with_missed is not None:
        src, cfg = with_missed

        def fill_missed(tt, hh, ff, rr):
            rf = load_reference(src / "reference.csv")
            vis = [r for r in rf if not r["missed_hand"]]
            for r in rr:
                if r["missed_hand"]:
                    amp = float(np.median([v["amp_hand"] for v in vis]))
                    hh[:] = np.round(hh + amp * np.exp(
                        -0.5 * ((tt - r["onset_s"] - 0.14) / 0.06) ** 2)).astype(np.int64)
                    break

        run_case("пропуск заполнен волной", src, fill_missed)

    # 4. из разметки удалён каждый седьмой удар
    def drop_rows(tt, hh, ff, rr):
        kept = [r for i, r in enumerate(rr) if i % 7 != 3]
        rr[:] = kept
        for i, r in enumerate(rr):
            r["beat_id"] = i

    run_case("потеря седьмого удара", base_dir, drop_rows)

    # 5. шум руки удвоен в квадрате (x4 по sigma)
    def quadruple_noise(tt, hh, ff, rr):
        hh[:] = np.round(hh + rng.normal(0.0, 3.0 * s_hand, hh.size)).astype(np.int64)

    run_case("шум руки x4", base_dir, quadruple_noise)

    # 6. стопа пришла на 0.2 с раньше
    def shift_foot(tt, hh, ff, rr):
        shift = int(round(0.2 * fs))
        ff[:] = np.round(np.interp(np.arange(ff.size) - shift,
                                   np.arange(ff.size), ff)).astype(np.int64)

    run_case("стопа пришла раньше", base_dir, shift_foot)


# ---------------------------------------------------------------- артефакты задачи


def check_deliverables(run_dir: Path, rep: Report) -> None:
    report = run_dir / "report.md"
    ok = report.exists() and report.stat().st_size > 2000
    rep.check(ok, "отчёт: report.md существует и непуст",
              f"{report.stat().st_size} байт" if report.exists() else "нет файла")
    if not ok:
        return
    text = report.read_text(encoding="utf-8")
    rep.check("[порождено]" in text, "отчёт: абзацы помечены [порождено]")
    names = re.findall(r"[\w./\\-]+\.png", text)
    images = set()
    for name in names:
        p = (run_dir / name.strip("`")).resolve()
        if p.exists() and p.stat().st_size > 0:
            images.add(p)
    rep.check(len(images) >= 2, "отчёт: ссылается на 2 рисунка, файлы на месте",
              f"{len(images)} существующих png из {len(set(names))} упомянутых")
    rep.check(any("checks" in line or "check_1" in line for line in text.splitlines()),
              "отчёт: указана команда проверки")


# ---------------------------------------------------------------- CLI


def resolve_run(path: str | None) -> Path:
    if path:
        run = Path(path)
        if not run.is_absolute():
            run = PROJECT / run
        if not (run / "modulator.py").exists():
            raise SystemExit(f"нет {run / 'modulator.py'}")
        return run
    candidates = [p for p in (PROJECT / "runs").glob("t1-*")
                  if (p / "modulator.py").exists()]
    if not candidates:
        raise SystemExit("не найден ни один runs/t1-*/modulator.py — укажите --run")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser(description="Приёмка модулятора T1")
    ap.add_argument("--run", default=None, help="каталог прогона runs/t1-wX-rY")
    ap.add_argument("--regenerate", action="store_true",
                    help="пересобрать наборы командой modulator.py --all")
    ap.add_argument("--no-selftest", action="store_true",
                    help="не запускать мутационные контроли")
    args = ap.parse_args()

    started = time.time()
    run_dir = resolve_run(args.run)
    sets_dir = run_dir / "sets"
    print(f"прогон: {run_dir.relative_to(PROJECT) if run_dir.is_relative_to(PROJECT) else run_dir}")

    if args.regenerate or not sets_dir.exists():
        print("пересборка наборов: modulator.py --all")
        subprocess.run([sys.executable, str(run_dir / "modulator.py"),
                        "--out", str(sets_dir), "--all"], check=True, cwd=run_dir)

    set_dirs = sorted(d for d in sets_dir.iterdir()
                      if d.is_dir() and (d / "signal.csv").exists())
    if not set_dirs:
        print(f"FAIL  в {sets_dir} нет наборов")
        return 1

    rep = Report()
    found: list[tuple[Path, dict]] = []
    for d in set_dirs:
        cfg = load_params(d / "params.json")["config"]
        missing = [k for k in REQUIRED_KEYS if k not in cfg]
        rep.check(not missing, f"[{d.name}] контракт: обязательные ключи params.json",
                  "нет: " + ", ".join(missing) if missing else "все на месте")
        if missing:
            continue
        found.append((d, cfg))
        check_set(d, rep)

    check_scenarios(found, rep)
    check_reproducibility(run_dir, rep)
    if not args.no_selftest:
        check_selftest(found, rep)
    check_deliverables(run_dir, rep)

    failed = rep.failed
    print("-" * 72)
    print(f"{len(rep.results) - len(failed)}/{len(rep.results)} проверок пройдено; "
          f"наборов {SUMMARY['sets']}, ударов {SUMMARY['beats']}, "
          f"{time.time() - started:.1f} с")
    for _, name, detail in failed:
        print(f"  FAILED: {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
