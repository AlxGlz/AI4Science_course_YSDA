"""Модулятор синтетических пар ФПГ «палец руки — палец стопы» (T1, неделя 2).

Модель сигнала (подробное обоснование — в report.md):

    x_c(t) = DC_c + drift_c(t) + sum_k g_{c,k} * A_{c,k} * shape_c(t - onset_k - delay_c)
             + noise_c(t)

* shape_c — нормированная (max = 1) форма волны: лог-нормальное ядро
  (t/tp)^a exp(a(1 - t/tp)) плюс отражённая диастолическая волна, обрезанная
  гладким окном на [0, W] так, что волна строго равна нулю в начале и в конце
  опоры. Это нужно, чтобы «амплитуда» и «начало волны» в разметке были
  определены точно, а не через сегментацию.
* onset_k — начало волны руки; волна стопы сдвинута на delay_c (время
  распространения пульсовой волны) и шире (медленнее подъём).
* A_{c,k} — амплитуда удара с дыхательной модуляцией; g_{foot,k} — коэффициент
  стопы (базовое ослабление * эпизоды затухания).
* Пропущенный удар = волна в канале не генерируется; в разметке сохраняются
  номинальные положение пика и амплитуда, а отсутствие волны задаётся
  флагом missed_hand / missed_foot.
* Сигнал квантуется в целые отсчёты АЦП после сложения с шумом.

Публичный интерфейс: Config, build(), PRESETS, CLI (--out/--preset/--preset-name/--all).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FS = 320.0
OVERSAMPLE = 8


@dataclass
class Config:
    """Параметры одного набора синтетики."""

    name: str = "clean"
    seed: int = 20251021
    duration_s: float = 300.0
    fs: float = FS

    # ритм
    hr_bpm: float = 72.0
    hr_variability: float = 0.05     # дыхательная синусовая модуляция IBI
    ibi_jitter: float = 0.012        # случайная часть IBI, доля от среднего
    resp_period_s: float = 4.4

    # шум (СКО белого шума, отсчёты АЦП)
    noise_hand: float = 18.0
    noise_foot: float = 24.0

    # форма и задержка волны
    foot_delay_s: float = 0.08       # время распространения рука -> стопа
    rise_hand_s: float = 0.14        # заданное положение пика от начала волны
    rise_foot_s: float = 0.18
    width_frac: float = 0.56         # опора волны, доля минимального IBI
    width_min_s: float = 0.28
    a_hand: float = 9.0              # крутизна лог-нормального ядра
    a_foot: float = 8.0
    refl_hand: float = 0.18          # амплитуда диастолической волны, доля
    refl_foot: float = 0.25
    refl_frac: float = 0.50          # положение отражённой волны: tp + frac*(W-tp)

    # амплитуды
    amp_hand: float = 240.0
    amp_mod: float = 0.08            # дыхательная модуляция амплитуды
    amp_jitter: float = 0.02
    foot_gain: float = 0.85          # базовое ослабление стопы
    foot_gain_mod: float = 0.04

    # затухание стопы
    n_attenuation: int = 0
    attenuation_depth: float = 0.45  # минимальный множитель затухания
    attenuation_dur_s: tuple = (50.0, 70.0)

    # пропуски
    n_missed_hand: int = 0
    n_missed_foot: int = 0

    # медленная составляющая
    dc_hand: float = 15500.0
    dc_foot: float = 22000.0
    drift_hand: float = 250.0
    drift_foot: float = 350.0


def _preset(name: str) -> Config:
    if name == "clean":
        return Config(name="clean", seed=1201, duration_s=300.0, hr_bpm=72.0,
                      noise_hand=16.0, noise_foot=22.0, amp_hand=240.0,
                      foot_gain=0.85, n_attenuation=0, n_missed_hand=0,
                      n_missed_foot=0, drift_hand=250.0, drift_foot=350.0)
    if name == "baseline":
        return Config(name="baseline", seed=1202, duration_s=420.0, hr_bpm=80.0,
                      noise_hand=22.0, noise_foot=26.0, amp_hand=210.0,
                      foot_gain=0.80, n_attenuation=2, attenuation_depth=0.50,
                      attenuation_dur_s=(50.0, 70.0), n_missed_hand=2,
                      n_missed_foot=2, drift_hand=350.0, drift_foot=500.0)
    if name == "hard":
        return Config(name="hard", seed=1203, duration_s=360.0, hr_bpm=90.0,
                      noise_hand=28.0, noise_foot=30.0, amp_hand=205.0,
                      foot_gain=0.75, n_attenuation=3, attenuation_depth=0.40,
                      attenuation_dur_s=(40.0, 65.0), n_missed_hand=4,
                      n_missed_foot=5, drift_hand=500.0, drift_foot=750.0,
                      hr_variability=0.06, ibi_jitter=0.015)
    raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")


PRESETS = {"clean": lambda: _preset("clean"),
           "baseline": lambda: _preset("baseline"),
           "hard": lambda: _preset("hard")}


# --------------------------------------------------------------------------- форма


def _lognormal(tau: np.ndarray, tp: float, a: float) -> np.ndarray:
    """Ядро (t/tp)^a exp(a(1 - t/tp)): 0 при t = 0, максимум 1 при t = tp."""
    x = np.asarray(tau, dtype=float) / tp
    out = np.zeros_like(x)
    m = x > 0
    out[m] = np.exp(a * (1.0 - x[m]) + a * np.log(x[m]))
    return out


def _shape(tp: float, W: float, a: float, r: float,
           refl_frac: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Нормированная форма волны на [0, W], сетка в OVERSAMPLE раз плотнее fs.

    Возвращает (tau, shape, tp_eff), где tp_eff — эффективное положение пика
    (с уточнением параболой), оно же попадает в разметку и в params.
    Отражённая (диастолическая) волна ставится в tp + refl_frac*(W - tp),
    чтобы при любом ритме оставаться внутри опоры.
    """
    edge = 0.80 * W
    tp2 = tp + refl_frac * (W - tp)
    n = int(np.ceil(W * FS * OVERSAMPLE)) + 1
    tau = np.linspace(0.0, W, n)
    f = _lognormal(tau, tp, a) + r * _lognormal(tau, tp2, a * 0.55)
    y = np.clip((W - tau) / (W - edge), 0.0, 1.0)
    f = f * np.sin(0.5 * np.pi * y) ** 2
    f /= float(f.max())
    i = int(np.argmax(f))
    tp_eff = float(tau[i])
    if 0 < i < n - 1:
        tt = tau[i - 1:i + 2]
        coef = np.polyfit(tt, f[i - 1:i + 2], 2)
        if coef[0] < 0:
            tv = float(np.clip(-coef[1] / (2 * coef[0]), tt[0], tt[-1]))
            tp_eff = tv
    return tau, f, tp_eff


def _drift(rng: np.random.Generator, t: np.ndarray, amp: float) -> np.ndarray:
    """Медленный дрейф DC: несколько синусоид (11-97 с) + линейный тренд."""
    d = np.zeros_like(t)
    for period in (11.0, 23.0, 47.0, 97.0):
        d += rng.uniform(0.10, 0.35) * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    d += rng.uniform(-0.4, 0.4) * (2.0 * t / t[-1] - 1.0)
    return amp * d


def _onsets(cfg: Config, rng: np.random.Generator, n_samples: int) -> np.ndarray:
    ibi0 = 60.0 / cfg.hr_bpm
    phase = rng.uniform(0.0, 2 * np.pi)
    out: list[float] = []
    o = 0.25
    while o < cfg.duration_s:
        out.append(o)
        ibi = ibi0 * (1.0 + cfg.hr_variability * np.sin(2 * np.pi * o / cfg.resp_period_s + phase))
        ibi += rng.normal(0.0, cfg.ibi_jitter * ibi0)
        o += max(ibi, 0.35)
    tail = cfg.foot_delay_s + max(cfg.width_min_s, cfg.width_frac * 60.0 / cfg.hr_bpm) + 0.35
    return np.array([x for x in out if x + tail <= n_samples / cfg.fs])


def _gain_curve(cfg: Config, rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    """Множитель затухания стопы во времени (1 = норма, depth = минимум)."""
    g = np.ones_like(t)
    for _ in range(cfg.n_attenuation):
        centre = rng.uniform(0.12, 0.88) * cfg.duration_s
        width = rng.uniform(*cfg.attenuation_dur_s)
        w = np.clip(1.0 - np.abs(t - centre) / (0.5 * width), 0.0, 1.0)
        g -= (1.0 - cfg.attenuation_depth) * np.sin(0.5 * np.pi * w) ** 2
    return np.clip(g, cfg.attenuation_depth, 1.0)


def _pick_missed(rng: np.random.Generator, count: int, n: int, used: set[int]) -> set[int]:
    chosen: set[int] = set()
    guard = 0
    while len(chosen) < count and guard < 200 * max(count, 1):
        i = int(rng.integers(2, max(3, n - 3)))
        if any(abs(i - j) < 2 for j in used | chosen):
            guard += 1
            continue
        chosen.add(i)
        guard += 1
    return chosen


def build(cfg: Config | None = None) -> dict:
    """Построить набор: сигналы, разметку, конфигурацию."""
    cfg = cfg or _preset("clean")
    rng = np.random.default_rng(cfg.seed)
    n = int(round(cfg.duration_s * cfg.fs))
    t = np.arange(n) / cfg.fs

    onsets = _onsets(cfg, rng, n)
    ibi_min = float(np.min(np.diff(onsets)))
    W = max(cfg.width_min_s, min(cfg.width_frac * ibi_min, 0.95 * ibi_min - 0.12))
    if W < cfg.width_min_s - 1e-9:
        raise ValueError(f"опора волны не помещается в IBI: min IBI {ibi_min:.3f} s")

    tau_h, shape_h, rise_h = _shape(cfg.rise_hand_s, W, cfg.a_hand, cfg.refl_hand, cfg.refl_frac)
    tau_f, shape_f, rise_f = _shape(cfg.rise_foot_s, W, cfg.a_foot, cfg.refl_foot, cfg.refl_frac)

    attenuation = _gain_curve(cfg, rng, t)
    missed_h = _pick_missed(rng, cfg.n_missed_hand, onsets.size, set())
    missed_f = _pick_missed(rng, cfg.n_missed_foot, onsets.size, missed_h)

    hand = np.zeros(n)
    foot = np.zeros(n)
    rows: list[dict] = []
    phase = rng.uniform(0.0, 2 * np.pi)

    for k, onset in enumerate(onsets):
        amp_h = cfg.amp_hand * (1.0 + cfg.amp_mod * np.sin(2 * np.pi * onset / cfg.resp_period_s + phase))
        amp_h *= 1.0 + rng.normal(0.0, cfg.amp_jitter)
        gain_f = cfg.foot_gain * attenuation[min(n - 1, int(round(onset * cfg.fs)))]
        gain_f *= 1.0 + rng.normal(0.0, cfg.foot_gain_mod)

        miss_h, miss_f = k in missed_h, k in missed_f
        if not miss_h:
            _add(hand, t, onset, amp_h, tau_h, shape_h)
        if not miss_f:
            _add(foot, t, onset + cfg.foot_delay_s, amp_h * gain_f, tau_f, shape_f)

        rows.append({
            "beat_id": k,
            "onset_s": round(float(onset), 6),
            "peak_s": round(float(onset + rise_h), 6),
            "peak_foot_s": round(float(onset + cfg.foot_delay_s + rise_f), 6),
            # амплитуды номинальные: это высота волны, которая должна была быть
            # в этом ударе; отсутствие волны задаётся флагами missed_*
            "amp_hand": round(float(amp_h), 3),
            "amp_foot": 0.0 if miss_f else round(float(amp_h * gain_f), 3),
            "missed_hand": int(miss_h),
            "missed_foot": int(miss_f),
            "foot_gain": 0.0 if miss_f else round(float(gain_f), 6),
        })

    hand_raw = cfg.dc_hand + _drift(rng, t, cfg.drift_hand) + hand + rng.normal(0.0, cfg.noise_hand, n)
    foot_raw = cfg.dc_foot + _drift(rng, t, cfg.drift_foot) + foot + rng.normal(0.0, cfg.noise_foot, n)

    hand_int = np.clip(np.round(hand_raw), 0, 65535).astype(np.int64)
    foot_int = np.clip(np.round(foot_raw), 0, 65535).astype(np.int64)
    if np.any(hand_int != np.round(hand_raw)) or np.any(foot_int != np.round(foot_raw)):
        raise ValueError("сигнал вышел за диапазон АЦП 0..65535")

    cfg_out = asdict(cfg)
    cfg_out["rise_hand_s"] = round(rise_h, 6)     # эффективное положение пика
    cfg_out["rise_foot_s"] = round(rise_f, 6)
    cfg_out["design"] = {
        "rise_hand_s": cfg.rise_hand_s, "rise_foot_s": cfg.rise_foot_s,
        "width_s": W, "width_frac": cfg.width_frac,
        "a_hand": cfg.a_hand, "a_foot": cfg.a_foot,
        "n_beats": int(onsets.size), "min_ibi_s": round(ibi_min, 4),
        "n_missed_hand": int(len(missed_h)), "n_missed_foot": int(len(missed_f)),
    }
    return {"t": t, "hand": hand_int, "foot": foot_int, "reference": rows, "config": cfg_out}


def _add(sig: np.ndarray, t: np.ndarray, onset: float, amp: float,
         tau: np.ndarray, shape: np.ndarray) -> None:
    i0 = int(np.ceil(onset * FS))
    i1 = min(sig.size, int(np.floor((onset + tau[-1]) * FS)) + 1)
    if i1 <= i0:
        return
    sig[i0:i1] += amp * np.interp(t[i0:i1] - onset, tau, shape, left=0.0, right=0.0)


# --------------------------------------------------------------------------- вывод


def save(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([result["t"], result["hand"], result["foot"]])
    np.savetxt(out_dir / "signal.csv", data, fmt="%.6f,%d,%d", delimiter="",
               header="t,hand,foot", comments="", newline="\r\n")

    cols = ["beat_id", "onset_s", "peak_s", "peak_foot_s", "amp_hand", "amp_foot",
            "missed_hand", "missed_foot", "foot_gain"]
    with open(out_dir / "reference.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\r\n")
        w.writeheader()
        for r in result["reference"]:
            w.writerow({c: (f"{r[c]:.6f}" if isinstance(r[c], float) else r[c]) for c in cols})

    payload = {"config": result["config"], "preset": result["config"]["name"]}
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    (out_dir / "params.json").write_bytes(text.replace("\n", "\r\n").encode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Синтетические пары ФПГ рука-стопа")
    ap.add_argument("--out", required=True, type=Path, help="каталог для наборов")
    ap.add_argument("--preset", default="clean", choices=sorted(PRESETS))
    ap.add_argument("--preset-name", default=None, help="имя подкаталога набора")
    ap.add_argument("--all", action="store_true", help="сгенерировать все пресеты")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--duration-s", type=float, default=None)
    args = ap.parse_args()

    names = sorted(PRESETS) if args.all else [args.preset]
    for nm in names:
        cfg = PRESETS[nm]()
        if args.seed is not None:
            cfg.seed = args.seed
        if args.duration_s is not None:
            cfg.duration_s = args.duration_s
        res = build(cfg)
        target = args.out / (args.preset_name or f"set_{nm}") if not args.all else args.out / f"set_{nm}"
        save(res, target)
        n_miss = len(res["reference"]) - sum(
            1 for r in res["reference"] if not (r["missed_hand"] or r["missed_foot"]))
        print(f"{target}: {len(res['reference'])} ударов, {res['config']['design']['n_beats']} "
              f"шагов, пропусков {n_miss}, fs {cfg.fs:.0f} Гц, {cfg.duration_s:.0f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
