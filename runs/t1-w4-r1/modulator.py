"""modulator.py — синтетические пары ФПГ «палец руки — палец стопы» (задача T1).

Модель взята из заметок проекта, а не придумана заново:

    x_c(t) = DC_c + d_c(t) + sum_k g_{c,k} A_{c,k} s_c(t - onset_k) + n_c(t)

* s_hand — model-based PWD: логнормальное систолическое ядро плюс отражённая
  диастолическая компонента (`Тезис — PWD раскладывает волну на гауссовы и
  логнормальные компоненты`). Форма нормирована так, что её максимум равен 1
  и стоит ровно в `rise_hand_s` после начала волны.
* s_foot — выход однополюсного ФНЧ с временем константы tau от ТОЙ ЖЕ волны
  (`Формула — однополюсный фильтр затухания`). Один параметр tau задаёт сразу
  затухание амплитуды, рост времени подъёма и задержку — как требует
  `Тезис — На стопе пульс затухает и запаздывает`.
* A_{c,k} = amp_c * (1 + m sin(2 pi t_k / T_resp + phi)) — дыхательная
  модуляция амплитуды (`Метод — модель волны следующей версии`, п. 3).
* g_hand,k = 1; g_foot,k = foot_gain * a(t_k), где a(t) — медленная кривая
  эпизодов затухания со минимумом `attenuation_depth` от базового усиления.
* d_c(t) — дрейф ниже 0.5 Гц; n_c(t) — независимый по каналам белый шум;
  отсчёты квантуются в целые АЦП 0..65535 (`data/signals.md`).
* Пропуски — независимые по каналам: волна не генерируется только в своём
  канале, второй канал в этом цикле не меняется.

Разметка считается аналитически из формы, а не находится по сигналу:
    peak_s      = onset_s + rise_hand_s              (максимум s_hand)
    peak_foot_s = onset_s + T_peak(tau)              (максимум ФНЧ-выхода)
    foot_delay_s = theta15_foot - theta15_hand       (сдвиг точки 15 % амплитуды)
    rise_foot_s  = T_peak - foot_delay_s             (> rise_hand_s)

Запуск:
    python modulator.py --out DIR --all
    python modulator.py --out DIR --preset clean [--preset-name NAME] [--seed N]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

FS = 320.0
TWO_PI = 2.0 * np.pi
ADC_MAX = 65535


# --------------------------------------------------------------- форма волны


@dataclass(frozen=True)
class Shape:
    """Параметры model-based PWD формы волны (доли времени подъёма)."""

    sigma: float = 0.26        # ширина логнормального систолического ядра
    refl_amp: float = 0.28     # амплитуда отражённой (диастолической) компоненты
    refl_at: float = 1.85      # её положение
    refl_sigma: float = 0.42   # её ширина


def pwd(shape: Shape, xi: np.ndarray) -> np.ndarray:
    """Ненормированная форма w(xi): w(0) = 0, максимум около xi = 1."""
    out = np.zeros_like(xi, dtype=float)
    pos = xi > 0
    x = xi[pos]
    core = np.exp(-np.log(x) ** 2 / (2.0 * shape.sigma ** 2))
    refl = shape.refl_amp * np.exp(
        -np.log(x / shape.refl_at) ** 2 / (2.0 * shape.refl_sigma ** 2))
    out[pos] = core + refl
    return out


def xi_peak(shape: Shape) -> float:
    xi = np.linspace(1e-4, 8.0, 80001)
    return float(xi[int(np.argmax(pwd(shape, xi)))])


def unit_shape(shape: Shape, rise_s: float, fs: float = FS,
               span_s: float = 1.2) -> np.ndarray:
    """Единичная волна: 0 в начале, максимум 1 в точке rise_s."""
    top = xi_peak(shape)
    n = int(round(span_s * fs)) + 1
    theta = np.arange(n) / fs
    w = pwd(shape, theta * top / rise_s)
    return w / w.max()


def lowpass_single_pole(x: np.ndarray, tau_s: float, fs: float = FS) -> np.ndarray:
    """y[n] = y[n-1] + a (x[n] - y[n-1]), a = dt / (tau + dt)."""
    dt = 1.0 / fs
    a = dt / (tau_s + dt)
    return lfilter([a], [1.0, -(1.0 - a)], x, zi=[0.0])[0]


def cross_up(x: np.ndarray, level: float, fs: float = FS) -> float:
    """Первый момент роста сигнала через level, с (линейная интерполяция)."""
    above = np.flatnonzero(x > level)
    if above.size == 0:
        return float("nan")
    j = int(above[0])
    if j == 0:
        return 0.0
    x0, x1 = float(x[j - 1]), float(x[j])
    frac = (level - x0) / (x1 - x0) if x1 != x0 else 0.0
    return (j - 1 + frac) / fs


def argmax_sub(x: np.ndarray, fs: float = FS) -> float:
    """Положение максимума с параболическим уточнением, с."""
    i = int(np.argmax(x))
    if 0 < i < x.size - 1:
        y0, y1, y2 = float(x[i - 1]), float(x[i]), float(x[i + 1])
        den = y0 - 2.0 * y1 + y2
        if den != 0.0:
            return (i + 0.5 * (y0 - y2) / den) / fs
    return i / fs


@dataclass(frozen=True)
class FootShape:
    """Следствия фильтрации: затухание, задержка и время подъёма стопы."""

    tau_s: float
    gain: float          # gamma — коэффициент передачи ФНЧ на одиночной волне
    peak_s: float        # T_peak: задержка максимума относительно начала волны
    delay_s: float       # сдвиг точки 15 % амплитуды стопы против руки
    rise_s: float        # peak_s - delay_s (> rise_hand_s)


def foot_shape(shape: Shape, rise_hand_s: float, tau_s: float,
               fs: float = FS) -> FootShape:
    u = unit_shape(shape, rise_hand_s, fs)
    v = lowpass_single_pole(np.concatenate([u, np.zeros(int(0.8 * fs))]), tau_s, fs)
    theta_hand = cross_up(u, 0.15 * u.max(), fs)
    theta_foot = cross_up(v, 0.15 * v.max(), fs)
    t_peak = argmax_sub(v, fs)
    delay = theta_foot - theta_hand
    return FootShape(tau_s=tau_s, gain=float(v.max()), peak_s=t_peak,
                     delay_s=delay, rise_s=t_peak - delay)


# --------------------------------------------------------------- конфигурация


@dataclass
class Config:
    name: str = "clean"
    seed: int = 20261021
    fs: float = FS
    duration_s: float = 300.0
    hr_bpm: float = 72.0
    hr_variability: float = 0.035
    dc_hand: float = 15000.0
    dc_foot: float = 22000.0
    drift_hand: float = 140.0
    drift_foot: float = 120.0
    noise_hand: float = 22.0
    noise_foot: float = 24.0
    amp_hand: float = 700.0
    foot_gain: float = 0.46
    rise_hand_s: float = 0.15
    tau_s: float = 0.150
    n_attenuation: int = 0
    attenuation_depth: float = 0.40
    n_missed_hand: int = 0
    n_missed_foot: int = 0
    resp_depth: float = 0.12
    resp_period_s: float = 4.2
    drift_share: float = 0.45
    shape: Shape = Shape()


def presets() -> dict[str, Config]:
    """Пресеты покрывают роли из контракта набора."""
    base = Config()
    return {
        # чисто: без пропусков и затухания, шум стопы не выше шума руки
        "clean": replace(base, name="clean", seed=20261021, duration_s=300.0,
                         hr_bpm=72.0, noise_hand=22.0, noise_foot=22.0,
                         drift_hand=140.0, drift_foot=120.0, resp_depth=0.12),
        # дрейф и шум + шумная стопа (шум стопы в 1.36 раза выше шума руки)
        "drift_noise": replace(base, name="drift_noise", seed=20261022,
                               duration_s=420.0, hr_bpm=78.0,
                               noise_hand=28.0, noise_foot=34.0,
                               drift_hand=260.0, drift_foot=220.0,
                               drift_share=0.5, resp_depth=0.15,
                               amp_hand=800.0),
        # пропуски в обоих каналах
        "dropouts": replace(base, name="dropouts", seed=20261023,
                            duration_s=360.0, hr_bpm=68.0,
                            noise_hand=24.0, noise_foot=26.0,
                            n_missed_hand=6, n_missed_foot=4, resp_depth=0.10),
        # два эпизода затухания на стопе
        "attenuation": replace(base, name="attenuation", seed=20261024,
                               duration_s=420.0, hr_bpm=74.0,
                               noise_hand=26.0, noise_foot=30.0,
                               n_attenuation=2, attenuation_depth=0.38,
                               resp_depth=0.12),
    }


# --------------------------------------------------------------- генерация


def onsets(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    ibi_mean = 60.0 / cfg.hr_bpm
    out: list[float] = []
    t = 0.7 + float(rng.uniform(0.0, 0.2))
    stop = cfg.duration_s - max(4.0 * cfg.rise_hand_s, 0.8)
    while t < stop:
        out.append(t)
        ibi = ibi_mean * (1.0 + cfg.hr_variability * float(rng.normal()))
        t += float(np.clip(ibi, 0.45, 1.60))
    return np.array(out)


def attenuation(t: np.ndarray, cfg: Config) -> np.ndarray:
    """Кривая эпизодов затухания: 1 вне эпизодов, depth в середине эпизода."""
    a = np.ones_like(t)
    if cfg.n_attenuation <= 0:
        return a
    length = min(30.0, cfg.duration_s / 7.0)
    centres = np.linspace(0.22, 0.78, cfg.n_attenuation) * cfg.duration_s
    for c in centres:
        lo, hi = c - length / 2.0, c + length / 2.0
        m = (t >= lo) & (t <= hi)
        if not m.any():
            continue
        x = (t[m] - lo) / (hi - lo)
        a[m] = 1.0 - (1.0 - cfg.attenuation_depth) * 0.5 * (1.0 - np.cos(TWO_PI * x))
    return a


def drift(t: np.ndarray, amp: float, rng: np.random.Generator) -> np.ndarray:
    """Медленный дрейф ниже 0.5 Гц: сумма синусоид со случайными фазами."""
    out = np.zeros_like(t)
    for f in (0.017, 0.041, 0.083, 0.16, 0.31):
        out += np.sin(TWO_PI * f * t + float(rng.uniform(0.0, TWO_PI)))
    out -= out.mean()
    return amp * out / float(np.std(out))


def build(cfg: Config) -> dict:
    rng = np.random.default_rng(cfg.seed)
    fs = float(cfg.fs)
    n = int(round(cfg.duration_s * fs))
    t = np.arange(n) / fs

    # 1. ритм и амплитуды
    on = onsets(cfg, rng)
    nb = on.size
    phase = float(rng.uniform(0.0, TWO_PI))
    amp = cfg.amp_hand * (1.0 + cfg.resp_depth *
                          np.sin(TWO_PI * on / cfg.resp_period_s + phase))

    # 2. форма волны и следствия фильтра
    shape = cfg.shape
    u = unit_shape(shape, cfg.rise_hand_s, fs)
    foot = foot_shape(shape, cfg.rise_hand_s, cfg.tau_s, fs)
    g0 = cfg.foot_gain / foot.gain          # внутренний множитель ФНЧ
    env = attenuation(t, cfg)
    a_beat = np.interp(on, t, env)
    gain_beat = cfg.foot_gain * a_beat      # объявленный foot_gain по ударам

    # 3. пропуски — независимые по каналам
    interior = np.arange(3, nb - 3)
    miss_h = np.zeros(nb, dtype=bool)
    miss_f = np.zeros(nb, dtype=bool)
    if cfg.n_missed_hand:
        miss_h[rng.choice(interior, size=cfg.n_missed_hand, replace=False)] = True
    if cfg.n_missed_foot:
        free = interior[~miss_h[interior]]
        miss_f[rng.choice(free, size=cfg.n_missed_foot, replace=False)] = True

    # 4. пульсовая составляющая
    def train(keep: np.ndarray) -> np.ndarray:
        x = np.zeros(n)
        for k in np.flatnonzero(keep):
            i0 = int(round(on[k] * fs))
            m = min(u.size, n - i0)
            x[i0:i0 + m] += amp[k] * u[:m]
        return x

    hand_ac = train(~miss_h)
    # стопа фильтруется по своему поезду: пропуск руки не гасит волну стопы,
    # пропуск стопы убирает её волну до фильтрации
    foot_ac = g0 * env * lowpass_single_pole(train(~miss_f), cfg.tau_s, fs)

    # 5. дрейф, шум, квантование
    hand_raw = (cfg.dc_hand + drift(t, cfg.drift_share * cfg.drift_hand, rng)
                + hand_ac + rng.normal(0.0, cfg.noise_hand, n))
    foot_raw = (cfg.dc_foot + drift(t, cfg.drift_share * cfg.drift_foot, rng)
                + foot_ac + rng.normal(0.0, cfg.noise_foot, n))
    hand_i = np.clip(np.round(hand_raw), 0, ADC_MAX).astype(np.int64)
    foot_i = np.clip(np.round(foot_raw), 0, ADC_MAX).astype(np.int64)

    # 6. разметка
    ref = []
    for k in range(nb):
        ref.append({
            "beat_id": k,
            "onset_s": float(on[k]),
            "peak_s": float(on[k] + cfg.rise_hand_s),
            "peak_foot_s": float(on[k] + foot.peak_s),
            "amp_hand": float(amp[k]),
            "amp_foot": float(amp[k] * gain_beat[k]),
            "missed_hand": int(miss_h[k]),
            "missed_foot": int(miss_f[k]),
            "foot_gain": 0.0 if miss_f[k] else float(gain_beat[k]),
        })

    params = {
        "name": cfg.name,
        "seed": cfg.seed,
        "fs": fs,
        "duration_s": cfg.duration_s,
        "hr_bpm": cfg.hr_bpm,
        "hr_variability": cfg.hr_variability,
        "dc_hand": cfg.dc_hand,
        "dc_foot": cfg.dc_foot,
        "drift_hand": cfg.drift_hand,
        "drift_foot": cfg.drift_foot,
        "noise_hand": cfg.noise_hand,
        "noise_foot": cfg.noise_foot,
        "amp_hand": cfg.amp_hand,
        "foot_delay_s": float(foot.delay_s),
        "rise_hand_s": cfg.rise_hand_s,
        "rise_foot_s": float(foot.rise_s),
        "foot_gain": cfg.foot_gain,
        "n_attenuation": cfg.n_attenuation,
        "attenuation_depth": cfg.attenuation_depth,
        "n_missed_hand": cfg.n_missed_hand,
        "n_missed_foot": cfg.n_missed_foot,
        # --- дополнительные ключи (контракт их разрешает, но не требует)
        "tau_s": cfg.tau_s,
        "lpf_gain": foot.gain,
        "foot_peak_s": foot.peak_s,
        "resp_depth": cfg.resp_depth,
        "resp_period_s": cfg.resp_period_s,
        "drift_share": cfg.drift_share,
        "shape": asdict(shape),
        "n_beats": int(nb),
    }
    return {"t": t, "hand": hand_i, "foot": foot_i, "ref": ref, "config": params}


# --------------------------------------------------------------- запись


def write_set(dst: Path, data: dict) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "signal.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write("t,hand,foot\r\n")
        for ti, hi, fi in zip(data["t"], data["hand"], data["foot"]):
            fh.write(f"{ti:.6f},{hi},{fi}\r\n")
    cols = ["beat_id", "onset_s", "peak_s", "peak_foot_s", "amp_hand", "amp_foot",
            "missed_hand", "missed_foot", "foot_gain"]
    with open(dst / "reference.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\r\n")
        for r in data["ref"]:
            fh.write(f"{r['beat_id']},{r['onset_s']:.6f},{r['peak_s']:.6f},"
                     f"{r['peak_foot_s']:.6f},{r['amp_hand']:.3f},{r['amp_foot']:.3f},"
                     f"{r['missed_hand']},{r['missed_foot']},{r['foot_gain']:.6f}\r\n")
    payload = {"preset": data["config"]["name"], "config": data["config"]}
    (dst / "params.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Синтетические пары ФПГ (T1)")
    ap.add_argument("--out", required=True, help="каталог для наборов")
    ap.add_argument("--all", action="store_true", help="собрать все пресеты")
    ap.add_argument("--preset", default=None, help="имя пресета")
    ap.add_argument("--preset-name", default=None,
                    help="имя каталога набора (по умолчанию — имя пресета)")
    ap.add_argument("--seed", type=int, default=None, help="seed генератора")
    args = ap.parse_args()

    out = Path(args.out)
    table = presets()
    if args.all:
        for name, cfg in table.items():
            if args.seed is not None:
                cfg = replace(cfg, seed=args.seed)
            data = build(cfg)
            write_set(out / f"set_{name}", data)
            print(f"set_{name}: {data['config']['n_beats']} ударов, "
                  f"{cfg.duration_s:.0f} с")
        return
    if not args.preset:
        raise SystemExit("нужен --all или --preset NAME")
    if args.preset not in table:
        raise SystemExit(f"нет пресета {args.preset!r}; есть: {sorted(table)}")
    cfg = table[args.preset]
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    name = args.preset_name or cfg.name
    data = build(cfg)
    write_set(out / name, data)
    print(f"{name}: {data['config']['n_beats']} ударов, {cfg.duration_s:.0f} с")


if __name__ == "__main__":
    main()
