"""Общий (правильный) модулятор для selftest-проверки check_1.py.

Генерирует пару ФПГ-сигналов «рука + стопа» и эталонную разметку по
спецификации задачи 1. Используется run.py в каждой папке прогона selftest.
"""

import json
from pathlib import Path

import numpy as np


def wave(x: np.ndarray) -> np.ndarray:
    """Модельная форма пульсовой волны: пик в x=0.3, на краях ~0 (x в [0,1]).

    Узкий гаусс: на x=0 значение ~0.002, чтобы база волны в onset была ~0
    и измеренная по сигналу амплитуда (пик - база) совпадала с amp_hand.
    """
    return np.exp(-((x - 0.3) / 0.12) ** 2)


def generate(out_dir, params, tau_override=None, drop_override=None,
             seed_override=None, amp_foot_scale=1.0, drop_waves=0) -> None:
    """Строит signal.csv / truth.csv / params.json в out_dir.

    Override-параметры позволяют намеренно «сломать» модулятор для selftest:
      tau_override      — игнорировать заданный tau (задержка стопы);
      drop_override     — игнорировать заданный drop (доля пропусков);
      seed_override     — игнорировать заданный seed (детерминизм);
      amp_foot_scale    — масштаб amp_foot (нарушить amp_foot = k*amp_hand);
      drop_waves        — сколько волн стереть из signal.csv (нарушить число волн).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = params["duration"]
    fs = params["fs"]
    hr = params["hr"]
    k = params["k"]
    tau = params["tau"] if tau_override is None else tau_override
    sigma = params["sigma"]
    drop = params["drop"] if drop_override is None else drop_override
    seed = params["seed"] if seed_override is None else seed_override

    rng = np.random.default_rng(seed)

    width = 0.5          # длительность волны, с
    peak_frac = 0.3      # положение пика внутри волны
    peak_delay = peak_frac * width
    period = 60.0 / hr

    # последовательность ударов со средней частотой hr и вариабельностью интервалов
    n_est = int(round(duration * hr / 60.0))
    intervals = period * (1.0 + 0.05 * rng.standard_normal(n_est))
    onsets = np.cumsum(intervals) - intervals[0]
    onsets = onsets[onsets < duration - width]
    n = len(onsets)

    t = np.arange(0.0, duration, 1.0 / fs)
    hand = np.zeros_like(t)
    foot = np.zeros_like(t)

    present = rng.random(n) >= drop

    beats = []
    for i, o in enumerate(onsets):
        p = bool(present[i])
        amp = max(1.0 + 0.1 * rng.standard_normal(), 0.3)
        peak_t = o + peak_delay

        idx = (t >= o) & (t < o + width)
        hand[idx] += amp * wave((t[idx] - o) / width)

        if p:
            idx2 = (t >= o + tau) & (t < o + tau + width)
            foot[idx2] += k * amp * wave((t[idx2] - o - tau) / width)

        beats.append({
            "beat": i + 1,
            "t_hand_onset": o,
            "t_hand_peak": peak_t,
            "t_foot_onset": o + tau,
            "t_foot_peak": peak_t + tau,
            "amp_hand": amp,
            "amp_foot": k * amp * amp_foot_scale if p else 0.0,
            "present": 1 if p else 0,
        })

    hand += sigma * rng.standard_normal(len(t))
    foot += sigma * rng.standard_normal(len(t))

    # намеренно стереть drop_waves волн из signal.csv (для selftest числа волн)
    if drop_waves > 0:
        for o in onsets[:drop_waves]:
            idx = (t >= o) & (t < o + width)
            hand[idx] = 0.0

    with open(out_dir / "signal.csv", "w") as f:
        f.write("t,hand,foot\n")
        for tv, hv, fv in zip(t, hand, foot):
            f.write(f"{tv:.6f},{hv:.6f},{fv:.6f}\n")

    with open(out_dir / "truth.csv", "w") as f:
        f.write("beat,t_hand_onset,t_hand_peak,t_foot_onset,t_foot_peak,amp_hand,amp_foot,present\n")
        for b in beats:
            f.write(
                f"{b['beat']},{b['t_hand_onset']:.6f},{b['t_hand_peak']:.6f},"
                f"{b['t_foot_onset']:.6f},{b['t_foot_peak']:.6f},"
                f"{b['amp_hand']:.6f},{b['amp_foot']:.6f},{b['present']}\n"
            )

    with open(out_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)