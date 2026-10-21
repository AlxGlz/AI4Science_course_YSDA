"""Численная сверка того, что видно на рисунках figures/*.png.

Проверка независимая: считает по signal.csv и reference.csv сама, тем же
набором оценок, что и приёмка (медианный уровень в окне 1 с, сглаживание
40 мс), но своим кодом. Результат — figures/facts.json и строки PASS/FAIL.

    .venv\\Scripts\\python runs\\t1-w4-r1\\check_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

HERE = Path(__file__).resolve().parent
SETS = HERE / "sets"
FIGS = HERE / "figures"
FS = 320.0


def load_set(name: str):
    d = SETS / name
    data = np.loadtxt(d / "signal.csv", delimiter=",", skiprows=1)
    with open(d / "reference.csv", encoding="utf-8", newline="") as fh:
        ref = list(csv.DictReader(fh))
    for r in ref:
        for k in ("onset_s", "peak_s", "peak_foot_s", "amp_hand", "amp_foot", "foot_gain"):
            r[k] = float(r[k])
        r["missed_hand"] = int(r["missed_hand"])
        r["missed_foot"] = int(r["missed_foot"])
    cfg = json.loads((d / "params.json").read_text(encoding="utf-8"))["config"]
    return data[:, 0], data[:, 1], data[:, 2], ref, cfg


def smooth(x, win_s: float = 0.04) -> np.ndarray:
    return uniform_filter1d(x, int(win_s * FS) | 1, mode="nearest")


def pedestal(x, win_s: float = 1.0) -> np.ndarray:
    return median_filter(x, size=int(win_s * FS) | 1, mode="nearest")


def peak_of(x_search, x_value, lo_s, hi_s, base_s):
    lo = max(0, int(round(lo_s * FS)))
    hi = min(x_search.size, int(round(hi_s * FS)))
    i = lo + int(np.argmax(x_search[lo:hi]))
    b0, b1 = max(0, int(round((base_s - 0.05) * FS))), max(1, int(round((base_s - 0.01) * FS)))
    return i / FS, float(x_value[i]) - float(np.median(x_value[b0:b1]))


class Rep:
    def __init__(self) -> None:
        self.res: list[tuple[bool, str, str]] = []

    def ck(self, ok: bool, name: str, detail: str = "") -> None:
        self.res.append((bool(ok), name, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name} — {detail}")

    @property
    def ok(self) -> bool:
        return all(r[0] for r in self.res)


def main() -> int:
    rep = Rep()
    facts: dict[str, dict] = {}

    # ---- fig1: чистый набор, окно 40-50 с -------------------------------
    t, hand, foot, ref, cfg = load_set("set_clean")
    hs, fs_ = smooth(hand), smooth(foot)
    h_sear = hs - pedestal(hand)
    t0, t1 = 40.0, 50.0
    beats = [r for r in ref if t0 <= r["onset_s"] <= t1]
    amp_h, amp_f, dt_h, dt_f = [], [], [], []
    for r in beats:
        _, a = peak_of(h_sear, hs, r["onset_s"] - 0.05, r["peak_s"] + 0.2, r["onset_s"])
        amp_h.append(a / r["amp_hand"])
        p, _ = peak_of(h_sear, hs, r["onset_s"] - 0.05, r["peak_s"] + 0.2, r["onset_s"])
        dt_h.append(1000 * (p - r["peak_s"]))
        if not r["missed_foot"]:
            _, af = peak_of(fs_ - pedestal(foot), fs_, r["peak_foot_s"] - 0.25,
                            r["peak_foot_s"] + 0.2, r["onset_s"] + cfg["foot_delay_s"])
            amp_f.append(af / r["amp_foot"])
            pf, _ = peak_of(fs_ - pedestal(foot), fs_, r["peak_foot_s"] - 0.25,
                            r["peak_foot_s"] + 0.2, r["onset_s"] + cfg["foot_delay_s"])
            dt_f.append(1000 * (pf - r["peak_foot_s"]))
    facts["fig1_clean"] = {
        "window_s": [t0, t1],
        "beats": len(beats),
        "measured_over_declared_hand": float(np.median(amp_h)),
        "measured_over_declared_foot": float(np.median(amp_f)),
        "peak_time_error_ms_hand": float(np.median(dt_h)),
        "peak_time_error_ms_foot": float(np.median(dt_f)),
    }
    rep.ck(len(beats) >= 10, "fig1: в окне 10 с достаточно ударов",
           f"{len(beats)} ударов")
    rep.ck(abs(np.median(amp_h) - 1) < 0.12, "fig1: высота волны руки видна как объявленная",
           f"измерено/объявлено {np.median(amp_h):.3f}")
    rep.ck(abs(np.median(amp_f) - 1) < 0.12, "fig1: высота волны стопы видна как объявленная",
           f"измерено/объявлено {np.median(amp_f):.3f}")
    rep.ck(abs(np.median(dt_h)) < 15 and abs(np.median(dt_f)) < 15,
           "fig1: пики нарисованы там, где они в сигнале",
           f"{np.median(dt_h):+.1f} мс рука, {np.median(dt_f):+.1f} мс стопа")

    # ---- fig2: эпизод затухания ---------------------------------------
    t, hand, foot, ref, cfg = load_set("set_attenuation")
    hs, fs_ = smooth(hand), smooth(foot)
    c = 0.22 * cfg["duration_s"]
    inside = [r for r in ref if abs(r["onset_s"] - c) < 3.0]
    outside = [r for r in ref if 8.0 < abs(r["onset_s"] - c) < 20.0]

    def amp_at(r, x_s, x_v):
        _, a = peak_of(x_s, x_v, r["peak_foot_s"] - 0.25, r["peak_foot_s"] + 0.2,
                       r["onset_s"] + cfg["foot_delay_s"])
        return a

    base = cfg["foot_gain"]

    def rel_to_base(r):
        """Высота стопы относительно «без затухания»: amp_hand * base."""
        return amp_at(r, fs_ - pedestal(foot), fs_) / (r["amp_hand"] * base)

    ratio_in = float(np.median([rel_to_base(r) for r in inside]))
    ratio_out = float(np.median([rel_to_base(r) for r in outside]))
    gains = np.array([r["foot_gain"] for r in ref])
    facts["fig2_attenuation"] = {
        "episode_centres_s": [float(x) for x in np.linspace(0.22, 0.78,
                                                           cfg["n_attenuation"]) * cfg["duration_s"]],
        "declared_min_gain_over_base": float(gains.min() / cfg["foot_gain"]),
        "declared_depth": float(cfg["attenuation_depth"]),
        "measured_foot_amp_over_declared_in_core": ratio_in,
        "measured_foot_amp_over_declared_outside": ratio_out,
        "measured_drop_ratio": ratio_in / ratio_out,
    }
    rep.ck(abs(gains.min() / cfg["foot_gain"] - cfg["attenuation_depth"]) < 0.05,
           "fig2: минимум foot_gain равен объявленной глубине",
           f"{gains.min() / cfg['foot_gain']:.3f} против {cfg['attenuation_depth']:.2f}")
    rep.ck(abs(ratio_in / ratio_out - cfg["attenuation_depth"]) < 0.15,
           "fig2: провал амплитуды на рисунке равен глубине затухания",
           f"измеренное отношение {ratio_in / ratio_out:.3f}")

    # ---- fig3: пропуски ------------------------------------------------
    t, hand, foot, ref, cfg = load_set("set_dropouts")
    hs, fs_ = smooth(hand), smooth(foot)
    h_sear, f_sear = hs - pedestal(hand), fs_ - pedestal(foot)
    nb_h = [r["amp_hand"] for r in ref if not r["missed_hand"]]
    nb_f = [r["amp_foot"] for r in ref if not r["missed_foot"]]
    bumps_h, bumps_f = [], []
    for r in ref:
        if r["missed_hand"]:
            _, a = peak_of(h_sear, hs, r["onset_s"] + 0.02, r["onset_s"] + 0.45, r["onset_s"])
            bumps_h.append(a / float(np.median(nb_h)))
        if r["missed_foot"]:
            lo = r["onset_s"] + cfg["foot_delay_s"] + 0.02
            _, a = peak_of(f_sear, fs_, lo, lo + 0.45, lo)
            bumps_f.append(a / float(np.median(nb_f)))
    lost = [r for r in ref if r["missed_hand"] and not r["missed_foot"]]
    keep_f = []
    for r in lost:
        _, a = peak_of(f_sear, fs_, r["peak_foot_s"] - 0.25, r["peak_foot_s"] + 0.2,
                       r["onset_s"] + cfg["foot_delay_s"])
        keep_f.append(a / r["amp_foot"])
    facts["fig3_missed"] = {
        "n_missed_hand": len(bumps_h),
        "n_missed_foot": len(bumps_f),
        "max_hand_bump_share": float(max(bumps_h)),
        "max_foot_bump_share": float(max(bumps_f)),
        "foot_kept_when_hand_missed": float(np.median(keep_f)),
    }
    rep.ck(max(bumps_h) < 0.40, "fig3: на месте пропуска руки волны нет",
           f"макс. остаток {100 * max(bumps_h):.0f} % соседней волны")
    rep.ck(max(bumps_f) < 0.55, "fig3: на месте пропуска стопы волны нет",
           f"макс. остаток {100 * max(bumps_f):.0f} %")
    rep.ck(abs(np.median(keep_f) - 1) < 0.25, "fig3: пропуск руки не гасит волну стопы",
           f"высота стопы {np.median(keep_f):.2f} от объявленной")

    # ---- fig4: форма волны, задержка и время подъёма --------------------
    t, hand, foot, ref, cfg = load_set("set_clean")
    hs, fs_ = smooth(hand), smooth(foot)
    delays, rises_h, rises_f = [], [], []
    for r in ref:
        if r["missed_hand"] or r["missed_foot"]:
            continue
        ph, _ = peak_of(hs - pedestal(hand), hs, r["onset_s"] - 0.05,
                        r["peak_s"] + 0.2, r["onset_s"])
        pf, _ = peak_of(fs_ - pedestal(foot), fs_, r["peak_foot_s"] - 0.25,
                        r["peak_foot_s"] + 0.2, r["onset_s"] + cfg["foot_delay_s"])
        delays.append(1000 * (pf - ph))
        rises_h.append(1000 * (ph - r["onset_s"]))
        rises_f.append(1000 * (pf - r["onset_s"] - cfg["foot_delay_s"]))
    expect = 1000 * (cfg["foot_delay_s"] + cfg["rise_foot_s"] - cfg["rise_hand_s"])
    facts["fig4_shape"] = {
        "declared_rise_hand_ms": 1000 * cfg["rise_hand_s"],
        "declared_rise_foot_ms": 1000 * cfg["rise_foot_s"],
        "declared_foot_delay_ms": 1000 * cfg["foot_delay_s"],
        "declared_peak_delay_ms": expect,
        "measured_peak_delay_ms": float(np.median(delays)),
        "measured_rise_hand_ms": float(np.median(rises_h)),
        "measured_rise_foot_ms_from_delay": float(np.median(rises_f)),
        "tau_ms": 1000 * cfg["tau_s"],
    }
    rep.ck(abs(np.median(delays) - expect) < 25, "fig4: задержка пика стопы на рисунке",
           f"измерено {np.median(delays):.0f} мс, объявлено {expect:.0f} мс")
    rep.ck(abs(np.median(rises_f) - 1000 * cfg["rise_foot_s"]) < 15,
           "fig4: время подъёма стопы от точки 15 %",
           f"измерено {np.median(rises_f):.0f} мс, объявлено "
           f"{1000 * cfg['rise_foot_s']:.0f} мс")

    FIGS.mkdir(parents=True, exist_ok=True)
    (FIGS / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print("-" * 72)
    print(f"{sum(r[0] for r in rep.res)}/{len(rep.res)} сверок рисунков пройдено; "
          f"факты — figures/facts.json")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
