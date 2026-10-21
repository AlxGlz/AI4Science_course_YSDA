"""Численная сверка рисунков: то, что видно на картинке, сверяется с CSV.

Проверяются не пиксели, а величины, которые изображены: окна, состав ударов,
положение пиков относительно объявленной разметки, высоты волн, провал
амплитуды стопы в эпизоде затухания, пустота пропущенных ударов.
Все числа пересчитываются независимо от figures.py — по signal.csv,
reference.csv и params.json.

Запуск:  .venv\\Scripts\\python runs\\t1-w2-r1\\check_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

HERE = Path(__file__).resolve().parent
SETS, FIGS = HERE / "sets", HERE / "figures"
FS = 320.0
SMOOTH = int(0.04 * FS) | 1
RES: list[tuple[bool, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    RES.append((bool(ok), name))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


def load(name: str):
    d = SETS / name
    t, hand, foot = np.loadtxt(d / "signal.csv", delimiter=",", skiprows=1).T
    with open(d / "reference.csv", encoding="utf-8", newline="") as fh:
        ref = [{k: (float(v) if k not in ("beat_id", "missed_hand", "missed_foot") else int(v))
                for k, v in r.items()} for r in csv.DictReader(fh)]
    cfg = json.loads((d / "params.json").read_text(encoding="utf-8"))["config"]
    return t, hand, foot, ref, cfg


def measure(x: np.ndarray, xs: np.ndarray, lo_s: float, hi_s: float,
            base_s: float) -> tuple[float, float]:
    lo, hi = max(0, int(round(lo_s * FS))), min(x.size, int(round(hi_s * FS)))
    i = lo + int(np.argmax(xs[lo:hi]))
    base = float(xs[max(0, int(round(base_s * FS)))])
    a, b = max(0, i - 5), min(x.size, i + 6)
    if b - a >= 5:
        tt = (np.arange(a, b) - i) / FS
        coef = np.polyfit(tt, x[a:b], 2)
        if coef[0] < 0:
            tv = float(np.clip(-coef[1] / (2 * coef[0]), tt[0], tt[-1]))
            return (i + tv * FS) / FS, float(np.polyval(coef, tv)) - base
    return i / FS, float(x[i]) - base


def main() -> int:
    facts = json.loads((FIGS / "facts.json").read_text(encoding="utf-8"))["figures"]
    check(len(facts) >= 2, "рисунков не меньше двух", f"{len(facts)} шт.")

    for fig in facts:
        tag = fig["figure"]
        p = FIGS / tag
        head = p.read_bytes()[:8]
        check(p.exists() and p.stat().st_size > 30_000 and head == b"\x89PNG\r\n\x1a\n",
              f"[{tag}] файл PNG на месте", f"{p.stat().st_size / 1024:.0f} КБ")
        img = plt.imread(p)
        check(img.shape[0] > 500 and img.shape[1] > 800 and float(np.std(img)) > 0.02,
              f"[{tag}] рисунок не пустой",
              f"{img.shape[1]}x{img.shape[0]} px, разброс яркости {float(np.std(img)):.3f}")

        t, hand, foot, ref, cfg = load(fig["set"])
        hb, fb = hand.astype(float), foot.astype(float)
        hb -= median_filter(hb, size=int(FS) | 1, mode="nearest")
        fb -= median_filter(fb, size=int(FS) | 1, mode="nearest")
        hs = uniform_filter1d(hb, SMOOTH, mode="nearest")
        fs_ = uniform_filter1d(fb, SMOOTH, mode="nearest")

        in_win = [b for b in fig["beats"]]
        if "windows" in fig:
            for w in fig["windows"]:
                inw = [b for b in ref if w["t0"] <= b["onset_s"] <= w["t1"]]
                check([b["beat_id"] for b in inw] == [b["beat_id"] for b in w["beats"]],
                      f"[{tag}] окно {w['t0']:.1f}-{w['t1']:.1f} с: состав ударов "
                      f"совпадает с reference.csv", f"{len(inw)} ударов")
        else:
            in_win = [b for b in ref if fig["t0"] <= b["onset_s"] <= fig["t1"]]
            check([b["beat_id"] for b in in_win] == [b["beat_id"] for b in fig["beats"]],
                  f"[{tag}] состав ударов в окне совпадает с reference.csv",
                  f"{len(in_win)} ударов, окно {fig['t0']:.1f}-{fig['t1']:.1f} с")
        check(len(in_win) >= 8 or ("windows" in fig and
                                   min(len(w["beats"]) for w in fig["windows"]) >= 6),
              f"[{tag}] в окне достаточно ударов", f"{len(in_win)} в изображении")

        # окно не обрезает волны: до и после окна есть запас сигнала
        check(fig["t0"] > 0.5 and fig["t1"] < t[-1] - 0.5,
              f"[{tag}] окно внутри записи", f"{t[-1]:.0f} с всего")

        dt_h, dt_f, sc_h, sc_f, bump_h, bump_f = [], [], [], [], [], []
        for b in in_win:
            if b["missed_hand"]:
                _, v = measure(hb, hs, b["onset_s"] + 0.02, b["onset_s"] + 0.45, b["onset_s"])
                near = [x["amp_hand"] for x in ref if not x["missed_hand"]
                        and abs(x["onset_s"] - b["onset_s"]) < 5.0]
                bump_h.append(v / float(np.median(near)))
            else:
                pk, v = measure(hb, hs, b["onset_s"] - 0.05, b["peak_s"] + 0.20, b["onset_s"])
                dt_h.append(abs(pk - b["peak_s"]))
                sc_h.append(v / b["amp_hand"])
            if b["missed_foot"]:
                lo = b["onset_s"] + cfg["foot_delay_s"] + 0.02
                _, v = measure(fb, fs_, lo, lo + 0.45, lo)
                near = [x["amp_foot"] for x in ref if not x["missed_foot"]
                        and abs(x["onset_s"] - b["onset_s"]) < 5.0]
                bump_f.append(v / float(np.median(near)))
            else:
                pk, v = measure(fb, fs_, b["peak_foot_s"] - 0.25, b["peak_foot_s"] + 0.20,
                                b["onset_s"] + cfg["foot_delay_s"])
                dt_f.append(abs(pk - b["peak_foot_s"]))
                sc_f.append(v / b["amp_foot"])

        if dt_h:
            check(float(np.median(dt_h)) < 0.030, f"[{tag}] пики руки на максимумах волн",
                  f"медиана |dt| {1000 * float(np.median(dt_h)):.1f} мс, "
                  f"{len(dt_h)} ударов")
            check(0.90 < float(np.median(sc_h)) < 1.20,
                  f"[{tag}] высота волн руки соответствует разметке",
                  f"медиана изм/объявл {float(np.median(sc_h)):.3f}")
        if dt_f:
            check(float(np.median(dt_f)) < 0.050, f"[{tag}] пики стопы на максимумах волн",
                  f"медиана |dt| {1000 * float(np.median(dt_f)):.1f} мс, "
                  f"{len(dt_f)} ударов")
            check(0.90 < float(np.median(sc_f)) < 1.20,
                  f"[{tag}] высота волн стопы соответствует разметке",
                  f"медиана изм/объявл {float(np.median(sc_f)):.3f}")
        if bump_h:
            check(max(bump_h) < 0.40, f"[{tag}] пропущенные удары руки пусты",
                  f"{len(bump_h)} шт., максимум {100 * max(bump_h):.0f}% соседней волны")
        if bump_f:
            check(max(bump_f) < 0.55, f"[{tag}] пропущенные удары стопы пусты",
                  f"{len(bump_f)} шт., максимум {100 * max(bump_f):.0f}% соседней волны")

        # соотношение амплитуд в окне
        vis = [b for b in in_win if not (b["missed_hand"] or b["missed_foot"])]
        if vis:
            rel = [b["amp_foot"] / (b["amp_hand"] * b["foot_gain"]) for b in vis]
            check(float(np.max(np.abs(np.array(rel) - 1.0))) < 0.01,
                  f"[{tag}] amp_foot = amp_hand * foot_gain",
                  f"отклонение не больше {100 * float(np.max(np.abs(np.array(rel) - 1))):.2f}%")

    # специфика второго и третьего рисунков
    fig2 = next(f for f in facts if f["figure"] == "fig2_attenuation.png")
    t, hand, foot, ref, cfg = load(fig2["set"])
    amps = np.array([b["amp_foot"] for b in ref if not b["missed_foot"]])
    win_amps = np.array([b["amp_foot"] for b in fig2["beats"] if not b["missed_foot"]])
    check(fig2["min_gain"] < 0.6, "[fig2] минимум коэффициента стопы ниже 0,6",
          f"{fig2['min_gain']:.3f}")
    check(float(win_amps.min()) < 0.6 * float(np.median(amps)),
          "[fig2] в окне виден провал амплитуды стопы",
          f"минимум в окне {win_amps.min():.0f} отсчётов против медианы записи "
          f"{np.median(amps):.0f}")
    check(len(fig2.get("gain_series", [])) == len(ref),
          "[fig2] кривая коэффициента построена по всем ударам",
          f"{len(fig2.get('gain_series', []))} точек")

    fig3 = next(f for f in facts if f["figure"] == "fig3_missed.png")
    miss = [b for b in fig3["beats"] if b["missed_hand"] or b["missed_foot"]]
    check(len(miss) >= 2, "[fig3] в окне не меньше двух пропущенных ударов",
          f"{len(miss)} шт. (рука {sum(b['missed_hand'] for b in miss)}, "
          f"стопа {sum(b['missed_foot'] for b in miss)})")
    check(any(not b["missed_hand"] for b in fig3["beats"]),
          "[fig3] в том же окне есть удары второго канала с волной")

    failed = [r for r in RES if not r[0]]
    print("-" * 72)
    print(f"{len(RES) - len(failed)}/{len(RES)} проверок рисунков пройдено")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
