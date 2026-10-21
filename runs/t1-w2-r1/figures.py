"""Рисунки прогона t1-w2-r1: пары «рука — стопа» с эталонной разметкой.

Читает ровно те файлы, что лежат в sets/<набор>/ (signal.csv, reference.csv,
params.json), и ничего не берёт из памяти генератора. Рядом с рисунками
пишется figures/facts.json — какие окна и удары изображены и какие числа
подписаны; check_figures.py независимо пересчитывает их по CSV.

Запуск:  .venv\\Scripts\\python runs\\t1-w2-r1\\figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
SETS = HERE / "sets"
FIGS = HERE / "figures"
FS = 320.0


def load_set(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    d = SETS / name
    t, hand, foot = np.loadtxt(d / "signal.csv", delimiter=",", skiprows=1).T
    with open(d / "reference.csv", encoding="utf-8", newline="") as fh:
        ref = [{k: (float(v) if k not in ("beat_id", "missed_hand", "missed_foot") else int(v))
                for k, v in r.items()} for r in csv.DictReader(fh)]
    cfg = json.loads((d / "params.json").read_text(encoding="utf-8"))["config"]
    return t, hand, foot, ref, cfg


def detrend(x: np.ndarray) -> np.ndarray:
    """Вычитание скользящей медианы 1 с — то же, что делает проверка."""
    return x - median_filter(x, size=int(FS) | 1, mode="nearest")


def window_beats(ref: list[dict], t0: float, t1: float) -> list[dict]:
    return [b for b in ref if t0 <= b["onset_s"] <= t1]


def draw_pair(ax_h, ax_f, t, hand, foot, beats, ta, tb, title: str) -> dict:
    m = (t >= ta) & (t <= tb)
    ax_h.plot(t[m], hand[m], lw=0.7, color="#1f4e79")
    ax_f.plot(t[m], foot[m], lw=0.7, color="#7b3f00")
    for b in beats:
        for ax, sig, missed, peak_key in ((ax_h, hand, bool(b["missed_hand"]), "peak_s"),
                                          (ax_f, foot, bool(b["missed_foot"]), "peak_foot_s")):
            ax.axvline(b["onset_s"], color="#999999", lw=0.8, ls=":", zorder=1)
            ypk = float(np.interp(b[peak_key], t, sig))
            if missed:
                ax.axvspan(b["onset_s"], b["onset_s"] + 0.33, color="#d62728",
                           alpha=0.10, zorder=0)
                ax.plot(b[peak_key], ypk, marker="x", ms=9, color="#d62728",
                        mew=1.6, zorder=3)
            else:
                ax.plot(b[peak_key], ypk, marker="^", ms=5, color="#2ca02c",
                        mfc="none", zorder=3)
    for ax, lbl in ((ax_h, "палец руки"), (ax_f, "палец стопы")):
        ax.set_ylabel(f"{lbl}, отсчёты АЦП")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(ta, tb)
    ax_h.set_title(title, fontsize=10)
    return {"t0": float(ta), "t1": float(tb),
            "beats": [{"beat_id": b["beat_id"], "onset_s": b["onset_s"],
                       "peak_s": b["peak_s"], "peak_foot_s": b["peak_foot_s"],
                       "amp_hand": b["amp_hand"], "amp_foot": b["amp_foot"],
                       "foot_gain": b["foot_gain"],
                       "missed_hand": int(b["missed_hand"]),
                       "missed_foot": int(b["missed_foot"])} for b in beats]}


def fig_clean() -> dict:
    t, hand, foot, ref, cfg = load_set("set_clean")
    h, f = detrend(hand), detrend(foot)
    ta, tb = 12.0, 24.0
    beats = window_beats(ref, ta, tb)
    fig, (axh, axf) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    info = draw_pair(axh, axf, t, h, f, beats, ta, tb,
                     "Набор set_clean, ЧСС %.0f: волны руки и стопы; ▲ — пик, "
                     "пунктир — начало волны" % cfg["hr_bpm"])
    axf.set_xlabel("время, с (от начала записи)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_clean.png", dpi=130)
    plt.close(fig)
    return {"figure": "fig1_clean.png", "set": "set_clean", **info}


def fig_attenuation() -> dict:
    t, hand, foot, ref, cfg = load_set("set_baseline")
    h, f = detrend(hand), detrend(foot)
    # окно вокруг минимума объявленного коэффициента стопы
    cand = [b for b in ref if not b["missed_foot"]]
    bmin = min(cand, key=lambda b: b["foot_gain"])
    ta, tb = bmin["onset_s"] - 14.0, bmin["onset_s"] + 16.0
    beats = window_beats(ref, ta, tb)
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.85], hspace=0.35)
    axh, axf, axg = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    info = draw_pair(axh, axf, t, h, f, beats, ta, tb,
                     "Набор set_baseline: эпизод затухания стопы; %d ударов в окне"
                     % len(beats))
    axf.set_xlabel("время, с")
    axg.plot([b["onset_s"] for b in ref], [b["foot_gain"] for b in ref],
             lw=0.9, color="#444444")
    axg.axhline(0.6, color="#d62728", lw=0.8, ls="--")
    axg.axvspan(ta, tb, color="#1f77b4", alpha=0.12)
    axg.set_ylabel("коэффициент стопы")
    axg.set_xlabel("время, с (вся запись); красная линия — уровень 0,6")
    axg.grid(alpha=0.25, lw=0.5)
    fig.savefig(FIGS / "fig2_attenuation.png", dpi=130)
    plt.close(fig)
    return {"figure": "fig2_attenuation.png", "set": "set_baseline", **info,
            "min_gain": float(bmin["foot_gain"]),
            "gain_series": [[b["onset_s"], b["foot_gain"]] for b in ref]}


def fig_missed() -> dict:
    t, hand, foot, ref, cfg = load_set("set_hard")
    h, f = detrend(hand), detrend(foot)
    def best_centre(flag: str) -> float:
        def score(c: float) -> tuple[int, float]:
            inb = [b for b in ref if c - 2.2 <= b["onset_s"] <= c + 2.2]
            return (sum(1 for b in inb if b[flag]), -c)
        return max((b["onset_s"] for b in ref if b[flag]), key=score)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.75)
    wins: list[dict] = []
    for flag, ax_h, ax_f, lbl in (("missed_foot", axes[0], axes[1], "палец стопы"),
                                  ("missed_hand", axes[2], axes[3], "палец руки")):
        c = best_centre(flag)
        ta, tb = c - 2.2, c + 2.2
        beats = window_beats(ref, ta, tb)
        wins.append(draw_pair(
            ax_h, ax_f, t, h, f, beats, ta, tb,
            f"Набор set_hard: пропуск в канале «{lbl}» (красная заливка и ×), "
            f"во втором канале волна на месте"))
        ax_h.set_xlabel("время, с")
    axes[-1].set_xlabel("время, с")
    fig.savefig(FIGS / "fig3_missed.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    seen: dict[int, dict] = {}
    for w in wins:
        for b in w["beats"]:
            seen.setdefault(b["beat_id"], b)
    return {"figure": "fig3_missed.png", "set": "set_hard",
            "t0": min(w["t0"] for w in wins), "t1": max(w["t1"] for w in wins),
            "windows": wins, "beats": [seen[k] for k in sorted(seen)]}


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    facts = {"figures": [fig_clean(), fig_attenuation(), fig_missed()]}
    text = json.dumps(facts, ensure_ascii=False, indent=2)
    (FIGS / "facts.json").write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    for f in facts["figures"]:
        print(f"{f['figure']}: {f['set']}, окно {f['t0']:.1f}-{f['t1']:.1f} с, "
              f"{len(f['beats'])} ударов")


if __name__ == "__main__":
    main()
