"""Рисунки прогона t1-w4-r1: пары «рука — стопа» с эталонной разметкой.

Строит четыре рисунка рядом с отчётом:

    figures/fig1_clean.png       чистый набор, 10 с: начала, пики, амплитуды
    figures/fig2_attenuation.png эпизод затухания и curve foot_gain по всей записи
    figures/fig3_missed.png      пропуски в обоих каналах
    figures/fig4_shape.png       одна волна: рука против стопы, точки 15 % и пики

Скрипт только рисует: числа, которые видно на рисунках, независимо
пересчитывает check_figures.py.

    .venv\\Scripts\\python runs\\t1-w4-r1\\figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        r["beat_id"] = int(r["beat_id"])
    cfg = json.loads((d / "params.json").read_text(encoding="utf-8"))["config"]
    return data[:, 0], data[:, 1], data[:, 2], ref, cfg


def ac_part(x: np.ndarray, win_s: float = 1.0) -> np.ndarray:
    """Пульсовая составляющая: сигнал минус медиана в окне 1 с."""
    return x - median_filter(x, size=int(win_s * FS) | 1, mode="nearest")


def mark_beats(ax, ref, t0, t1, show_foot: bool = True, missed_color="#999999"):
    for r in ref:
        if not (t0 <= r["onset_s"] <= t1):
            continue
        if r["missed_hand"]:
            ax.axvspan(r["onset_s"], r["onset_s"] + 0.35, color=missed_color,
                       alpha=0.25, lw=0)
            ax.plot([r["onset_s"]], [ax.get_ylim()[1]], marker="v", color="#444444",
                    ms=6, clip_on=False)
            continue
        ax.axvline(r["onset_s"], color="#2ca02c", ls=":", lw=1.0)
        ax.plot([r["peak_s"]], [0.0], marker="o", ms=0.0)
    for r in ref:
        if not (t0 <= r["onset_s"] <= t1) or r["missed_hand"]:
            continue
        ax.axvline(r["peak_s"], color="#d62728", ls="--", lw=1.0)
        if show_foot and not r["missed_foot"]:
            ax.axvline(r["peak_foot_s"], color="#1f77b4", ls="-.", lw=1.0)


def fig_clean() -> Path:
    t, hand, foot, ref, cfg = load_set("set_clean")
    h, f = ac_part(hand), ac_part(foot)
    t0, t1 = 40.0, 50.0
    m = (t >= t0) & (t <= t1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
    for ax, x, name, color in ((axes[0], h, "hand (finger)", "#d62728"),
                               (axes[1], f, "foot (toe)", "#1f77b4")):
        ax.plot(t[m], x[m], color=color, lw=0.9)
        ax.set_ylabel(f"{name}, ADC", fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_xlim(t0, t1)
    mark_beats(axes[0], ref, t0, t1, show_foot=True)
    mark_beats(axes[1], ref, t0, t1, show_foot=True)

    # амплитуда одного удара: от уровня разметки до пика
    beat = next(r for r in ref if t0 + 3 < r["onset_s"] < t0 + 4)
    i_on = int(round(beat["onset_s"] * FS))
    i_pk = int(round(beat["peak_s"] * FS))
    base = float(np.median(hand[i_on - 16:i_on - 3]))
    axes[0].annotate("", xy=(beat["peak_s"], hand[i_pk] - base), xytext=(beat["peak_s"], 0),
                     arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    axes[0].text(beat["peak_s"] + 0.05, 0.45 * (hand[i_pk] - base),
                 f"amp_hand = {beat['amp_hand']:.0f}", fontsize=9)
    axes[1].annotate("", xy=(beat["peak_foot_s"], f[int(round(beat["peak_foot_s"] * FS))]),
                     xytext=(beat["peak_foot_s"], 0),
                     arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    axes[1].text(beat["peak_foot_s"] + 0.05, 0.45 * f[int(round(beat["peak_foot_s"] * FS))],
                 f"amp_foot = {beat['amp_foot']:.0f}\nfoot_gain = {beat['foot_gain']:.2f}",
                 fontsize=9)

    axes[1].set_xlabel("time, s", fontsize=9)
    axes[0].set_title("set_clean: onset (green dotted), hand peak (red dashed), "
                      "foot peak (blue dash-dot)", fontsize=10)
    fig.tight_layout()
    out = FIGS / "fig1_clean.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_attenuation() -> Path:
    t, hand, foot, ref, cfg = load_set("set_attenuation")
    h, f = ac_part(hand), ac_part(foot)
    gains = np.array([r["foot_gain"] for r in ref])
    onsets = np.array([r["onset_s"] for r in ref])
    base = cfg["foot_gain"]
    c = 0.22 * cfg["duration_s"]          # центр первого эпизода
    t0, t1 = c - 14.0, c + 14.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6),
                             gridspec_kw={"height_ratios": [2.1, 1.0]})
    axes[0].plot(t, h, color="#d62728", lw=0.5, label="hand")
    axes[0].plot(t, f, color="#1f77b4", lw=0.5, label="foot")
    axes[0].set_xlim(t0, t1)
    axes[0].set_ylabel("AC, ADC", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("set_attenuation: episode of damping (foot amplitude falls to "
                      f"{cfg['attenuation_depth']:.2f} of base gain)", fontsize=10)
    axes[0].grid(alpha=0.25)

    axes[1].plot(onsets, gains / base, color="#1f77b4", lw=0.8)
    axes[1].axhline(1.0, color="grey", lw=0.8)
    axes[1].axhline(cfg["attenuation_depth"], color="black", ls="--", lw=0.8)
    axes[1].set_xlim(0, cfg["duration_s"])
    axes[1].set_ylim(0, 1.15)
    axes[1].set_xlabel("time, s", fontsize=9)
    axes[1].set_ylabel("foot_gain / base", fontsize=9)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    out = FIGS / "fig2_attenuation.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_missed() -> Path:
    t, hand, foot, ref, cfg = load_set("set_dropouts")
    h, f = ac_part(hand), ac_part(foot)
    mh = [r for r in ref if r["missed_hand"]]
    mf = [r for r in ref if r["missed_foot"]]
    anchor = mh[len(mh) // 2]
    t0, t1 = anchor["onset_s"] - 5.0, anchor["onset_s"] + 5.0
    m = (t >= t0) & (t <= t1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
    for ax, x, name, color in ((axes[0], h, "hand (finger)", "#d62728"),
                               (axes[1], f, "foot (toe)", "#1f77b4")):
        ax.plot(t[m], x[m], color=color, lw=0.9)
        ax.set_ylabel(f"{name}, ADC", fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_xlim(t0, t1)
    for ax in axes:
        for r in ref:
            if t0 <= r["onset_s"] <= t1:
                ax.axvline(r["onset_s"], color="#2ca02c", ls=":", lw=1.0)
                if not r["missed_hand"]:
                    ax.axvline(r["peak_s"], color="#d62728", ls="--", lw=0.9)
                if not r["missed_foot"]:
                    ax.axvline(r["peak_foot_s"], color="#1f77b4", ls="-.", lw=0.9)
                if r["missed_hand"]:
                    ax.axvspan(r["onset_s"], r["onset_s"] + 0.4, color="#bbbbbb",
                               alpha=0.35, lw=0)
                if r["missed_foot"]:
                    ax.axvspan(r["peak_foot_s"] - 0.2, r["peak_foot_s"] + 0.2,
                               color="#ffd27f", alpha=0.35, lw=0)
    axes[0].text(0.01, 0.92, "grey band — missed hand wave", transform=axes[0].transAxes,
                 fontsize=9)
    axes[1].text(0.01, 0.92, "orange band — missed foot wave", transform=axes[1].transAxes,
                 fontsize=9)
    axes[1].set_xlabel("time, s", fontsize=9)
    axes[0].set_title("set_dropouts: independent misses in the two channels", fontsize=10)
    fig.tight_layout()
    out = FIGS / "fig3_missed.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_shape() -> Path:
    t, hand, foot, ref, cfg = load_set("set_clean")
    h, f = ac_part(hand), ac_part(foot)
    beat = next(r for r in ref if 100.0 < r["onset_s"] < 101.0)
    t0, t1 = beat["onset_s"] - 0.1, beat["onset_s"] + 0.6
    m = (t >= t0) & (t <= t1)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(t[m] - beat["onset_s"], h[m] / beat["amp_hand"], color="#d62728", lw=1.3,
            label="hand / amp_hand")
    ax.plot(t[m] - beat["onset_s"], f[m] / beat["amp_hand"], color="#1f77b4", lw=1.3,
            label="foot / amp_hand")
    ax.axhline(0.15, color="grey", ls=":", lw=0.9)
    ax.axvline(0.0, color="#2ca02c", ls=":", lw=1.0)
    ax.axvline(cfg["rise_hand_s"], color="#d62728", ls="--", lw=1.0)
    ax.axvline(cfg["rise_hand_s"] + cfg["foot_delay_s"] + cfg["rise_foot_s"],
               color="#1f77b4", ls="-.", lw=1.0)
    ax.annotate("", xy=(cfg["rise_hand_s"], 1.02), xytext=(0.0, 1.02),
                arrowprops=dict(arrowstyle="<->", color="#d62728"))
    ax.text(0.5 * cfg["rise_hand_s"], 1.04, f"rise_hand = {1000 * cfg['rise_hand_s']:.0f} ms",
            fontsize=8, ha="center")
    pk_f = cfg["rise_hand_s"] + cfg["foot_delay_s"] + cfg["rise_foot_s"]
    ax.annotate("", xy=(pk_f, 0.62), xytext=(cfg["rise_hand_s"], 0.62),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(0.5 * (cfg["rise_hand_s"] + pk_f), 0.64,
            f"peak delay = {1000 * (pk_f - cfg['rise_hand_s']):.0f} ms", fontsize=8,
            ha="center")
    ax.set_xlabel("time from onset, s", fontsize=9)
    ax.set_ylabel("amplitude / amp_hand", fontsize=9)
    ax.set_title("Wave shape: hand = PWD mixture, foot = output of a single-pole "
                 f"low-pass filter, tau = {1000 * cfg['tau_s']:.0f} ms", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGS / "fig4_shape.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for fn in (fig_clean, fig_attenuation, fig_missed, fig_shape):
        p = fn()
        print(f"{p.relative_to(HERE)}  {p.stat().st_size} байт")


if __name__ == "__main__":
    main()
