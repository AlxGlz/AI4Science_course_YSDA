"""Figures for the task-1 report: signal pairs with the reference markup on top."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FS = 320.0
RUN = Path(__file__).resolve().parent
SETS = RUN / "sets"
FIGS = RUN / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def load(set_name: str):
    d = SETS / set_name
    data = np.loadtxt(d / "signal.csv", delimiter=",", skiprows=1)
    ref = list(csv.DictReader(open(d / "reference.csv", encoding="utf-8")))
    for r in ref:
        for k in ("onset_s", "peak_s", "peak_foot_s", "amp_hand", "amp_foot", "foot_gain"):
            r[k] = float(r[k])
        r["missed_hand"] = int(r["missed_hand"])
        r["missed_foot"] = int(r["missed_foot"])
    return data[:, 0], data[:, 1], data[:, 2], ref


def draw(ax_h, ax_f, t, hand, foot, ref, t0, t1, title):
    sel = [b for b in ref if t0 <= b["onset_s"] <= t1]
    for ax, ch, missed_key in ((ax_h, hand, "missed_hand"), (ax_f, foot, "missed_foot")):
        ax.plot(t, ch, lw=0.7, color="#1f3b73")
        for b in sel:
            if b[missed_key]:
                ax.axvspan(b["onset_s"], b["onset_s"] + 0.45, color="crimson", alpha=0.12, lw=0)
            else:
                ax.axvline(b["onset_s"], color="#2a9d8f", lw=0.8, ls=":")
    for b in sel:
        if not b["missed_hand"]:
            ax_h.plot(b["peak_s"], np.interp(b["peak_s"], t, hand), "^",
                      ms=3.2, color="crimson")
        if not b["missed_foot"]:
            ax_f.plot(b["peak_foot_s"], np.interp(b["peak_foot_s"], t, foot), "^",
                      ms=3.2, color="darkorange")
    miss = [b for b in sel if b["missed_hand"] or b["missed_foot"]]
    if miss:
        b = miss[0]
        ax_h.annotate("пропуск", xy=(b["onset_s"] + 0.2, ax_h.get_ylim()[1]),
                      xytext=(b["onset_s"] + 0.45, ax_h.get_ylim()[1]),
                      fontsize=8, color="crimson", va="top")
    ax_h.set_xlim(t0, t1)
    ax_f.set_xlim(t0, t1)
    ax_h.set_title(f"{title} — палец руки")
    ax_f.set_title("палец стопы")
    ax_f.set_xlabel("время, с")
    for ax in (ax_h, ax_f):
        ax.set_ylabel("АЦП")


def window(t, x, t0, t1):
    m = (t >= t0) & (t <= t1)
    return t[m], x[m]


# --- figure 1: clean set, no defects -----------------------------------------
t, hand, foot, ref = load("set_clean")
t0, t1 = 40.0, 68.0
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
tt, hh = window(t, hand, t0, t1)
_, ff = window(t, foot, t0, t1)
draw(a1, a2, tt, hh, ff, ref, t0, t1, "set_clean: чистая пара, 28 с")
fig.text(0.01, 0.01, "пунктир — начало волны (onset), ▲ — пик волны; "
         "стрелка вниз — разметка руки, вверх — разметка стопы",
         fontsize=7, color="#555")
fig.tight_layout()
fig.savefig(FIGS / "fig1_clean.png")
plt.close(fig)

# --- figure 2: baseline set, attenuation episode ------------------------------
t, hand, foot, ref = load("set_baseline")
gains = np.array([b["foot_gain"] for b in ref])
onsets = np.array([b["onset_s"] for b in ref])
i = int(np.argmin(gains))
centre = onsets[i]
t0, t1 = centre - 20.0, centre + 20.0
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)
tt, hh = window(t, hand, t0, t1)
_, ff = window(t, foot, t0, t1)
draw(a1, a2, tt, hh, ff, ref, t0, t1, "set_baseline: затухание на стопе, 40 с")
a2.annotate(f"амплитуда стопы падает до {100 * gains[i]:.0f}% от номинала",
            xy=(centre, np.interp(centre, tt, ff)),
            xytext=(t0 + 1.5, np.max(ff)), fontsize=8, color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))
fig.tight_layout()
fig.savefig(FIGS / "fig2_attenuation.png")
plt.close(fig)

# --- figure 3: hard set, missed beat on both channels ------------------------
t, hand, foot, ref = load("set_hard")
cand = [b for b in ref if b["missed_hand"] and b["missed_foot"]]
b = cand[len(cand) // 2] if cand else [b for b in ref if b["missed_foot"]][0]
t0, t1 = b["onset_s"] - 6.0, b["onset_s"] + 6.0
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
tt, hh = window(t, hand, t0, t1)
_, ff = window(t, foot, t0, t1)
draw(a1, a2, tt, hh, ff, ref, t0, t1, "set_hard: пропущенный удар, шум и затухание, 12 с")
a1.annotate("нет волны", xy=(b["onset_s"] + 0.2, np.mean(hh)),
            xytext=(t0 + 0.6, np.max(hh)), fontsize=8, color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))
fig.tight_layout()
fig.savefig(FIGS / "fig3_missed.png")
plt.close(fig)

for p in sorted(FIGS.glob("*.png")):
    print(p.name, p.stat().st_size // 1024, "KB")
