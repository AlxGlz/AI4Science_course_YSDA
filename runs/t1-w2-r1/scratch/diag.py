"""Быстрая диагностика модели в памяти (черновик, не часть проверки).

Считает те же величины, что и checks/checks.py (педистал 1 с, сглаживание
40 мс, параболическое уточнение пика), но без записи файлов — чтобы подбирать
параметры формы до запуска штатной проверки.

Запуск:  .venv\\Scripts\\python runs\\t1-w2-r1\\scratch\\diag.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import modulator  # noqa: E402

spec = importlib.util.spec_from_file_location("official_checks", PROJECT / "checks" / "checks.py")
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

FS = 320.0


def ped(x):
    return median_filter(x, size=int(1.0 * FS) | 1, mode="nearest")


def run(name: str) -> None:
    cfg = modulator.PRESETS[name]()
    res = modulator.build(cfg)
    hand = res["hand"].astype(float) - ped(res["hand"].astype(float))
    foot = res["foot"].astype(float) - ped(res["foot"].astype(float))
    hs = uniform_filter1d(hand, chk.SMOOTH, mode="nearest")
    fs_ = uniform_filter1d(foot, chk.SMOOTH, mode="nearest")
    ref = res["reference"]
    p = res["config"]
    s_h = chk.noise_sigma(res["hand"].astype(float))
    s_f = chk.noise_sigma(res["foot"].astype(float))

    out = [f"=== {name}: W {p['design']['width_s']:.3f} s, min IBI "
           f"{p['design']['min_ibi_s']:.3f} s, rise_h {p['rise_hand_s']:.3f} "
           f"(задано {p['design']['rise_hand_s']:.3f}), rise_f {p['rise_foot_s']:.3f} "
           f"(задано {p['design']['rise_foot_s']:.3f})"]
    out.append(f"    шум: рука {s_h:.1f} (задано {p['noise_hand']:.0f}), "
               f"стопа {s_f:.1f} (задано {p['noise_foot']:.0f})")

    def stats(x, xs, beats, lo_off, hi_off, base_off, amp_key, label):
        dt, ratio, err = [], [], []
        for b in beats:
            pk, pv = chk.measure(x, xs, b[lo_off[0]] + lo_off[1], b[hi_off[0]] + hi_off[1],
                                 b[base_off[0]] + base_off[1])
            dt.append(abs(pk - b[hi_off[2]]))
            ratio.append(pv / b[amp_key])
            err.append(pv - b[amp_key])
        dt = np.array(dt); ratio = np.array(ratio); err = np.array(err)
        out.append(f"    {label}: |dt| мед {1000*np.median(dt):.1f} мс, p99 "
                   f"{1000*np.percentile(dt, 99):.0f} мс, доля <30 мс "
                   f"{100*np.mean(dt < 0.030):.1f}% | amp изм/объявл "
                   f"{np.median(ratio):.3f} |err| мед {np.median(np.abs(err)):.0f} "
                   f"({np.median(np.abs(err))/ (s_h if label=='рука' else s_f):.1f} sigma)")

    vis_h = [b for b in ref if not b["missed_hand"]]
    vis_f = [b for b in ref if not b["missed_foot"]]
    stats(hand, hs, vis_h, ("onset_s", -0.05), ("peak_s", 0.20, "peak_s"),
          ("onset_s", 0.0), "amp_hand", "рука")
    stats(foot, fs_, vis_f, ("peak_foot_s", -0.25), ("peak_foot_s", 0.20, "peak_foot_s"),
          ("onset_s", p["foot_delay_s"]), "amp_foot", "стопа")

    bumps = []
    for b in [b for b in ref if b["missed_hand"]]:
        _, v = chk.measure(hand, hs, b["onset_s"] + 0.02, b["onset_s"] + 0.45, b["onset_s"])
        near = [x["amp_hand"] for x in ref if not x["missed_hand"]
                and abs(x["onset_s"] - b["onset_s"]) < 5.0]
        bumps.append(v / max(float(np.median(near)), 3 * s_h))
    bumps_f = []
    for b in [b for b in ref if b["missed_foot"]]:
        lo = b["onset_s"] + p["foot_delay_s"] + 0.02
        _, v = chk.measure(foot, fs_, lo, lo + 0.45, lo)
        near = [x["amp_foot"] for x in ref if not x["missed_foot"]
                and abs(x["onset_s"] - b["onset_s"]) < 5.0]
        bumps_f.append(v / max(float(np.median(near)), 6 * s_f))
    out.append(f"    пропуски: рука max {100*max(bumps, default=0):.0f}% (порог 40%), "
               f"стопа max {100*max(bumps_f, default=0):.0f}% (порог 55%)")

    gains = np.array([b["foot_gain"] for b in vis_f])
    amps = np.array([b["amp_foot"] for b in vis_f])
    amps_h = np.array([b["amp_hand"] for b in vis_f])
    r = amps / (amps_h * gains)
    out.append(f"    gain: min {gains.min():.3f}, медиана {np.median(gains):.3f}, "
               f"CV отношения {100*np.std(r)/np.mean(r):.2f}%")
    print("\n".join(out))


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["clean", "baseline", "hard"]:
        run(nm)
