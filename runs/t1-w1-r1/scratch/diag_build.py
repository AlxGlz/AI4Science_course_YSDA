"""Compare markup written to reference.csv with a fresh in-process build."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "runs/t1-w1-r1")
import modulator  # noqa: E402

cfg = modulator.Config(name="clean", **modulator.PRESETS["clean"])
data = modulator.build(cfg)
pulse = data["pulse_hand"]
fs = cfg.fs
print("rise_hand_s =", cfg.rise_hand_s, " dicrotic delay", cfg.dicrotic_delay_hand_s)
for k in (14, 15, 16, 17):
    b = data["beats"][k]
    lo = int(round(b.onset_s * fs))
    hi = min(pulse.size, lo + int(0.6 * fs))
    p = lo + int(np.argmax(pulse[lo:hi]))
    print(f"beat {k}: onset={b.onset_s:.3f} markup_peak={b.peak_s:.3f} "
          f"pulse_argmax={p / fs:.3f} (offset {1000 * (p / fs - b.onset_s):.0f} ms) "
          f"markup_amp={b.amp_hand:.0f} pulse_amp={pulse[p] - pulse[lo]:.0f}")
    print("   pulse:", " ".join(f"{pulse[lo + j]:.0f}" for j in range(0, 200, 10)))
    d = np.diff(pulse[lo:hi])
    print(f"   n sign changes of derivative near peak: {np.sum(np.diff(np.sign(d)) != 0)}")
