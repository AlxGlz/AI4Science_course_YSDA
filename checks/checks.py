"""Independent checks for the task-1 modulator (synthetic PPG pairs).

Run from the project root:

    .venv\\Scripts\\python checks\\checks.py
    .venv\\Scripts\\python checks\\checks.py --regenerate

Exit code 0 means every check passed. The checks read only the emitted files
(signal.csv, reference.csv, params.json) and measure the signal themselves, so
a mistake shared by generator and markup cannot pass unnoticed.

Two measurement details matter and are documented where they are used:
  * the running reference level is a 1 s median (a 3 s median is biased by 5-9%
    because the pulse occupies about half of the cardiac cycle);
  * peaks are located on a 40 ms moving average, because at the flat top of a
    wave white noise of 30 counts shifts a raw argmax by 20-30 ms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

PROJECT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT / "runs" / "t1-w1-r1"
SETS = RUN_DIR / "sets"
MODULATOR = RUN_DIR / "modulator.py"
FS = 320.0
SMOOTH = int(0.04 * FS) | 1

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    RESULTS.append((bool(ok), name, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


def load_signal(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]


def load_reference(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [{
        "beat_id": int(r["beat_id"]),
        "onset_s": float(r["onset_s"]),
        "peak_s": float(r["peak_s"]),
        "peak_foot_s": float(r["peak_foot_s"]),
        "amp_hand": float(r["amp_hand"]),
        "amp_foot": float(r["amp_foot"]),
        "missed_hand": bool(int(r["missed_hand"])),
        "missed_foot": bool(int(r["missed_foot"])),
        "foot_gain": float(r["foot_gain"]),
    } for r in rows]


def pedestal(x: np.ndarray, win_s: float = 1.0) -> np.ndarray:
    """Running reference level (DC + drift) as a 1 s median.

    One second spans 1-1.5 beats at 72-96 bpm: short enough to follow a
    250-900 count drift, long enough for the median to sit on the diastolic
    level. Measured amplitude bias on the generated sets: +0.2% (clean),
    -1.3% (baseline), +0.6% (hard); a 3 s median is biased by 5-9% because a
    pulse occupies about half of each cardiac cycle.
    """
    return median_filter(x, size=int(win_s * FS) | 1, mode="nearest")


def measure(x: np.ndarray, xs: np.ndarray, lo_s: float, hi_s: float,
            base_s: float) -> tuple[float, float]:
    """Peak time and wave height in [lo_s, hi_s].

    Wave height is defined exactly as in the markup: peak value minus the value
    at the wave onset (`base_s`), where the pulse starts from zero. Both samples
    come from the same baseline-subtracted signal a fraction of a second apart,
    so the residual pedestal bias cancels and only the drift slope over that
    interval matters (below 40 counts).

    The peak time comes from the smoothed signal `xs`: white noise shifts a raw
    argmax by 20-30 ms at the flat top of a wave.
    """
    lo = max(0, int(round(lo_s * FS)))
    hi = min(x.size, int(round(hi_s * FS)))
    i = lo + int(np.argmax(xs[lo:hi]))
    base = float(xs[min(xs.size - 1, max(0, int(round(base_s * FS))))])
    a, b = max(0, i - 5), min(x.size, i + 6)
    if b - a >= 5:
        tt = (np.arange(a, b) - i) / FS  # centred: an absolute time axis makes polyfit ill-conditioned
        coef = np.polyfit(tt, x[a:b], 2)
        if coef[0] < 0:
            tv = float(np.clip(-coef[1] / (2 * coef[0]), tt[0], tt[-1]))
            return (i + tv * FS) / FS, float(np.polyval(coef, tv)) - base
    return i / FS, float(x[i]) - base


def noise_sigma(x: np.ndarray) -> float:
    """White-noise sigma from second differences: var(d2) = 6 sigma^2 for smooth signal."""
    return float(np.std(np.diff(x, n=2)) / np.sqrt(6.0))


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def check_set(set_dir: Path) -> None:
    tag = f"[{set_dir.name}]"
    cfg = json.loads((set_dir / "params.json").read_text(encoding="utf-8"))
    p = cfg["config"]
    t, hand_raw, foot_raw = load_signal(set_dir / "signal.csv")
    ref = load_reference(set_dir / "reference.csv")

    hand = hand_raw - pedestal(hand_raw)
    foot = foot_raw - pedestal(foot_raw)
    hand_s = uniform_filter1d(hand, SMOOTH, mode="nearest")
    foot_s = uniform_filter1d(foot, SMOOTH, mode="nearest")

    # 1. sampling grid, duration, format
    fs_meas = 1.0 / float(np.median(np.diff(t)))
    check(abs(fs_meas - p["fs"]) < 0.01, f"{tag} fs = {p['fs']:.0f} Hz",
          f"measured {fs_meas:.4f} Hz")
    check(t.size == int(round(p["duration_s"] * p["fs"])), f"{tag} duration {p['duration_s']:.0f} s",
          f"{t.size} samples, last t = {t[-1]:.3f} s")
    check(bool(np.all(np.diff(t) > 0)), f"{tag} time axis strictly increasing")
    check(bool(np.allclose(hand_raw, np.round(hand_raw)) and np.allclose(foot_raw, np.round(foot_raw))),
          f"{tag} channels quantised to integer ADC counts")

    # 2. markup structure
    onsets = np.array([b["onset_s"] for b in ref])
    peaks = np.array([b["peak_s"] for b in ref])
    pfoot = np.array([b["peak_foot_s"] for b in ref])
    check(bool(np.all(np.diff(onsets) > 0.3)), f"{tag} onsets strictly increasing")
    check(bool(np.all(peaks > onsets) and np.all(pfoot > peaks)),
          f"{tag} onset < peak_hand < peak_foot on every beat")
    check([b["beat_id"] for b in ref] == list(range(len(ref))), f"{tag} beat ids contiguous")

    # 3. rhythm
    ibi = np.diff(onsets)
    hr_meas = 60.0 / float(np.median(ibi))
    check(abs(hr_meas - p["hr_bpm"]) / p["hr_bpm"] < 0.06,
          f"{tag} median HR {p['hr_bpm']:.0f} bpm", f"measured {hr_meas:.1f} bpm")
    n_expected = p["duration_s"] * p["hr_bpm"] / 60.0
    check(abs(len(ref) - n_expected) / n_expected < 0.10,
          f"{tag} beat count", f"{len(ref)} beats, expected ~{n_expected:.0f}")

    vis_h = [b for b in ref if not b["missed_hand"]]
    vis_f = [b for b in ref if not b["missed_foot"]]
    s_hand, s_foot = noise_sigma(hand_raw), noise_sigma(foot_raw)

    # 4. declared peaks sit on top of real waves
    dt_h, pa_h, pm_h = [], [], []
    for b in vis_h:
        pk_t, pk_v = measure(hand, hand_s, b["onset_s"] - 0.05, b["peak_s"] + 0.20,
                    b["onset_s"])
        dt_h.append(abs(pk_t - b["peak_s"]))
        pa_h.append(b["amp_hand"])
        pm_h.append(pk_v)
    check(float(np.median(dt_h)) < 0.015 and np.percentile(dt_h, 99) < 0.05,
          f"{tag} hand peaks coincide with wave maxima",
          f"median |dt| {1000 * float(np.median(dt_h)):.1f} ms, "
          f"p99 {1000 * float(np.percentile(dt_h, 99)):.0f} ms")
    share_h = float(np.mean(np.array(dt_h) < 0.030))
    check(share_h > 0.97, f"{tag} hand peak times within 30 ms of markup",
          f"{100 * share_h:.1f}% of {len(dt_h)} beats")
    # amplitude noise: an unbiased estimator of a peak still carries about 1.2 sigma,
    # so the tolerance is expressed in units of the measured noise, not in percent
    err_h = np.abs(np.array(pm_h) - np.array(pa_h))
    check(float(np.median(err_h)) < 3 * s_hand and np.percentile(err_h, 95) < 5 * s_hand,
          f"{tag} hand amplitudes match markup",
          f"median |err| {np.median(err_h):.0f} counts ({100 * np.median(err_h) / np.median(pa_h):.1f}%), "
          f"p95 {np.percentile(err_h, 95):.0f} counts; one sigma of noise is {s_hand:.0f}")
    scale_h = float(np.median(pm_h)) / float(np.median(pa_h))
    check(0.90 < scale_h < 1.20, f"{tag} hand amplitude scale reproduced",
          f"median measured/declared = {scale_h:.3f} over {len(pa_h)} beats; "
          f"peak estimator bias is ~{1.2 * s_hand / np.median(pa_h):.2f} of an amplitude")

    dt_f, pa_f, pm_f = [], [], []
    weak_f = 0
    for b in vis_f:
        if b["amp_foot"] < 3 * s_foot:
            weak_f += 1
            continue
        pk_t, pk_v = measure(foot, foot_s, b["peak_foot_s"] - 0.25, b["peak_foot_s"] + 0.20,
                        b["onset_s"] + p["foot_delay_s"])
        dt_f.append(abs(pk_t - b["peak_foot_s"]))
        pa_f.append(b["amp_foot"])
        pm_f.append(pk_v)
    note_f = f" ({weak_f} beats below 3 sigma of noise skipped)" if weak_f else ""
    check(float(np.median(dt_f)) < 0.015 and np.percentile(dt_f, 99) < 0.06,
          f"{tag} foot peaks coincide with wave maxima",
          f"median |dt| {1000 * float(np.median(dt_f)):.1f} ms, "
          f"p99 {1000 * float(np.percentile(dt_f, 99)):.0f} ms{note_f}")
    share_f = float(np.mean(np.array(dt_f) < 0.050))
    check(share_f > 0.95, f"{tag} foot peak times within 50 ms of markup",
          f"{100 * share_f:.1f}% of {len(dt_f)} beats{note_f}; "
          f"at SNR {20 * np.log10(np.median(pa_f) / s_foot):.0f} dB the noise-limited "
          f"timing error of a 0.15 s upstroke is ~11 ms")
    err_f = np.abs(np.array(pm_f) - np.array(pa_f))
    check(float(np.median(err_f)) < 3 * s_foot and np.percentile(err_f, 95) < 5 * s_foot,
          f"{tag} foot amplitudes match markup",
          f"median |err| {np.median(err_f):.0f} counts "
          f"({100 * np.median(err_f) / np.median(pa_f):.1f}%), "
          f"p95 {np.percentile(err_f, 95):.0f} counts over {len(err_f)} beats; "
          f"one sigma of noise is {s_foot:.0f}{note_f}")
    scale_f = float(np.median(pm_f)) / float(np.median(pa_f))
    check(0.90 < scale_f < 1.20, f"{tag} foot amplitude scale reproduced",
          f"median measured/declared = {scale_f:.3f} over {len(pa_f)} beats; "
          f"peak estimator bias is ~{1.2 * s_foot / np.median(pa_f):.2f} of an amplitude")

    # 5. missed beats carry no wave
    def local_ref(beat: dict, amp_key: str, missed_key: str, span: float = 5.0) -> float:
        near = [b[amp_key] for b in ref
                if not b[missed_key] and abs(b["onset_s"] - beat["onset_s"]) < span]
        return float(np.median(near)) if near else float(np.median(
            [b[amp_key] for b in ref if not b[missed_key]]))

    bumps = []
    for b in [b for b in ref if b["missed_hand"]]:
        _, v = measure(hand, hand_s, b["onset_s"] + 0.02, b["onset_s"] + 0.45, b["onset_s"])
        bumps.append(v / max(local_ref(b, "amp_hand", "missed_hand"), 3 * s_hand))
    check(max(bumps, default=0.0) < 0.40, f"{tag} missed hand beats are empty",
          f"{len(bumps)} beats, largest bump {100 * max(bumps, default=0):.0f}% "
          f"of neighbouring pulse")

    bumps_f = []
    floor_f = 6 * s_foot
    for b in [b for b in ref if b["missed_foot"]]:
        lo = b["onset_s"] + p["foot_delay_s"] + 0.02
        _, v = measure(foot, foot_s, lo, lo + 0.45, lo)
        ref_amp = max(local_ref(b, "amp_foot", "missed_foot"), floor_f)
        bumps_f.append(v / ref_amp)
    check(max(bumps_f, default=0.0) < 0.55, f"{tag} missed foot beats are empty",
          f"{len(bumps_f)} beats, largest bump {100 * max(bumps_f, default=0):.0f}% "
          f"of the neighbouring pulse or of the 6 sigma noise floor ({floor_f:.0f} counts); "
          f"a 140-sample noise maximum itself reaches ~2.7 sigma")

    # 6. hand -> foot propagation
    both = [b for b in ref if not (b["missed_hand"] or b["missed_foot"])]
    expect_peak = p["foot_delay_s"] + p["rise_foot_s"] - p["rise_hand_s"]
    d_markup = float(np.median([b["peak_foot_s"] - b["peak_s"] for b in both]))
    check(abs(d_markup - expect_peak) < 0.020,
          f"{tag} peak delay hand->foot ~{1000 * expect_peak:.0f} ms",
          f"markup {1000 * d_markup:.0f} ms (arrival {1000 * p['foot_delay_s']:.0f} ms "
          f"+ wave-width difference)")
    d_signal = []
    for b in both:
        if b["amp_foot"] < 3 * s_foot:
            continue
        th, _ = measure(hand, hand_s, b["onset_s"] - 0.05, b["peak_s"] + 0.20,
                    b["onset_s"])
        tf, _ = measure(foot, foot_s, b["peak_foot_s"] - 0.25, b["peak_foot_s"] + 0.20,
                        b["onset_s"] + p["foot_delay_s"])
        d_signal.append(tf - th)
    check(abs(float(np.median(d_signal)) - expect_peak) < 0.025,
          f"{tag} peak delay visible in signals",
          f"measured {1000 * float(np.median(d_signal)):.0f} ms "
          f"(IQR {1000 * float(np.percentile(d_signal, 25)):.0f}-"
          f"{1000 * float(np.percentile(d_signal, 75)):.0f} ms)")

    # 7. attenuation of the foot channel
    gains = np.array([b["foot_gain"] for b in vis_f])
    amps = np.array([b["amp_foot"] for b in vis_f])
    amps_h = np.array([b["amp_hand"] for b in vis_f])
    ratio = amps / (amps_h * gains)
    check(float(np.std(ratio) / np.mean(ratio)) < 0.15,
          f"{tag} foot amplitude follows foot_gain",
          f"amp_foot / (amp_hand * gain) CV {100 * float(np.std(ratio) / np.mean(ratio)):.1f}%")
    if p["n_attenuation"] > 0:
        check(float(gains.min()) < 0.6, f"{tag} attenuation episodes present",
              f"min gain {gains.min():.2f}, {p['n_attenuation']} episodes, "
              f"median gain {np.median(gains):.2f}")

    # 8. noise
    check(abs(s_hand - p["noise_hand"]) / p["noise_hand"] < 0.25,
          f"{tag} hand noise sigma ~{p['noise_hand']:.0f} counts", f"measured {s_hand:.1f}")
    check(abs(s_foot - p["noise_foot"]) / p["noise_foot"] < 0.25,
          f"{tag} foot noise sigma ~{p['noise_foot']:.0f} counts", f"measured {s_foot:.1f}")
    check(s_foot > s_hand, f"{tag} foot is noisier than hand", f"{s_foot:.1f} vs {s_hand:.1f}")
    med_h = float(np.median(pa_h))
    check(med_h / s_hand > 5.0, f"{tag} pulse stands above noise on hand",
          f"SNR {20 * np.log10(med_h / s_hand):.1f} dB")


def check_reproducibility() -> None:
    sys.path.insert(0, str(RUN_DIR))
    import modulator  # noqa: PLC0415 — import after sys.path setup

    a = modulator.build(modulator.Config(name="repro", seed=777, duration_s=60.0))
    b = modulator.build(modulator.Config(name="repro", seed=777, duration_s=60.0))
    h_a = hashlib.md5(a["hand"].astype(np.int64).tobytes()).hexdigest()
    h_b = hashlib.md5(b["hand"].astype(np.int64).tobytes()).hexdigest()
    check(h_a == h_b, "generator is deterministic for a fixed seed", h_a[:12])

    c = modulator.build(modulator.Config(name="repro", seed=778, duration_s=60.0))
    h_c = hashlib.md5(c["hand"].astype(np.int64).tobytes()).hexdigest()
    check(h_a != h_c, "different seed gives a different signal", h_c[:12])

    with tempfile.TemporaryDirectory() as tmp:
        for target in ("s1", "s2"):
            subprocess.run([sys.executable, str(MODULATOR), "--out", tmp,
                            "--preset", "clean", "--preset-name", target],
                           check=True, capture_output=True)
        m1 = md5(Path(tmp) / "s1" / "signal.csv")
        m2 = md5(Path(tmp) / "s2" / "signal.csv")
        check(m1 == m2, "CLI output stable across two runs", m1[:12])


def main() -> int:
    ap = argparse.ArgumentParser(description="Checks for the task-1 PPG modulator")
    ap.add_argument("--regenerate", action="store_true", help="rebuild the sets first")
    args = ap.parse_args()

    if args.regenerate or not SETS.exists():
        subprocess.run([sys.executable, str(MODULATOR), "--out", str(SETS), "--all"], check=True)

    for set_dir in sorted(d for d in SETS.iterdir() if d.is_dir()):
        check_set(set_dir)
    check_reproducibility()

    failed = [r for r in RESULTS if not r[0]]
    print("-" * 72)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for _, nm, dt in failed:
        print(f"  FAILED: {nm} — {dt}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
