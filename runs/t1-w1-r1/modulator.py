"""Synthetic PPG modulator: paired "finger of hand / toe" signals with ground truth.

Why this model (numbers measured on data/private, see real_stats*.json):
  * rhythm     73-81 bpm, IBI p05..p95 0.67-0.89 s, occasional IBI up to 2.1 s
               (missed beats)          -> RR = base * (1 + jitter*z) * (1 + RSA)
  * pulse band 0.5-8 Hz RMS 213-241 counts (hand), 256-348 (foot)
               -> alpha-function beats of ~520 counts on top of DC ~1.5e4
  * noise      >12 Hz RMS 32 counts (hand)  -> white Gaussian sigma 30-45 in presets
  * drift      std 1000-1700 counts, range +-4000, mostly < 0.4 Hz
               -> slow sines + interpolated random walk
  * delay      hand -> foot 81-84 ms (IQR 3-12 ms)
               -> foot beats shifted by foot_delay_s + small jitter
  * foot       amplitude envelope min/max 0.14-0.19 inside a record
               -> attenuation episodes with smooth ramps

The reference markup is measured on the noise-free pulse component, therefore
peak times and amplitudes are exact by construction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

FS_DEFAULT = 320.0


@dataclass
class Config:
    """All generator knobs. Units: seconds, counts (ADC), bpm."""

    name: str = "baseline"
    duration_s: float = 420.0
    fs: float = FS_DEFAULT
    seed: int = 101

    # rhythm
    hr_bpm: float = 78.0
    hr_sd_bpm: float = 3.0
    rsa_depth: float = 0.05
    resp_hz: float = 0.22

    # pulse amplitude
    amp_hand: float = 520.0
    amp_foot: float = 0.95  # foot/hand peak ratio (measured 0.72-1.05 on data/private)
    amp_beat_cv: float = 0.09
    amp_resp_depth: float = 0.10

    # pulse shape, hand (alpha-function: peak at rise_s, dicrotic wave after)
    rise_hand_s: float = 0.135
    shape_a_hand: float = 4.0
    dicrotic_delay_hand_s: float = 0.21
    dicrotic_ratio_hand: float = 0.26
    dicrotic_rise_hand_s: float = 0.09

    # pulse shape, foot (slower upstroke, flatter dicrotic wave)
    rise_foot_s: float = 0.155
    shape_a_foot: float = 3.4
    dicrotic_delay_foot_s: float = 0.24
    dicrotic_ratio_foot: float = 0.20
    dicrotic_rise_foot_s: float = 0.10

    # propagation
    foot_delay_s: float = 0.082
    foot_delay_jitter_s: float = 0.004

    # sensor / ADC
    dc_hand: float = 15500.0
    dc_foot: float = 22000.0
    noise_hand: float = 30.0
    noise_foot: float = 45.0
    drift_std_hand: float = 550.0
    drift_std_foot: float = 350.0
    resp_baseline_hand: float = 60.0
    resp_baseline_foot: float = 35.0

    # defects
    n_missed_hand: int = 3
    n_missed_foot_only: int = 3
    n_attenuation: int = 2
    attenuation_depth: float = 0.30
    attenuation_dur_s: tuple[float, float] = (15.0, 35.0)
    attenuation_ramp_s: float = 2.0


@dataclass
class Beat:
    beat_id: int
    onset_s: float
    peak_s: float
    peak_foot_s: float
    amp_hand: float
    amp_foot: float
    missed_hand: bool
    missed_foot: bool
    foot_gain: float


PRESETS: dict[str, dict] = {
    "clean": dict(
        duration_s=300.0, hr_bpm=72.0, hr_sd_bpm=2.5, noise_hand=15.0, noise_foot=20.0,
        n_missed_hand=1, n_missed_foot_only=0, n_attenuation=0, amp_foot=1.00,
        drift_std_hand=250.0, drift_std_foot=180.0, seed=101,
    ),
    "baseline": dict(seed=202),
    "hard": dict(
        hr_bpm=96.0, hr_sd_bpm=6.0, noise_hand=45.0, noise_foot=70.0,
        n_missed_hand=6, n_missed_foot_only=9, n_attenuation=3, attenuation_depth=0.15,
        amp_foot=0.85, drift_std_hand=900.0, drift_std_foot=600.0, seed=303,
    ),
}


def alpha_pulse(tau: np.ndarray, rise_s: float, a: float) -> np.ndarray:
    """Unit-height pulse starting at tau=0, peaking exactly at tau=rise_s."""
    out = np.zeros_like(tau)
    m = tau > 0
    if not np.any(m):
        return out
    u = tau[m] / rise_s
    out[m] = u**a * np.exp(a * (1.0 - u))
    return out


def wave(tau: np.ndarray, rise: float, a: float, d_delay: float, d_ratio: float,
         d_rise: float) -> np.ndarray:
    """Systolic + dicrotic wave, unit amplitude at the systolic peak."""
    main = alpha_pulse(tau, rise, a)
    dicro = alpha_pulse(tau - d_delay, d_rise, 3.0) * d_ratio
    return main + dicro


def soft_ramp(x: np.ndarray) -> np.ndarray:
    """Raised-cosine ramp on [0, 1], clamped outside."""
    xc = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * xc)


def schedule_beats(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Beat onsets: base RR modulated by beat noise and respiratory sinus arrhythmia."""
    rr0 = 60.0 / cfg.hr_bpm
    onsets = [5.0]
    while onsets[-1] < cfg.duration_s - 2.0:
        rel_jitter = rng.normal(0.0, cfg.hr_sd_bpm / cfg.hr_bpm)
        rsa = cfg.rsa_depth * np.sin(2 * np.pi * cfg.resp_hz * onsets[-1] + 0.7)
        rr = rr0 * (1.0 + rel_jitter) * (1.0 + rsa)
        onsets.append(onsets[-1] + max(rr, 0.35 * rr0))
    return np.array(onsets, dtype=float)


def attenuation_envelope(t: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative gain 1.0 outside episodes, attenuation_depth inside."""
    gain = np.ones_like(t)
    if cfg.n_attenuation == 0:
        return gain
    span = cfg.duration_s - 40.0
    starts = np.sort(rng.uniform(20.0, 20.0 + span, cfg.n_attenuation))
    for s in starts:
        dur = rng.uniform(*cfg.attenuation_dur_s)
        depth = cfg.attenuation_depth * rng.uniform(0.7, 1.0)
        ramp = cfg.attenuation_ramp_s
        up = soft_ramp((t - s) / ramp)
        down = soft_ramp((s + dur - t) / ramp)
        gain *= 1.0 - (1.0 - depth) * np.minimum(up, down)
    return gain


def drift(t: np.ndarray, std: float, resp_amp: float, resp_hz: float,
          rng: np.random.Generator) -> np.ndarray:
    """Slow baseline wander: 3 slow sines + interpolated random walk + respiration."""
    out = np.zeros_like(t)
    for _ in range(3):
        f = rng.uniform(0.004, 0.05)
        out += rng.uniform(0.4, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    knots = np.arange(t[0], t[-1] + 60.0, 25.0)
    walk = np.cumsum(rng.normal(0.0, 1.0, knots.size))
    walk -= walk.mean()
    out += 0.8 * np.interp(t, knots, walk) / max(np.std(walk), 1e-9)
    out += resp_amp / max(std, 1e-9) * 0.1 * np.sin(2 * np.pi * resp_hz * t + 1.3)
    return out / max(np.std(out), 1e-9) * std


def build(cfg: Config) -> dict:
    rng = np.random.default_rng(cfg.seed)
    fs = cfg.fs
    n = int(round(cfg.duration_s * fs))
    t = np.arange(n) / fs

    onsets = schedule_beats(cfg, rng)
    onsets = onsets[(onsets > 1.0) & (onsets < cfg.duration_s - 1.0)]
    nb = onsets.size

    # --- defects -------------------------------------------------------------
    def pick(count: int, taken: set[int], margin: int = 8) -> list[int]:
        free = [i for i in range(margin, nb - margin) if all(abs(i - j) > 1 for j in taken)]
        if not free:
            return []
        idx = rng.choice(free, size=min(count, len(free)), replace=False)
        return [int(i) for i in idx]

    missed_hand = set(pick(cfg.n_missed_hand, set()))
    missed_foot = set(missed_hand) | set(pick(cfg.n_missed_foot_only, missed_hand))

    # --- per-beat amplitude and foot gain ------------------------------------
    beat_gain = np.exp(rng.normal(0.0, cfg.amp_beat_cv, nb) - 0.5 * cfg.amp_beat_cv**2)
    resp_amp_mod = 1.0 + cfg.amp_resp_depth * np.sin(2 * np.pi * cfg.resp_hz * onsets + 2.1)
    amp_hand = cfg.amp_hand * beat_gain * resp_amp_mod
    att = attenuation_envelope(t, cfg, rng)
    att_at_beat = np.interp(np.minimum(onsets + cfg.rise_hand_s, t[-1]), t, att)
    foot_gain = cfg.amp_foot * att_at_beat * np.exp(
        rng.normal(0.0, cfg.amp_beat_cv, nb) - 0.5 * cfg.amp_beat_cv**2
    )
    delay = cfg.foot_delay_s + rng.normal(0.0, cfg.foot_delay_jitter_s, nb)

    # --- pulse components (noise free) ---------------------------------------
    pulse_hand = np.zeros(n)
    pulse_foot = np.zeros(n)
    for k in range(nb):
        if k not in missed_hand:
            lo = int(round(onsets[k] * fs))
            hi = min(n, lo + int(0.7 * fs))
            tau = t[lo:hi] - onsets[k]
            pulse_hand[lo:hi] += amp_hand[k] * wave(
                tau, cfg.rise_hand_s, cfg.shape_a_hand, cfg.dicrotic_delay_hand_s,
                cfg.dicrotic_ratio_hand, cfg.dicrotic_rise_hand_s)
        if k not in missed_foot:
            lo = int(round((onsets[k] + delay[k]) * fs))
            hi = min(n, lo + int(0.7 * fs))
            tau = t[lo:hi] - (onsets[k] + delay[k])
            pulse_foot[lo:hi] += amp_hand[k] * foot_gain[k] * wave(
                tau, cfg.rise_foot_s, cfg.shape_a_foot, cfg.dicrotic_delay_foot_s,
                cfg.dicrotic_ratio_foot, cfg.dicrotic_rise_foot_s)

    # --- reference markup measured on the noise-free pulse --------------------
    beats: list[Beat] = []
    win = int(0.6 * fs)
    for k in range(nb):
        on = onsets[k]
        lo = int(round(on * fs))
        hi = min(n, lo + win)
        peak_h, amp_h, peak_f, amp_f = on, 0.0, on, 0.0
        if k not in missed_hand:
            p = lo + int(np.argmax(pulse_hand[lo:hi]))
            peak_h = p / fs
            amp_h = float(pulse_hand[p] - pulse_hand[lo])
        if k not in missed_foot:
            onf = on + delay[k]
            lof = int(round(onf * fs))
            hif = min(n, lof + win)
            p = lof + int(np.argmax(pulse_foot[lof:hif]))
            peak_f = p / fs
            amp_f = float(pulse_foot[p] - pulse_foot[lof])
        else:
            peak_f = on + delay[k] + cfg.rise_foot_s

        beats.append(Beat(
            beat_id=k,
            onset_s=round(float(on), 4),
            peak_s=round(float(peak_h if k not in missed_hand else on + cfg.rise_hand_s), 4),
            peak_foot_s=round(float(peak_f), 4),
            amp_hand=round(amp_h, 2),
            amp_foot=round(amp_f, 2),
            missed_hand=k in missed_hand,
            missed_foot=k in missed_foot,
            foot_gain=round(float(foot_gain[k]), 4),
        ))

    # --- observable channels -------------------------------------------------
    hand_clean = cfg.dc_hand + drift(t, cfg.drift_std_hand, cfg.resp_baseline_hand,
                                     cfg.resp_hz, rng) + pulse_hand
    foot_clean = cfg.dc_foot + drift(t, cfg.drift_std_foot, cfg.resp_baseline_foot,
                                     cfg.resp_hz, rng) + pulse_foot
    hand = np.round(hand_clean + rng.normal(0.0, cfg.noise_hand, n))
    foot = np.round(foot_clean + rng.normal(0.0, cfg.noise_foot, n))

    return {
        "config": cfg, "t": t, "hand": hand, "foot": foot,
        "pulse_hand": pulse_hand, "pulse_foot": pulse_foot,
        "beats": beats, "onsets": onsets,
        "missed_hand": sorted(missed_hand), "missed_foot": sorted(missed_foot),
        "noise_hand": float(np.std(hand - hand_clean)),
        "noise_foot": float(np.std(foot - foot_clean)),
    }


def write_set(out: Path, cfg: Config, data: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t = data["t"]
    hand = data["hand"].astype(np.int64)
    foot = data["foot"].astype(np.int64)
    with open(out / "signal.csv", "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write("t_s,hand,foot\n")
        for i in range(t.size):
            fh.write(f"{t[i]:.6f},{hand[i]},{foot[i]}\n")

    rows = ["beat_id,onset_s,peak_s,peak_foot_s,amp_hand,amp_foot,missed_hand,missed_foot,foot_gain"]
    for b in data["beats"]:
        rows.append(
            f"{b.beat_id},{b.onset_s:.4f},{b.peak_s:.4f},{b.peak_foot_s:.4f},"
            f"{b.amp_hand:.2f},{b.amp_foot:.2f},{int(b.missed_hand)},{int(b.missed_foot)},"
            f"{b.foot_gain:.4f}"
        )
    with open(out / "reference.csv", "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write("\n".join(rows) + "\n")

    summary = {
        "config": asdict(cfg),
        "n_samples": int(t.size),
        "n_beats": len(data["beats"]),
        "n_visible_hand": int(sum(not b.missed_hand for b in data["beats"])),
        "n_visible_foot": int(sum(not b.missed_foot for b in data["beats"])),
        "n_missed_hand": len(data["missed_hand"]),
        "n_missed_foot": len(data["missed_foot"]),
        "measured": {
            "hr_bpm": round(float(60.0 / np.median(np.diff(data["onsets"]))), 2),
            "ibi_p05_p95_s": [round(float(np.percentile(np.diff(data["onsets"]), 5)), 3),
                              round(float(np.percentile(np.diff(data["onsets"]), 95)), 3)],
            "amp_hand_median": round(float(np.median([b.amp_hand for b in data["beats"]
                                                      if not b.missed_hand])), 1),
            "amp_foot_median": round(float(np.median([b.amp_foot for b in data["beats"]
                                                      if not b.missed_foot])), 1),
            "noise_hand_std": round(data["noise_hand"], 1),
            "noise_foot_std": round(data["noise_foot"], 1),
            "delay_ms_median": round(1000.0 * float(np.median(
                [b.peak_foot_s - b.peak_s for b in data["beats"] if not b.missed_foot])), 1),
            "foot_arrival_delay_ms": round(1000.0 * cfg.foot_delay_s, 1),
            "foot_peak_delay_ms": round(1000.0 * (cfg.foot_delay_s + cfg.rise_foot_s
                                                  - cfg.rise_hand_s), 1),
        },
    }
    (out / "params.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2).replace("\n", "\r\n"),
        encoding="utf-8", newline="")


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic PPG pair generator")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--preset", default="baseline", choices=sorted(PRESETS) + ["baseline"])
    ap.add_argument("--preset-name", default=None, help="destination folder name")
    ap.add_argument("--all", action="store_true", help="generate every preset")
    args = ap.parse_args()

    if args.all:
        for name, over in PRESETS.items():
            cfg = Config(name=name, **over)
            data = build(cfg)
            write_set(args.out / f"set_{name}", cfg, data)
            print(f"set_{name}: {len(data['beats'])} beats, "
                  f"{len(data['missed_hand'])} missed (hand), "
                  f"{len(data['missed_foot'])} missed (foot) -> {args.out / f'set_{name}'}")
        return

    cfg = Config(name=args.preset, **PRESETS[args.preset])
    data = build(cfg)
    target = args.out / args.preset_name if args.preset_name else args.out
    write_set(target, cfg, data)
    print(f"{cfg.name}: {len(data['beats'])} beats -> {target}")


if __name__ == "__main__":
    main()
