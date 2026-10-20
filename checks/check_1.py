#!/usr/bin/env python3
"""check_1.py — проверка прогона модулятора (задача 1).

Запуск: .venv\\Scripts\\python checks\\check_1.py <папка прогона>

Проверяет:
  1. формат файлов signal.csv / truth.csv / params.json (как в задаче);
  2. воспроизводимость: повторный запуск run.py в temp-папке даёт те же файлы;
  3. число ударов и долю пропусков против params.json;
  4. amp_hand и задержку стопы против самого сигнала (signal.csv).

Выход: печатает PASS/FAIL по каждой проверке, exit code 0 если всё прошло,
1 если хоть одна проверка не прошла.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SIGNAL_HEADER = ["t", "hand", "foot"]
EXPECTED_TRUTH_HEADER = [
    "beat", "t_hand_onset", "t_hand_peak",
    "t_foot_onset", "t_foot_peak", "amp_hand", "amp_foot", "present",
]
REQUIRED_PARAMS = ["duration", "fs", "hr", "k", "tau", "sigma", "drop", "seed"]

# Допуски (физически обоснованные, не подогнаны под конкретный результат).
BEAT_TOL = 2            # число ударов может отличаться на пару из-за вариабельности интервалов
DROP_TOL = 0.05         # доля пропусков, абсолютная погрешность
AMP_RTOL = 0.10         # относительная погрешность амплитуды руки (шум на пике)
AMP_ATOL = 1e-3         # абсолютный флор амплитуды
LAG_TOL_SAMPLES = 2     # задержка стопы, в отсчётах
REPRO_ATOL = 1e-9       # точность воспроизводимости (детерминированный seed)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", dtype=float)
    return df


def check_format(run_dir: Path) -> list[str]:
    """Проверка формата файлов прогона."""
    problems = []
    sig_path = run_dir / "signal.csv"
    truth_path = run_dir / "truth.csv"
    params_path = run_dir / "params.json"

    if not sig_path.exists():
        problems.append("нет signal.csv")
    if not truth_path.exists():
        problems.append("нет truth.csv")
    if not params_path.exists():
        problems.append("нет params.json")
    if problems:
        return problems

    # params.json
    try:
        params = json.loads(params_path.read_text())
    except Exception as e:
        return [f"params.json не читается: {e}"]
    missing = [k for k in REQUIRED_PARAMS if k not in params]
    if missing:
        problems.append(f"params.json: нет ключей {missing}")

    # signal.csv
    try:
        sig = _read_csv(sig_path)
    except Exception as e:
        problems.append(f"signal.csv не читается как CSV с десятичной точкой: {e}")
        sig = None
    if sig is not None:
        if list(sig.columns) != EXPECTED_SIGNAL_HEADER:
            problems.append(f"signal.csv: заголовок {list(sig.columns)} != {EXPECTED_SIGNAL_HEADER}")
        else:
            t = sig["t"].to_numpy()
            fs = params.get("fs")
            if fs:
                if not np.allclose(np.diff(t), 1.0 / fs, atol=1e-6):
                    problems.append("signal.csv: шаг по времени != 1/fs")
                if not np.isclose(t[-1], params.get("duration"), atol=1.0 / fs):
                    problems.append("signal.csv: последний момент времени != duration")

    # truth.csv
    try:
        tr = _read_csv(truth_path)
    except Exception as e:
        problems.append(f"truth.csv не читается как CSV с десятичной точкой: {e}")
        tr = None
    if tr is not None:
        if list(tr.columns) != EXPECTED_TRUTH_HEADER:
            problems.append(f"truth.csv: заголовок {list(tr.columns)} != {EXPECTED_TRUTH_HEADER}")
        else:
            present = tr["present"].to_numpy()
            if not set(np.unique(present)).issubset({0.0, 1.0}):
                problems.append("truth.csv: present содержит значения вне {0,1}")
            # у пропущенных ударов amp_foot == 0
            bad = tr.loc[present == 0, "amp_foot"]
            if not np.allclose(bad, 0.0, atol=1e-9):
                problems.append("truth.csv: у пропущенных ударов amp_foot != 0")
            # у присутствующих ударов моменты стопы позже моментов руки
            pr = tr.loc[present == 1]
            if len(pr) and not (pr["t_foot_peak"] > pr["t_hand_peak"]).all():
                problems.append("truth.csv: у присутствующих ударов t_foot_peak <= t_hand_peak")
            # beat идёт последовательно от 1
            if not np.array_equal(tr["beat"].to_numpy(), np.arange(1, len(tr) + 1)):
                problems.append("truth.csv: beat не идёт последовательно от 1")

    return problems


def check_reproducible(run_dir: Path, run_py: Path) -> list[str]:
    """Перезапуск run.py в temp-папке должен дать те же файлы."""
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            subprocess.run(
                [sys.executable, str(run_py), str(tmp_dir)],
                cwd=str(run_dir), check=True, capture_output=True, text=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as e:
            return [f"run.py упал при перезапуске: {e.stderr.strip()[:300]}"]
        except subprocess.TimeoutExpired:
            return ["run.py не завершился за 120 c"]

        for name in ["signal.csv", "truth.csv", "params.json"]:
            saved = run_dir / name
            re = tmp_dir / name
            if not re.exists():
                problems.append(f"перезапуск не создал {name}")
                continue
            if name == "params.json":
                if json.loads(saved.read_text()) != json.loads(re.read_text()):
                    problems.append("params.json: перезапуск дал другие параметры")
            else:
                a = _read_csv(saved).to_numpy()
                b = _read_csv(re).to_numpy()
                if a.shape != b.shape or not np.allclose(a, b, atol=REPRO_ATOL, rtol=0):
                    problems.append(f"{name}: перезапуск дал другой результат")
    return problems


def check_beats_and_drop(run_dir: Path) -> list[str]:
    """Число ударов и доля пропусков против params.json."""
    problems = []
    params = json.loads((run_dir / "params.json").read_text())
    tr = _read_csv(run_dir / "truth.csv")
    duration, hr, drop = params["duration"], params["hr"], params["drop"]
    n = len(tr)
    expected = int(round(duration * hr / 60.0))
    if abs(n - expected) > BEAT_TOL:
        problems.append(f"число ударов {n} != ожидаемое ~{expected} (±{BEAT_TOL})")
    drop_actual = float((tr["present"] == 0).mean())
    if abs(drop_actual - drop) > DROP_TOL:
        problems.append(f"доля пропусков {drop_actual:.3f} != заданная {drop} (±{DROP_TOL})")
    return problems


def check_amp_hand(run_dir: Path) -> list[str]:
    """amp_hand из truth должен соответствовать амплитуде волны в signal.csv."""
    problems = []
    params = json.loads((run_dir / "params.json").read_text())
    fs = params["fs"]
    sig = _read_csv(run_dir / "signal.csv")
    tr = _read_csv(run_dir / "truth.csv")
    hand = sig["hand"].to_numpy()
    t = sig["t"].to_numpy()
    rel_errs = []
    for _, row in tr.iterrows():
        i_onset = int(round(row["t_hand_onset"] * fs))
        i_peak = int(round(row["t_hand_peak"] * fs))
        if i_peak >= len(hand) or i_onset >= len(hand):
            problems.append("truth: момент пика/начала руки вне сигнала")
            continue
        amp_meas = hand[i_peak] - hand[i_onset]
        amp_true = row["amp_hand"]
        tol = AMP_RTOL * abs(amp_true) + AMP_ATOL
        if abs(amp_meas - amp_true) > tol:
            rel_errs.append(abs(amp_meas - amp_true) / max(abs(amp_true), 1e-9))
    if rel_errs:
        problems.append(
            f"amp_hand не сходится с сигналом: {len(rel_errs)}/{len(tr)} ударов, "
            f"макс. отн. ошибка {max(rel_errs):.3f}"
        )
    return problems


def check_foot_delay(run_dir: Path) -> list[str]:
    """Задержка стопы (tau) должна соответствовать сдвигу каналов в signal.csv."""
    problems = []
    params = json.loads((run_dir / "params.json").read_text())
    fs = params["fs"]
    tau = params["tau"]
    sig = _read_csv(run_dir / "signal.csv")
    tr = _read_csv(run_dir / "truth.csv")
    hand = sig["hand"].to_numpy()
    foot = sig["foot"].to_numpy()
    lag_tol = LAG_TOL_SAMPLES / fs
    lags = []
    pr = tr[tr["present"] == 1]
    for _, row in pr.iterrows():
        i_peak = int(round(row["t_hand_peak"] * fs))
        win = int(round(2 * tau * fs)) + 1
        lo = i_peak
        hi = min(i_peak + win, len(foot))
        if lo >= hi:
            continue
        seg = foot[lo:hi]
        lag = int(np.argmax(seg)) / fs
        lags.append(lag)
    if not lags:
        problems.append("нет присутствующих ударов для проверки задержки стопы")
        return problems
    med_lag = float(np.median(lags))
    if abs(med_lag - tau) > lag_tol:
        problems.append(
            f"задержка стопы из сигнала {med_lag:.4f} с != заданная tau={tau} (±{lag_tol:.4f})"
        )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Проверка прогона модулятора (задача 1)")
    ap.add_argument("run_dir", help="папка прогона")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"FAIL: папка прогона не найдена: {run_dir}")
        return 1
    run_py = run_dir / "run.py"
    if not run_py.exists():
        print(f"FAIL: в папке прогона нет run.py: {run_dir}")
        return 1

    checks = [
        ("формат файлов", check_format(run_dir)),
        ("воспроизводимость", check_reproducible(run_dir, run_py)),
        ("число ударов и доля пропусков", check_beats_and_drop(run_dir)),
        ("amp_hand против сигнала", check_amp_hand(run_dir)),
        ("задержка стопы против сигнала", check_foot_delay(run_dir)),
    ]

    all_ok = True
    for name, problems in checks:
        if problems:
            all_ok = False
            print(f"FAIL [{name}]:")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"PASS [{name}]")

    print("ИТОГ:", "OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())