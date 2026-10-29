"""Accuracy, calibration (ECE), auto-route curves, and isotonic recalibration.

Pure Python plus the stdlib. Used by score.py, recalibrate.py, and plot.py.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

# LiquidAI/d1-2B calibration.json "default" temperature. Choice questions have no
# per-bucket temperature, so every option softmax is scores / 1.3374.
SHIPPED_T = 1.337398453966105

# High-end bins: 77-way confidence piles up above 0.5, so equal-width 0.1 bins
# waste most of the diagram. These are the ones the README chart uses.
FOCUS_EDGES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0000001)
EQUAL_EDGES = tuple(i / 10 for i in range(11))  # 0.0, 0.1, ..., 1.0
ROUTE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)


def ece(conf: Sequence[float], correct: Sequence[bool], edges: Sequence[float] = EQUAL_EDGES) -> float:
    """Expected Calibration Error: bin-size-weighted |acc - mean confidence|."""
    n = len(conf)
    if n == 0:
        return 0.0
    error = 0.0
    for lo, hi in zip(edges, edges[1:]):
        # Last bin is closed on the right so confidence == 1.0 is counted.
        last = hi == edges[-1]
        idx = [i for i in range(n) if lo <= conf[i] < hi or (last and conf[i] == hi)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        mean_c = sum(conf[i] for i in idx) / len(idx)
        error += (len(idx) / n) * abs(acc - mean_c)
    return error


def reliability_bins(conf: Sequence[float], correct: Sequence[bool], edges: Sequence[float] = FOCUS_EDGES) -> list[dict]:
    n = len(conf)
    bins = []
    for lo, hi in zip(edges, edges[1:]):
        last = hi == edges[-1]
        idx = [i for i in range(n) if lo <= conf[i] < hi or (last and conf[i] == hi)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        mean_c = sum(conf[i] for i in idx) / len(idx)
        bins.append({
            "bin": f"{lo:.2f}-{min(hi, 1.0):.2f}",
            "lo": lo,
            "hi": min(hi, 1.0),
            "n": len(idx),
            "mean_conf": mean_c,
            "accuracy": acc,
            "gap": acc - mean_c,
        })
    return bins


def route_table(conf: Sequence[float], correct: Sequence[bool], thresholds: Sequence[float] = ROUTE_THRESHOLDS) -> list[dict]:
    n = len(conf)
    rows = []
    for t in thresholds:
        idx = [i for i in range(n) if conf[i] >= t]
        k = sum(correct[i] for i in idx)
        m = len(idx)
        lo, hi = wilson_interval(k, m) if m else (0.0, 0.0)
        rows.append({
            "threshold": t,
            "n": m,
            "coverage": m / n if n else 0.0,
            "accuracy": (k / m) if m else 0.0,
            "wilson95_lo": lo,
            "wilson95_hi": hi,
        })
    return rows


def at_least(conf: Sequence[float], correct: Sequence[bool], t: float) -> dict:
    n = len(conf)
    idx = [i for i in range(n) if conf[i] >= t]
    k = sum(correct[i] for i in idx)
    m = len(idx)
    lo, hi = wilson_interval(k, m) if m else (0.0, 0.0)
    return {
        "threshold": t,
        "n": m,
        "coverage": m / n if n else 0.0,
        "accuracy": (k / m) if m else 0.0,
        "wilson95_lo": lo,
        "wilson95_hi": hi,
    }


def lowest_threshold_for_precision(
    conf: Sequence[float],
    correct: Sequence[bool],
    target: float,
    min_n: int = 1,
    use_wilson_lo: bool = False,
) -> dict | None:
    """Lowest confidence cutoff whose routed tickets meet `target` precision.

    Tries unique rounded confidences high-to-low conceptually, but scans
    ascending so the first hit maximises coverage. `use_wilson_lo` requires
    the 95% Wilson lower bound (not the point estimate) to clear `target`.
    """
    n = len(conf)
    if n == 0:
        return None
    candidates = sorted({round(c, 4) for c in conf})
    best = None
    for t in candidates:
        idx = [i for i in range(n) if conf[i] >= t]
        m = len(idx)
        if m < min_n:
            continue
        k = sum(correct[i] for i in idx)
        acc = k / m
        lo, hi = wilson_interval(k, m)
        metric = lo if use_wilson_lo else acc
        if metric >= target:
            best = {
                "threshold": t,
                "n": m,
                "coverage": m / n,
                "accuracy": acc,
                "wilson95_lo": lo,
                "wilson95_hi": hi,
                "target": target,
                "rule": "wilson95_lo" if use_wilson_lo else "point_estimate",
            }
            break
    return best


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def undo_temperature(probs: dict[str, float], t: float = SHIPPED_T) -> float:
    """Undo d1's shipped softmax temperature: p_i^T, renormalised; return max.

    If p = softmax(s / T), then p_i^T ∝ exp(s_i), so this recovers softmax(s).
    """
    raised = [max(v, 1e-12) ** t for v in probs.values()]
    total = sum(raised)
    return max(raised) / total if total else 0.0


def pav(xs: Sequence[float], ys: Sequence[float]) -> list[tuple[float, float]]:
    """Isotonic regression (pool adjacent violators). Returns sorted knots (x, y)."""
    blocks: list[list[float]] = []  # [sum_y, count, max_x]
    for x, y in sorted(zip(xs, ys)):
        blocks.append([y, 1.0, x])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c, mx = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
            blocks[-1][2] = mx
    return [(b[2], b[0] / b[1]) for b in blocks]


def apply_knots(knots: Sequence[tuple[float, float]], x: float) -> float:
    if not knots:
        return x
    for kx, ky in knots:
        if x <= kx:
            return ky
    return knots[-1][1]


def summarise(rows: Sequence[dict]) -> dict:
    n = len(rows)
    correct = [r["pred"] == r["label"] for r in rows]
    conf = [float(r["confidence"]) for r in rows]
    times = [float(r["sec_per_ticket"]) for r in rows if "sec_per_ticket" in r]
    tokens = [int(r["input_tokens"]) for r in rows if "input_tokens" in r]
    wrong_sure = [r for r, ok in zip(rows, correct) if not ok and r["confidence"] >= 0.95]
    out = {
        "n": n,
        "accuracy": (sum(correct) / n) if n else 0.0,
        "mean_conf": (sum(conf) / n) if n else 0.0,
        "ece_equal_10": ece(conf, correct, EQUAL_EDGES),
        "ece_focus": ece(conf, correct, FOCUS_EDGES),
        "bins": reliability_bins(conf, correct, FOCUS_EDGES),
        "bins_equal": reliability_bins(conf, correct, EQUAL_EDGES),
        "route": route_table(conf, correct),
        "at_0.90": at_least(conf, correct, 0.90),
        "at_0.95": at_least(conf, correct, 0.95),
        "auto_route_point": {
            str(t): lowest_threshold_for_precision(conf, correct, t)
            for t in (0.95, 0.98, 0.99)
        },
        "auto_route_wilson": {
            str(t): lowest_threshold_for_precision(conf, correct, t, use_wilson_lo=True)
            for t in (0.95, 0.98, 0.99)
        },
        "wrong_at_95": len(wrong_sure),
        "wrong_at_95_pairs": Counter(f'{r["label"]} → {r["pred"]}' for r in wrong_sure).most_common(10),
    }
    if times:
        st = sorted(times)
        out["sec_per_ticket"] = {
            "mean": sum(st) / len(st),
            "median": st[len(st) // 2],
            "p90": st[int(0.9 * (len(st) - 1))],
            "min": st[0],
            "max": st[-1],
        }
    if tokens:
        out["input_tokens_mean"] = sum(tokens) / len(tokens)
    return out
