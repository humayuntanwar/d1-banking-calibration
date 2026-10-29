#!/usr/bin/env python3
"""Held-out threshold fitting, as the d1-2B model card suggests.

Split: odd vs even rows of predictions.jsonl. BANKING77 test is grouped by
intent (40 each), so odd/even is 20/20 per intent.

  1. Pick a confidence cutoff on the fit half; report coverage and precision
     on the other half. That is the production protocol: 'calibrate any
     decision threshold on data from your own use.'
  2. Optionally refit the probability itself with isotonic regression (PAV)
     on the fit half, then pick the cutoff on the mapped scores.

Also reports the diagnostic of undoing d1's shipped temperature (1.3374).
"""
from __future__ import annotations

import json
from pathlib import Path

from metrics import (
    SHIPPED_T,
    apply_knots,
    at_least,
    ece,
    lowest_threshold_for_precision,
    pav,
    undo_temperature,
)

ROOT = Path(__file__).resolve().parent
TARGETS = (0.95, 0.98, 0.99)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def pack(rows: list[dict], conf: list[float]) -> dict:
    ok = [r["pred"] == r["label"] for r in rows]
    out = {
        "n": len(rows),
        "accuracy": sum(ok) / len(ok) if ok else 0.0,
        "mean_conf": sum(conf) / len(conf) if conf else 0.0,
        "ece_equal_10": ece(conf, ok),
        "at_0.90": at_least(conf, ok, 0.90),
        "at_0.95": at_least(conf, ok, 0.95),
        "cutoffs_point": {},
        "cutoffs_wilson": {},
    }
    for t in TARGETS:
        out["cutoffs_point"][str(t)] = lowest_threshold_for_precision(conf, ok, t)
        out["cutoffs_wilson"][str(t)] = lowest_threshold_for_precision(conf, ok, t, use_wilson_lo=True)
    return out


def eval_cutoff_on_test(fit_conf, fit_ok, test_conf, test_ok, target: float, use_wilson_lo: bool) -> dict | None:
    """Choose t on the fit half, score it on the test half."""
    chosen = lowest_threshold_for_precision(fit_conf, fit_ok, target, use_wilson_lo=use_wilson_lo)
    if chosen is None:
        return None
    t = chosen["threshold"]
    idx = [i for i in range(len(test_conf)) if test_conf[i] >= t]
    if not idx:
        return {
            "fit_threshold": t,
            "fit_n": chosen["n"],
            "fit_coverage": chosen["coverage"],
            "fit_accuracy": chosen["accuracy"],
            "test_n": 0,
            "test_coverage": 0.0,
            "test_accuracy": 0.0,
        }
    k = sum(test_ok[i] for i in idx)
    m = len(idx)
    from metrics import wilson_interval

    lo, hi = wilson_interval(k, m)
    return {
        "fit_threshold": t,
        "fit_n": chosen["n"],
        "fit_coverage": chosen["coverage"],
        "fit_accuracy": chosen["accuracy"],
        "test_n": m,
        "test_coverage": m / len(test_conf),
        "test_accuracy": k / m,
        "test_wilson95_lo": lo,
        "test_wilson95_hi": hi,
        "rule": chosen["rule"],
        "target": target,
    }


def main() -> None:
    rows = load_jsonl(ROOT / "results" / "predictions.jsonl")
    fit, test = rows[1::2], rows[0::2]
    fit_ok = [r["pred"] == r["label"] for r in fit]
    test_ok = [r["pred"] == r["label"] for r in test]
    fit_raw = [r["confidence"] for r in fit]
    test_raw = [r["confidence"] for r in test]

    report: dict = {
        "split": "even rows = test, odd rows = fit (20 tickets per intent each)",
        "fit": pack(fit, fit_raw),
        "test_raw": pack(test, test_raw),
        "threshold_transfer_point": {},
        "threshold_transfer_wilson": {},
    }
    for t in TARGETS:
        report["threshold_transfer_point"][str(t)] = eval_cutoff_on_test(
            fit_raw, fit_ok, test_raw, test_ok, t, False
        )
        report["threshold_transfer_wilson"][str(t)] = eval_cutoff_on_test(
            fit_raw, fit_ok, test_raw, test_ok, t, True
        )

    if "probs" in test[0]:
        fit_nt = [undo_temperature(r["probs"], SHIPPED_T) for r in fit]
        test_nt = [undo_temperature(r["probs"], SHIPPED_T) for r in test]
        report["no_temperature"] = {
            "note": f"softmax(s) recovered as p_i**{SHIPPED_T}, renormalised",
            "test": pack(test, test_nt),
            "threshold_transfer_point": {
                str(t): eval_cutoff_on_test(fit_nt, fit_ok, test_nt, test_ok, t, False)
                for t in TARGETS
            },
        }

    knots = pav(fit_raw, [float(ok) for ok in fit_ok])
    fit_iso = [apply_knots(knots, c) for c in fit_raw]
    test_iso = [apply_knots(knots, c) for c in test_raw]
    report["isotonic"] = {
        "n_knots": len(knots),
        "knots_head": knots[:8],
        "test": pack(test, test_iso),
        "threshold_transfer_point": {
            str(t): eval_cutoff_on_test(fit_iso, fit_ok, test_iso, test_ok, t, False)
            for t in TARGETS
        },
    }

    out = ROOT / "results" / "recalibration.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"fit n={len(fit)}  test n={len(test)}")
    print(f"test raw ECE={report['test_raw']['ece_equal_10']:.4f}  "
          f"acc={report['test_raw']['accuracy']:.3f}  mean_conf={report['test_raw']['mean_conf']:.3f}")
    if "no_temperature" in report:
        nt = report["no_temperature"]["test"]
        print(f"test no-T  ECE={nt['ece_equal_10']:.4f}  mean_conf={nt['mean_conf']:.3f}")
    iso = report["isotonic"]["test"]
    print(f"test isotonic ECE={iso['ece_equal_10']:.4f}  mean_conf={iso['mean_conf']:.3f}")
    print("threshold fit on odd rows, evaluated on even rows (raw confidence):")
    for t, block in report["threshold_transfer_point"].items():
        if block is None:
            print(f"  target {t}: no cutoff on the fit half")
            continue
        print(f"  target {t}: t>={block['fit_threshold']:.3f}  "
              f"fit cover {block['fit_coverage']:.1%} acc {block['fit_accuracy']:.1%} → "
              f"test cover {block['test_coverage']:.1%} acc {block['test_accuracy']:.1%} "
              f"n={block['test_n']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
