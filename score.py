#!/usr/bin/env python3
"""Score predictions.jsonl: accuracy, ECE, reliability bins, auto-route curve."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from metrics import summarise

ROOT = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def compact(obj):
    """JSON dump that keeps floats readable without turning them into 17-digit noise."""
    return json.dumps(obj, indent=2, default=_default)


def _default(o):
    raise TypeError(o)


def main() -> None:
    pred_path = ROOT / "results" / "predictions.jsonl"
    rows = load_jsonl(pred_path)
    report = summarise(rows)
    hw = ROOT / "results" / "hardware.json"
    if hw.exists():
        report["hardware"] = json.loads(hw.read_text())
    out = ROOT / "results" / "metrics.json"
    out.write_text(compact(report) + "\n")
    csv_path = ROOT / "results" / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["text", "label", "pred", "correct", "confidence", "second", "second_conf", "sec_per_ticket", "input_tokens"],
        )
        w.writeheader()
        for r in rows:
            w.writerow({
                "text": r["text"],
                "label": r["label"],
                "pred": r["pred"],
                "correct": int(r["pred"] == r["label"]),
                "confidence": f"{r['confidence']:.6f}",
                "second": r.get("second") or "",
                "second_conf": f"{r['second_conf']:.6f}" if r.get("second_conf") is not None else "",
                "sec_per_ticket": f"{r['sec_per_ticket']:.4f}",
                "input_tokens": r.get("input_tokens", ""),
            })
    print(f"wrote {csv_path}")

    s = report
    print(f"n={s['n']}  accuracy={s['accuracy']:.4f}  mean_conf={s['mean_conf']:.4f}")
    print(f"ECE (10 equal bins)={s['ece_equal_10']:.4f}  ECE (focus bins)={s['ece_focus']:.4f}")
    if "sec_per_ticket" in s:
        t = s["sec_per_ticket"]
        print(f"CPU s/ticket  median={t['median']:.3f}  mean={t['mean']:.3f}  p90={t['p90']:.3f}")
    a = s["at_0.90"]
    print(f">=0.90 conf: {a['n']}/{s['n']} tickets ({a['coverage']:.1%}), accuracy {a['accuracy']:.1%} "
          f"(Wilson 95% {a['wilson95_lo']:.1%}–{a['wilson95_hi']:.1%})")
    print("reliability bins (said vs right):")
    for b in s["bins"]:
        print(f"  {b['bin']:>11}  n={b['n']:4d}  said {b['mean_conf']:.3f}  right {b['accuracy']:.3f}  "
              f"gap {b['gap']:+.3f}")
    print("auto-route (point estimate on the full test set — optimistic; see recalibrate.py):")
    for r in s["route"]:
        print(f"  conf>={r['threshold']:.2f}  cover {r['coverage']:6.1%}  acc {r['accuracy']:6.1%}  "
              f"n={r['n']:4d}  Wilson lo {r['wilson95_lo']:.1%}")
    print("lowest cutoff for target precision (full set):")
    for key, block in s["auto_route_point"].items():
        if block is None:
            print(f"  {key}: never reached")
        else:
            print(f"  {key}: t>={block['threshold']:.3f}  cover {block['coverage']:.1%}  "
                  f"acc {block['accuracy']:.1%}  n={block['n']}")
    print("confident and wrong (conf>=0.95):", s["wrong_at_95_pairs"] or "none")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
