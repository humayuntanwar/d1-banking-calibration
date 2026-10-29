#!/usr/bin/env python3
"""Zero-shot BANKING77 intent classification with Liquid AI d1-2B.

Each ticket is one `choice` question over all 77 intents. d1-2B returns the
pick and a calibrated probability in one forward pass (no generated tokens).

Usage:
    python run_eval.py                  # full 3,080 tickets, resume-safe
    python run_eval.py --limit 20       # smoke test
    python run_eval.py --batch 4        # throughput (sec_per_ticket = batch/n)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def hardware_blob(device: str, dtype: str, model_id: str) -> dict:
    cpu_model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    mem_kb = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_kb = int(line.split()[1])
                break
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "dtype": dtype,
        "cuda": bool(torch.cuda.is_available()),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "mem_gb": round((mem_kb or 0) / (1024 * 1024), 2),
        "model": model_id,
        "started_unix": time.time(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def intent_question(intents: list[str]) -> dict:
    """One 77-way choice question. Criteria text is the label, spaces for underscores."""
    return {
        "intent": {
            "type": "choice",
            "instructions": "Which banking request is the customer making?",
            "criteria": {name: name.replace("_", " ").replace("?", "").strip() for name in intents},
        }
    }


def load_rows(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows or "text" not in rows[0] or "category" not in rows[0]:
        raise SystemExit(f"{path} must have columns text,category")
    return rows


def already_done(out_path: Path) -> int:
    if not out_path.exists():
        return 0
    n = 0
    with out_path.open() as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def record(r: dict, answer: dict, usage: dict, dt: float) -> dict:
    probs = {k: round(float(v), 6) for k, v in answer["probabilities"].items()}
    ranked = sorted(probs, key=probs.get, reverse=True)
    second = ranked[1] if len(ranked) > 1 else None
    return {
        "text": r["text"],
        "label": r["category"],
        "pred": answer["choice"],
        "confidence": float(answer["confidence"]),
        "second": second,
        "second_conf": float(probs[second]) if second else None,
        "sec_per_ticket": dt,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "probs": probs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="LiquidAI/d1-2B")
    p.add_argument("--revision", default="cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41",
                   help="Hub commit. Empty string = latest main.")
    p.add_argument("--data", default=str(ROOT / "data" / "banking77_test.csv"))
    p.add_argument("--out", default=str(ROOT / "results" / "predictions.jsonl"))
    p.add_argument("--limit", type=int, default=0, help="Score only the first N tickets (smoke test).")
    p.add_argument("--batch", type=int, default=1, help="Tickets per system_one_batch call. 1 = sequential latency.")
    p.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="bfloat16",
                   help="Checkpoint is bf16. Model card suggests float32 on CPU (~2s/ticket here); "
                        "bf16 is ~2x faster and is what this experiment reports.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    rows = load_rows(Path(args.data))
    intents = sorted({r["category"] for r in rows})
    if args.limit:
        rows = rows[: args.limit]
    question = intent_question(intents)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0 if args.overwrite else already_done(out_path)
    if done >= len(rows) and not args.overwrite:
        print(f"{out_path} already has {done} tickets (>= {len(rows)}); skipping inference. Pass --overwrite to rerun.")
        return
    if done:
        print(f"resuming {out_path}: {done} tickets already written", flush=True)
        rows = rows[done:]

    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
    threads = os.cpu_count() or 1
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    revision = args.revision or None
    hw = hardware_blob(device, str(dtype).replace("torch.", ""), args.model)
    hw["revision"] = revision
    hw["batch"] = args.batch
    hw_path = out_path.with_name("hardware.json")
    hw_path.write_text(json.dumps(hw, indent=2) + "\n")
    print(f"loading {args.model} rev={revision} on {device} {dtype} ({threads} threads)", flush=True)
    t_load = time.time()
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype, revision=revision)
    model.to(device)
    model.eval()
    print(f"loaded in {time.time() - t_load:.1f}s", flush=True)

    mode = "w" if args.overwrite or done == 0 else "a"
    n_total = done + len(rows)
    t0 = time.time()
    scored = done
    with out_path.open(mode) as f:
        for i in range(0, len(rows), args.batch):
            chunk = rows[i : i + args.batch]
            t = time.time()
            outs = model.system_one_batch([(r["text"], question) for r in chunk])
            elapsed = time.time() - t
            dt = elapsed / len(chunk)
            for r, o in zip(chunk, outs):
                rec = record(r, o["answers"]["intent"], o.get("usage") or {}, dt)
                f.write(json.dumps(rec) + "\n")
            f.flush()
            scored += len(chunk)
            if scored == done + len(chunk) or scored % 20 == 0 or scored == n_total:
                rate = (scored - done) / max(time.time() - t0, 1e-6)
                eta = (n_total - scored) / rate if rate else 0
                print(
                    f"{scored}/{n_total}  {dt:.3f}s/ticket  "
                    f"pred={outs[0]['answers']['intent']['choice']!r} "
                    f"p={outs[0]['answers']['intent']['confidence']:.3f}  "
                    f"ETA {eta / 60:.1f} min",
                    flush=True,
                )

    wall = time.time() - t0
    n_this = max(len(rows), 1)
    print(f"wrote {out_path}  this-session {wall:.0f}s ({wall / n_this:.3f}s/ticket, {n_this} tickets)")
    try:
        hw = json.loads(hw_path.read_text())
        hw["finished_unix"] = time.time()
        hw["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        hw["session_seconds"] = round(wall, 1)
        hw["n_tickets_this_session"] = n_this
        hw_path.write_text(json.dumps(hw, indent=2) + "\n")
    except OSError:
        pass


if __name__ == "__main__":
    main()
