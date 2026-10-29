#!/usr/bin/env bash
# One-command reproduction of the BANKING77 d1-2B calibration experiment.
# Full 3,080 tickets on CPU is ~30–60 minutes on 4 vCPU; use --limit 20 to smoke-test.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
mkdir -p results

python -m unittest test_metrics.py
python run_eval.py "$@"
python score.py
python recalibrate.py
python plot.py

echo
echo "Done. Headline metrics:"
python - <<'PY'
import json
from pathlib import Path
s = json.loads(Path("results/metrics.json").read_text())
a = s["at_0.90"]
t = s.get("sec_per_ticket", {})
print(f"  accuracy {s['accuracy']*100:.1f}%  ECE {s['ece_equal_10']*100:.1f} pp")
print(f"  conf>=0.90: {a['accuracy']*100:.1f}% right on {a['n']} tickets ({a['coverage']*100:.1f}%)")
if t:
    print(f"  median {t['median']:.3f}s/ticket")
print("  chart: results/reliability.png")
PY
