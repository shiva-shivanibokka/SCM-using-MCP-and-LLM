#!/usr/bin/env bash
# Run tests/test_stockout_velocity.py against the BASE commit's
# intelligence/stockout.py (copied into a scratch dir) and against this branch.
# Run from repo root:  PY=<python> bash eval_sop/repro/run_pytest_base_vs_fixed.sh
set -u
PY="${PY:-python}"
T="${REPRO_TMP:-$(cd ../.. && pwd)/repro_tmp}/pytest_base"
rm -rf "$T"; mkdir -p "$T/intelligence"; touch "$T/intelligence/__init__.py"
git show 65c1c06:intelligence/stockout.py > "$T/intelligence/stockout.py"
cp tests/test_stockout_velocity.py "$T/"
echo "### pytest tests/test_stockout_velocity.py against BASE 65c1c06:intelligence/stockout.py"
(cd "$T" && "$PY" -m pytest -q -p no:cacheprovider test_stockout_velocity.py 2>&1 | grep -E "passed|failed|Error|assert" | tail -8)
echo
echo "### pytest against FIXED (this branch)"
"$PY" -m pytest -q -p no:cacheprovider tests/test_stockout_velocity.py tests/test_intelligence.py 2>&1 | tail -2
