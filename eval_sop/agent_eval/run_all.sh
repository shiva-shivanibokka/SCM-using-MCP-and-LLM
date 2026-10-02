#!/usr/bin/env bash
# Full agent-eval schedule, sequential (one Ollama request at a time).
# Ordered so that partial completion still yields paired comparisons:
# seed by seed, main (fixed) then before-fix, then ablations.
# Run from repo root:  PY=<python> bash eval_sop/agent_eval/run_all.sh
set -u
PY="${PY:-python}"
M="${MODEL:-qwen2.5:7b}"
OUT=eval_sop/agent_eval/results
T="--call-timeout 3600"
export PYTHONIOENCODING=utf-8
for S in 0 1 2; do
  # main: fixed code, all 56 questions
  "$PY" -m eval_sop.agent_eval.run_agent_eval --model "$M" $T --variant after  --ablation none --seeds $S --out $OUT/main_after.jsonl
  # before-fix: original prompt + original stockout velocity, fix-relevant questions
  "$PY" -m eval_sop.agent_eval.run_agent_eval --model "$M" $T --variant before --ablation none --seeds $S --fix-relevant-only --out $OUT/before.jsonl
done
# ablations on conflict + multihop questions
for S in 0 1 2; do
  for AB in reconcile subset truncate; do
    "$PY" -m eval_sop.agent_eval.run_agent_eval --model "$M" $T --variant after --ablation $AB --seeds $S --categories conflict multihop --out $OUT/ablation_$AB.jsonl
  done
done
