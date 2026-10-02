#!/usr/bin/env bash
# Full agent-eval schedule, sequential (one Ollama request at a time).
# Run from repo root:  bash eval_sop/agent_eval/run_all.sh
set -u
PY="${PY:-python}"
M="${MODEL:-qwen2.5:7b}"
OUT=eval_sop/agent_eval/results
export PYTHONIOENCODING=utf-8
# 1) main: fixed code, all 56 questions, 3 seeds
"$PY" -m eval_sop.agent_eval.run_agent_eval --model "$M" --variant after  --ablation none --seeds 0 1 2 --out $OUT/main_after.jsonl
# 2) before-fix: original prompt + original stockout velocity, fix-relevant questions, 3 seeds
"$PY" -m eval_sop.agent_eval.run_agent_eval --model "$M" --variant before --ablation none --seeds 0 1 2 --fix-relevant-only --out $OUT/before.jsonl
# 3) ablations on conflict + multihop questions, seed 0 first, then seeds 1 2
for S in 0 1 2; do
  for AB in reconcile subset truncate; do
    "$PY" -m eval_sop.agent_eval.run_agent_eval --model "$M" --variant after --ablation $AB --seeds $S --categories conflict multihop --out $OUT/ablation_$AB.jsonl
  done
done
