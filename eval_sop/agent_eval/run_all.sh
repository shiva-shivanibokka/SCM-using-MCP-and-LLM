#!/usr/bin/env bash
# Agent-eval schedule, sequential (one Ollama request at a time), run on
# 2026-10-02 under the coordinator's constraints: local llama3.1:8b on the main
# Ollama server, context <= 8192 tokens, OMP_NUM_THREADS=2.
#
# 8192-token constraint: the full system prompt (~4.3k tokens) + all 54 tool
# schemas (~7.9k) do not fit, so every condition here uses the 10-tool
# "compact" set, tool outputs truncated to 1,000 chars, num_predict 1024, and
# a 7,100-token prompt budget (a run that would exceed it ends and is scored
# as a failure). This is NOT the deployed 54-tool configuration.
#
# Run from repo root:  PY=<python> bash eval_sop/agent_eval/run_all.sh
set -u
PY="${PY:-python}"
M="${MODEL:-llama3.1:8b}"
OUT=eval_sop/agent_eval/results
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2
C="--model $M --num-ctx 8192 --num-predict 1024 --max-prompt-tokens 7100 --tools compact --trunc-chars 1000 --call-timeout 1800"
for S in 0 1 2; do
  "$PY" -m eval_sop.agent_eval.run_agent_eval $C --variant after  --ablation none --seeds $S --out $OUT/main_after.jsonl
  "$PY" -m eval_sop.agent_eval.run_agent_eval $C --variant before --ablation none --seeds $S --out $OUT/before.jsonl
done
# Ablations (conflict + multihop questions). "subset" (54 vs ~10 tools) is not
# runnable within 8192 tokens and is skipped. "truncate" here means a LONGER
# per-result limit (4,000 chars) than the 1,000-char base.
for S in 0 1 2; do
  for AB in reconcile truncate; do
    "$PY" -m eval_sop.agent_eval.run_agent_eval $C --variant after --ablation $AB --seeds $S --categories conflict multihop --out $OUT/ablation_$AB.jsonl
  done
done
