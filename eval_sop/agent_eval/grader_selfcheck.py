"""Sanity checks for grade.py on hand-written answers (no LLM).
  python -m eval_sop.agent_eval.grader_selfcheck"""
import json
from pathlib import Path

from eval_sop.agent_eval.grade import grade

Q = {json.loads(l)["id"]: json.loads(l) for l in (Path(__file__).parent / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
CASES = [  # (qid, answer, expected strict, expected distractor_final)
    ("L01", "The price is ₹720.\n\nFINAL: 720", 1, None),
    ("C06", "SKU EXT_059 has 4.8 days of inventory cover.\n\nFINAL: 4.8 days", 1, 0),
    ("C06", "Velocity 293/day -> 1.2 days to zero.\nFINAL: 1.2 days", 0, 1),
    ("C01", "FINAL: 67", 0, 1),
    ("C01", "FINAL: 92 stores", 1, 0),
    ("A01", "FINAL: ₹22.67 crore", 1, None),
    ("A01", "FINAL: 226,690,575", 1, None),
    ("A01", "FINAL: ₹22,66,90,575", 1, None),  # Indian digit grouping (commas ignored)
    ("A01", "FINAL: ₹2.27 lakh", 0, None),     # wrong magnitude must fail
    ("C09", "No SKUs are critical.\nFINAL: 0", 1, 0),
    ("C09", "FINAL: 3 critical SKUs", 0, 1),
    ("C10", "FINAL: GIFT_002", 1, 0),
    ("C10", "FINAL: EXT_059", 0, 1),
    ("C11", "FINAL: NO; reorder point 316 units", 1, 0),
    ("C11", "FINAL: YES; 2,582", 0, 1),
    ("A02", "FINAL: Online; 50.2%", 1, None),
    ("M01", "FINAL: HUFT Home; 1.19%", 1, None),
    ("L02", "FINAL: Sara’s Kitchen (HUFT)", 1, None),
    ("L04", "no final line, Mumbai", 0, None),
]
bad = 0
for qid, ans, exp, expd in CASES:
    g = grade({"answer": ans, "variant": "after", "ablation": "none", "seed": 0, "tool_calls": [], "llm_calls": []}, Q[qid])
    ok = g["strict"] == exp and (expd is None or g["distractor_final"] == expd)
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} {qid} strict={g['strict']} distractor={g['distractor_final']} :: {ans[-40:]!r}")
print(f"{len(CASES) - bad}/{len(CASES)} grader self-checks pass")
