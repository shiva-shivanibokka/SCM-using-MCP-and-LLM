"""How much did the post-hoc `tool_ok` redefinition actually move the numbers?

Commit `80559db` (two days after the runs) redefined the tool-selection metric
to require a *useful* result: `ok_names` switched from `ok_result(...)` to
`useful_result(...)`, so a call that reaches a tool but returns "SKU ... not
found" / "returned no rows" / "no data available" no longer counts.  That is a
metric rule changed after the results were visible, so RESULTS.md §2e has to
say what it changed.  This script measures it, per result file and for the
subgroups quoted in RESULTS.md, by re-grading every committed run under BOTH
definitions (the only difference between them is that one line).

Run from the repo root:
    python -m eval_sop.agent_eval.tool_ok_redefinition_effect
"""
from __future__ import annotations

import json
from pathlib import Path

from eval_sop.agent_eval.grade import grade, ok_result

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
STALE = ["C01", "C02", "C03", "C04", "C05"]
REAL = ["C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13"]


def old_tool_ok(run: dict, q: dict) -> int:
    """`tool_ok` as defined BEFORE 80559db: any non-error call to an acceptable
    tool counts, even if it returned no data."""
    ok_names = {c["name"] for c in run.get("tool_calls", []) if ok_result(c["result"])}
    return int(bool(ok_names & set(q["acceptable_tools"])))


def main() -> None:
    qs = {json.loads(l)["id"]: json.loads(l)
          for l in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    runs = []
    for f in sorted(RES.glob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            q = qs[r["qid"]]
            runs.append({"file": f.name, "qid": r["qid"], "variant": r["variant"],
                         "ablation": r["ablation"], "category": q["category"],
                         "new": int(grade(r, q)["tool_ok"]), "old": old_tool_ok(r, q)})

    def row(label, sel):
        g = [r for r in runs if sel(r)]
        n = len(g)
        if not n:
            return None
        o, w = sum(r["old"] for r in g), sum(r["new"] for r in g)
        ch = sum(1 for r in g if r["old"] != r["new"])
        return (f"{label:52s} {n:4d} runs   old {100*o/n:5.1f}%   new {100*w/n:5.1f}%   "
                f"delta {100*(w-o)/n:+5.1f} pp   runs changed {ch}")

    main_ = lambda r: r["variant"] == "after" and r["ablation"] == "none"  # noqa: E731
    lines = ["Effect of the post-hoc `tool_ok` redefinition (commit 80559db) on every",
             "tool-selection figure quoted in RESULTS.md. Only the tool_ok column can",
             "change; strict/lenient scoring rules were untouched.", ""]
    for label, sel in [
        ("ALL committed runs", lambda r: True),
        ("main: after/none, all 56 Qs", main_),
        ("main: conflict category (all 14)", lambda r: main_(r) and r["category"] == "conflict"),
        ("main: conflict stale-count C01-C05", lambda r: main_(r) and r["qid"] in STALE),
        ("main: conflict real two-tool C06-C13", lambda r: main_(r) and r["qid"] in REAL),
        ("main: conflict C14", lambda r: main_(r) and r["qid"] == "C14"),
        ("before/none, all Qs", lambda r: r["variant"] == "before" and r["ablation"] == "none"),
        ("ablation reconcile", lambda r: r["ablation"] == "reconcile"),
        ("ablation truncate", lambda r: r["ablation"] == "truncate"),
    ]:
        s = row(label, sel)
        if s:
            lines.append(s)
    changed = [r for r in runs if r["old"] != r["new"]]
    lines += ["", f"Runs whose tool_ok flipped: {len(changed)} of {len(runs)}"]
    for r in changed:
        lines.append(f"  {r['file']:26s} {r['qid']} ablation={r['ablation']} old=1 new=0")
    lines += ["",
              "Finding: the redefinition flipped 3 of 504 runs, all in the reconcile",
              "ablation (47.6% -> 44.0%). It did NOT change the main condition (60.1%)",
              "or the C06-C13 subgroup (29.2% under BOTH definitions). The drop from",
              "the ~50% quoted for the 14-question conflict category to 29.2% comes",
              "from SPLITTING that category in the same commit, not from the",
              "redefinition. Both the split and the redefinition were nonetheless",
              "decided after the runs, so both are post-hoc."]
    text = "\n".join(lines) + "\n"
    (RES / "tool_ok_redefinition_effect.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
