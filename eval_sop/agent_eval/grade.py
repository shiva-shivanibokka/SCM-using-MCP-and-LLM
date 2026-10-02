"""Rule-based grader for agent-eval runs (deterministic, no LLM judge).

strict  : every answer part must appear in the `FINAL:` line.
lenient : every answer part must appear anywhere in the final answer text.
Numbers match within the question's absolute tolerance; Indian/western digit
grouping, ₹, %, and crore/lakh/million/k suffixes are understood.
Entities match case-insensitively on word boundaries against aliases.
"""
from __future__ import annotations

import re

FAIL_PREFIXES = ("TOOL_ERROR", "MCP tool call failed", "SecurityError", "Execution error",
                 "Unknown tool", "SQL error", "Only a single read-only", "SyntaxError",
                 "MySQL connection failed", "PostgreSQL connection failed", "web_search is disabled")
DB_TOOLS = {"query_mysql", "query_postgres", "test_mysql_connection", "test_postgres_connection",
            "log_forecast_to_postgres", "create_inventory_alert", "get_active_alerts", "get_monthly_kpis"}
FLAG_RE = re.compile(r"discrepan|conflict|inconsisten|mismatch|disagree|contradict|differ(?:s|ent|ence)? (?:from|between|across)", re.I)
MULT = {"crore": 1e7, "cr": 1e7, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "l": 1e5, "million": 1e6,
        "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "k": 1e3, "thousand": 1e3}
NUM_RE = re.compile(r"(?<![A-Za-z_])(-?\d[\d,]*(?:\.\d+)?)\s*(crore|cr|lakhs|lakh|lac|million|mn|billion|bn|thousand|k|m|l)?(?![A-Za-z_])", re.I)
WORDNUM = {"zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def norm(s: str) -> str:
    s = (s or "").replace("’", "'").replace("‘", "'").replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", s).strip()


def final_line(answer: str) -> str | None:
    ms = re.findall(r"FINAL\s*:\s*(.+)", norm_keep_lines(answer), flags=re.I)
    return norm(ms[-1]) if ms else None


def norm_keep_lines(s: str) -> str:
    return (s or "").replace("**", "").replace("`", "")


def numbers(text: str) -> list[float]:
    out = []
    for m in NUM_RE.finditer(text or ""):
        raw, suf = m.group(1), (m.group(2) or "").lower()
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        out.append(v)
        if suf in MULT:
            out.append(v * MULT[suf])
    low = (text or "").lower()
    for w, v in WORDNUM.items():
        if re.search(rf"\b{w}\b", low):
            out.append(float(v))
    return out


def ent_match(text: str, aliases: list[str]) -> bool:
    t = norm(text).lower()
    for a in aliases:
        a = norm(a).lower()
        if not a:
            continue
        pat = re.escape(a)
        if a[0].isalnum():
            pat = r"(?<![a-z0-9_])" + pat
        if a[-1].isalnum():
            pat = pat + r"(?![a-z0-9_])"
        if re.search(pat, t):
            return True
    return False


def part_match(text: str, part: dict) -> bool:
    if text is None:
        return False
    if part["kind"] == "number":
        return any(abs(v - part["value"]) <= part["tol"] for v in numbers(text))
    v = part["value"].upper()
    if v in ("YES", "NO"):
        y, n = ent_match(text, ["yes"]), ent_match(text, ["no", "not below"])
        return (y and not n) if v == "YES" else (n and not y)
    return ent_match(text, part["aliases"])


def distractor_hit(text: str, q: dict) -> bool:
    if text is None or not q.get("distractors"):
        return False
    p0 = q["parts"][0] if q["parts"] else None
    for d in q["distractors"]:
        if isinstance(d, (int, float)):
            tol = p0["tol"] if p0 and p0["kind"] == "number" else 0.05
            # tolerance for the distractor never wider than half the truth-distractor gap
            gap = abs(d - p0["value"]) / 2 if p0 and p0["kind"] == "number" else tol
            if any(abs(v - d) <= min(tol, gap) for v in numbers(text)):
                return True
        else:
            if str(d).upper() in ("YES", "NO"):
                if part_match(text, {"kind": "entity", "value": str(d), "aliases": [str(d)]}):
                    return True
            elif ent_match(text, [str(d)]):
                return True
    return False


def ok_result(r: str) -> bool:
    return not str(r).lstrip().startswith(FAIL_PREFIXES)


def grade(run: dict, q: dict) -> dict:
    ans = run.get("answer") or ""
    fin = final_line(ans)
    strict = fin is not None and all(part_match(fin, p) for p in q["parts"])
    lenient = bool(ans) and all(part_match(ans, p) for p in q["parts"])
    calls = run.get("tool_calls", [])
    names = [c["name"] for c in calls]
    ok_names = {c["name"] for c in calls if ok_result(c["result"])}
    acc_tools = set(q["acceptable_tools"])
    llm = run.get("llm_calls", [])
    srcs = q.get("distractor_sources", [])
    exposed = ("SYSTEM_PROMPT" in srcs and run["variant"] == "before") or any(s in names for s in srcs)
    return {
        "qid": q["id"], "category": q["category"], "variant": run["variant"], "ablation": run["ablation"],
        "seed": run["seed"], "fix_relevant": q["fix_relevant"],
        "strict": int(strict), "lenient": int(lenient), "has_final": int(fin is not None),
        "error": int(bool(run.get("error"))), "infra_error": int(bool(run.get("infra_error"))),
        "budget_exceeded": int(bool(run.get("context_budget_exceeded"))),
        "tool_ok": int(bool(ok_names & acc_tools)),
        "n_tool_calls": len(calls), "n_failed_tool_calls": sum(not ok_result(c["result"]) for c in calls),
        "n_db_tool_calls": sum(n in DB_TOOLS for n in names),
        "n_llm_calls": len(llm),
        "hit_max_iter": int(len(llm) >= 20),
        "distractor_final": int(distractor_hit(fin, q)) if q.get("distractors") else None,
        "distractor_answer": int(distractor_hit(ans, q)) if q.get("distractors") else None,
        "flagged_conflict": int(bool(FLAG_RE.search(ans))),
        "exposed": int(exposed) if srcs else None,
        "ollama_prompt_tokens": sum((x.get("prompt_eval_count") or 0) for x in llm),
        "output_tokens": sum((x.get("eval_count") or 0) for x in llm),
        "est_context_tokens": sum((x.get("est_prompt_tokens") or 0) for x in llm),
        "wall_s": run.get("wall_s"),
        "first_tool": names[0] if names else None,
        "final": fin,
    }
