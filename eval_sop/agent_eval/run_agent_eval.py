"""Run the repo's own ReAct agent (agent.agent.run_agent_with_steps, unmodified
loop: up to MAX_ITERATIONS=20 LLM turns, all MCP tools, in-process dispatch)
against the question bank, using a LOCAL Ollama model (free).

Only two seams are patched, both in-process and logged:
  * agent.agent._call_llm   -> Ollama /api/chat (native API so num_ctx/seed/
                               temperature can be set). Messages stay in the
                               agent's OpenAI/Groq format; this shim converts.
  * agent.agent._run_mcp_tool -> records every tool call + full result; stubs
                               web_search (no network in this eval); applies
                               the tool-subset / truncation ablations.

Variants:
  --variant before : original stale system prompt + original single-day
                     stockout velocity (reconstructed by reversing the fixes).
  --variant after  : the fixed code on this branch.
Ablations (applied on top of the variant): none | subset | truncate | reconcile

Usage (repo root):
  python -m eval_sop.agent_eval.run_agent_eval --model qwen2.5:7b \
      --variant after --ablation none --seeds 0 1 2 --out eval_sop/agent_eval/results/main.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Offline / no-remote-DB environment (set BEFORE importing project modules).
os.environ["DATABASE_URL"] = ""
os.environ["BYPASS_MCP_HTTP"] = "true"
os.environ["SERPAPI_KEY"] = ""
os.environ["GROQ_API_KEY"] = "unused-local-ollama"  # agent requires a non-empty key

import httpx  # noqa: E402

OLLAMA = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
QFILE = Path(__file__).resolve().parent / "questions.jsonl"
SUFFIX = ("\n\nEnd your reply with a final line of the form `FINAL: <answer>` "
          "(just the answer values, separated by '; ' if there are several).")

# Reverse of the stale-count fix applied to agent/agent.py on this branch.
PROMPT_FIX = [
    ("India's premier omnichannel pet supply company with 67 stores across India, 65 SKUs, 5000 customers,",
     "India's premier omnichannel pet supply company with 92 stores across India, 160 SKUs, 25,000 customers,"),
    ("You have access to 50 powerful tools.", "You have access to 54 powerful tools."),
    ("67-store demand vs national avg", "Store-level demand vs national avg"),
    ("  47,515 rows | 65 SKUs | All prices in ₹INR", "  175,360 rows | 160 SKUs | 2023-01-01 to 2025-12-31 | All prices in ₹INR"),
    ("huft_products.csv         — 65 SKUs", "huft_products.csv         — 160 SKUs"),
    ("huft_stores.csv           — 67 stores:", "huft_stores.csv           — 92 stores:"),
    ("huft_customers.csv        — 5,000 customers:", "huft_customers.csv        — 25,000 customers:"),
    ("huft_sales_transactions.csv — 50,000 transactions:", "huft_sales_transactions.csv — 326,883 transactions:"),
    ("huft_returns.csv          — 1,500 returns", "huft_returns.csv          — 9,806 returns"),
    ("huft_supplier_performance.csv — 624 monthly supplier reviews:", "huft_supplier_performance.csv — 936 monthly supplier reviews:"),
]
TOOLDESC_FIX = [("the Pet Store's 5000 customers", "the Pet Store's 25,000 customers"),
                ("the Pet Store's 67 stores", "the Pet Store's 92 stores")]

RECONCILE = """

━━━ RECONCILING CONFLICTING TOOL OUTPUTS ━━━
Different tools compute similar metrics from different tables, windows and formulas
(e.g. single-day vs 30-day velocity, store-level vs SKU-level inventory, all-years vs
one-year averages). Before answering:
  1. If two tool outputs (or this prompt) give different values for the same quantity,
     say so explicitly and name both values and their sources.
  2. Prefer the value whose definition matches the question (window, table, formula);
     if unsure, compute it directly with run_sql_query or python_repl.
  3. Never commit to the first number you saw without checking it against the others.
"""

# ~10-tool consolidated subset for ablation (a).
SUBSET = ["run_sql_query", "python_repl", "get_sku_360", "get_stockout_risk",
          "get_supplier_lead_time_tracker", "get_store_inventory_breakdown",
          "get_return_rate_analysis", "get_cold_chain_monitor",
          "get_channel_revenue_attribution", "get_brand_performance"]
TRUNC_CHARS = 4000

# "compact" tool set for an 8192-token context budget: the full system prompt
# (~4.3k llama3.1 tokens) + all 54 tool schemas (~7.9k) cannot fit in 8192, so
# the constrained configuration offers 10 tools (~1.7k tokens), including the
# buggy-before-fix get_stockout_prediction so the before/after test is live.
COMPACT = ["run_sql_query", "python_repl", "get_sku_360", "get_stockout_prediction",
           "get_stockout_risk", "get_supplier_lead_time_tracker", "get_return_rate_analysis",
           "get_cold_chain_monitor", "get_channel_revenue_attribution", "get_brand_performance"]


class ContextBudgetExceeded(RuntimeError):
    pass


def est_tokens(system: str, messages: list, tools: list) -> int:
    # Calibrated on llama3.1:8b prompt_eval_count: system prompt 19,385 chars ->
    # 4,330 tok (4.48 c/t); 10 tool schemas 6,790 chars -> 1,718 tok (3.95 c/t).
    # Conversation text is counted conservatively at 3.3 chars/token.
    conv = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    return int(len(system) / 4.4 + len(json.dumps(tools)) / 3.9 + conv / 3.3) + 30


class _Msg:
    def __init__(self, d):
        self._d = d

    def model_dump(self, **_):
        return json.loads(json.dumps(self._d))


class _Choice:
    def __init__(self, d):
        self.message = _Msg(d)


class _Raw:
    def __init__(self, d):
        self.choices = [_Choice(d)]


def to_ollama(messages, system_prompt):
    out = [{"role": "system", "content": system_prompt}]
    id2name = {}
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            tcs = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                id2name[tc.get("id")] = fn.get("name")
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                tcs.append({"function": {"name": fn.get("name"), "arguments": args or {}}})
            mm = {"role": "assistant", "content": m.get("content") or ""}
            if tcs:
                mm["tool_calls"] = tcs
            out.append(mm)
        elif role == "tool":
            out.append({"role": "tool", "content": str(m.get("content", "")),
                        "tool_name": id2name.get(m.get("tool_call_id"), "")})
        else:
            c = m.get("content")
            out.append({"role": role or "user", "content": c if isinstance(c, str) else json.dumps(c)})
    return out


class Runner:
    def __init__(self, args):
        self.args = args
        import agent.agent as A
        import mcp_server.server as S
        import intelligence.stockout as ST
        self.A, self.S, self.ST = A, S, ST
        self.fixed_prompt = A.SYSTEM_PROMPT
        self._orig_predict = ST.predict_stockouts
        self._orig_tools = [dict(t) for t in S.MCP_TOOLS]
        self.calls = []
        self.llm = []

    # -------------------------------------------------------------- variants
    def configure(self, variant, ablation):
        A, S, ST = self.A, self.S, self.ST
        prompt = self.fixed_prompt
        tools = [dict(t) for t in self._orig_tools]
        if variant == "before":
            for old, new in PROMPT_FIX:
                assert prompt.count(new) == 1, new
                prompt = prompt.replace(new, old)
            for t in tools:
                for old, new in TOOLDESC_FIX:
                    t["description"] = t["description"].replace(new, old)
            orig = self._orig_predict
            ST.predict_stockouts = lambda inv, safety_stock_days=7, risk_filter=None, **kw: orig(
                inv, safety_stock_days=safety_stock_days, risk_filter=risk_filter, velocity_window_days=1)
        else:
            ST.predict_stockouts = self._orig_predict
        if ablation == "reconcile":
            prompt = prompt + RECONCILE
        if ablation == "subset":
            tools = [t for t in tools if t["name"] in SUBSET]
        if self.args.tools == "compact":
            tools = [t for t in tools if t["name"] in COMPACT]
        A.SYSTEM_PROMPT = prompt
        self.tools = tools
        self.allowed = {t["name"] for t in tools}
        S.invalidate_tool_cache()

        import agent.mcp_client as MC

        async def _list_tools():
            return self.tools
        MC.list_tools = _list_tools
        A._call_llm = self._call_llm
        A._run_mcp_tool = self._run_tool
        self.ablation = ablation

    # -------------------------------------------------------------- seams
    async def _call_llm(self, messages, mcp_tools, provider, model, api_key):
        a = self.args
        body = {"model": a.model, "messages": to_ollama(messages, self.A.SYSTEM_PROMPT),
                "stream": False, "keep_alive": a.keep_alive,
                "options": {"num_ctx": a.num_ctx, "temperature": a.temperature,
                            "seed": self.seed, "num_predict": a.num_predict}}
        if mcp_tools:
            body["tools"] = self.A._mcp_tools_to_openai(mcp_tools)
        est_prompt_chars = len(json.dumps(body["messages"])) + len(json.dumps(body.get("tools", [])))
        est_tok = est_tokens(self.A.SYSTEM_PROMPT, body["messages"][1:], body.get("tools", []))
        if a.max_prompt_tokens and est_tok > a.max_prompt_tokens:
            self.budget_exceeded = est_tok
            self.llm.append({"t": 0, "prompt_eval_count": None, "eval_count": None,
                             "est_prompt_tokens": est_tok, "done_reason": "context_budget_exceeded",
                             "error": None})
            raise ContextBudgetExceeded(f"estimated prompt {est_tok} tok > budget {a.max_prompt_tokens}")
        t0 = time.time()
        err = None
        j = {}
        for attempt in range(6):  # shared local server: retry patiently
            try:
                async with httpx.AsyncClient(timeout=a.call_timeout) as cli:
                    r = await cli.post(f"{OLLAMA}/api/chat", json=body)
                j = r.json()
                if r.status_code != 200 or "error" in j:
                    err = f"HTTP {r.status_code}: {str(j.get('error'))[:300]}"
                    await asyncio.sleep(20 * (attempt + 1))
                    continue
                err = None
                break
            except Exception as e:  # timeout / connection
                err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(20 * (attempt + 1))
        dt = time.time() - t0
        msg = j.get("message", {}) if not err else {}
        self.llm.append({"t": round(dt, 1), "prompt_eval_count": j.get("prompt_eval_count"),
                         "eval_count": j.get("eval_count"), "est_prompt_tokens": est_tok, "est_prompt_chars": est_prompt_chars,
                         "done_reason": j.get("done_reason"), "error": err})
        if err:
            self.infra_error = err
            raise RuntimeError(f"LLM call failed: {err}")
        tcs = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            tcs.append({"id": f"call_{len(self.llm)}_{i}", "name": fn.get("name"), "input": args})
        text = msg.get("content") or None
        oai = {"role": "assistant", "content": text or ""}
        if tcs:
            oai["tool_calls"] = [{"id": t["id"], "type": "function",
                                  "function": {"name": t["name"], "arguments": json.dumps(t["input"])}}
                                 for t in tcs]
        return {"stop_reason": "tool_use" if tcs else "stop", "text": text,
                "tool_calls": tcs, "raw": _Raw(oai)}

    async def _run_tool(self, name, arguments):
        t0 = time.time()
        if name not in self.allowed:
            res = f"Unknown tool '{name}'. It is not available in this deployment."
        elif name == "web_search":
            res = "web_search is disabled in this offline evaluation (no network access)."
        else:
            from agent.mcp_client import call_tool
            res = await call_tool(name, arguments if isinstance(arguments, dict) else {})
        full_len = len(res)
        lim = TRUNC_CHARS if self.ablation == "truncate" else self.args.trunc_chars
        if lim and len(res) > lim:
            res = res[:lim] + f"\n...[truncated {full_len - lim} chars]"
        self.calls.append({"name": name, "input": arguments, "result": res,
                           "full_len": full_len, "t": round(time.time() - t0, 2)})
        return res

    # -------------------------------------------------------------- run one
    async def run_one(self, q, seed):
        self.seed = seed
        self.calls, self.llm = [], []
        self.infra_error = None
        self.budget_exceeded = None
        steps, answer, error = [], None, None
        t0 = time.time()
        try:
            async for ev in self.A.run_agent_with_steps(q["question"] + SUFFIX, provider="groq",
                                                        model=self.args.model):
                if ev["type"] == "answer":
                    answer = ev["text"]
                elif ev["type"] == "error":
                    error = ev["text"]
                elif ev["type"] == "thinking":
                    steps.append(ev["text"])
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        return {"answer": answer, "error": error, "infra_error": self.infra_error,
                "context_budget_exceeded": self.budget_exceeded, "thinking": steps, "tool_calls": self.calls,
                "llm_calls": self.llm, "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--variant", choices=["before", "after"], default="after")
    ap.add_argument("--ablation", choices=["none", "subset", "truncate", "reconcile"], default="none")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ids", nargs="*", default=None, help="question ids (default all)")
    ap.add_argument("--fix-relevant-only", action="store_true")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--num-ctx", dest="num_ctx", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--call-timeout", dest="call_timeout", type=float, default=1200)
    ap.add_argument("--tools", choices=["all", "compact"], default="all")
    ap.add_argument("--trunc-chars", dest="trunc_chars", type=int, default=0,
                    help="truncate every tool result to N chars (0 = full output)")
    ap.add_argument("--max-prompt-tokens", dest="max_prompt_tokens", type=int, default=0,
                    help="end a run (counted as failed) if the estimated prompt exceeds this")
    ap.add_argument("--num-predict", dest="num_predict", type=int, default=4096)
    ap.add_argument("--keep-alive", dest="keep_alive", default="30m")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qs = [json.loads(l) for l in QFILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids:
        qs = [q for q in qs if q["id"] in set(args.ids)]
    if args.fix_relevant_only:
        qs = [q for q in qs if q["fix_relevant"]]
    if args.categories:
        qs = [q for q in qs if q["category"] in set(args.categories)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if not r.get("infra_error"):  # infra failures get re-run on resume
                    done.add((r["variant"], r["ablation"], r["seed"], r["qid"]))

    runner = Runner(args)
    runner.configure(args.variant, args.ablation)
    ver = digest = None
    for _ in range(10):
        try:
            ver = httpx.get(f"{OLLAMA}/api/version", timeout=120).json().get("version")
            digest = next((m.get("digest") for m in httpx.get(f"{OLLAMA}/api/tags", timeout=120).json()["models"]
                           if m["name"] == args.model), None)
            break
        except Exception:
            time.sleep(30)
    for seed in args.seeds:
        for q in qs:
            key = (args.variant, args.ablation, seed, q["id"])
            if key in done:
                continue
            res = asyncio.run(runner.run_one(q, seed))
            rec = {"qid": q["id"], "category": q["category"], "variant": args.variant,
                   "ablation": args.ablation, "seed": seed, "model": args.model,
                   "model_digest": digest, "ollama_version": ver, "num_ctx": args.num_ctx,
                   "temperature": args.temperature, "n_tools_offered": len(runner.tools),
                   "tools_config": args.tools,
                   # effective per-result truncation actually applied this run:
                   # the "truncate" ablation forces TRUNC_CHARS regardless of --trunc-chars.
                   "trunc_chars": (TRUNC_CHARS if args.ablation == "truncate" else args.trunc_chars),
                   "trunc_chars_cli": args.trunc_chars,
                   "max_prompt_tokens": args.max_prompt_tokens, "num_predict": args.num_predict,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **res}
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            print(f"[{rec['ts']}] {args.variant}/{args.ablation} seed={seed} {q['id']} "
                  f"tools={len(res['tool_calls'])} llm={len(res['llm_calls'])} {res['wall_s']}s "
                  f"err={bool(res['error'])}", flush=True)


if __name__ == "__main__":
    main()
