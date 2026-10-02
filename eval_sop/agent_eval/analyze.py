"""Grade all agent-eval result files and summarise with bootstrap CIs.

  python -m eval_sop.agent_eval.analyze
Writes results/graded.csv, results/summary.json, results/summary.md

CIs: cluster bootstrap over QUESTIONS (all seeds of a resampled question move
together), 10,000 resamples, numpy seed 12345. "± std" is the std of the
metric across the 3 seed-level runs (seed-to-seed variability).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_sop.agent_eval.grade import grade

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
B = 10_000
RNG_SEED = 12345


def load_runs():
    qs = {json.loads(l)["id"]: json.loads(l) for l in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = []
    for f in sorted(RES.glob("*.jsonl")):
        seen = {}
        for l in f.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            key = (r["variant"], r["ablation"], r["seed"], r["qid"])
            # keep the last non-infra-error record for a key (resume semantics)
            if r.get("infra_error") and key in seen and not seen[key].get("infra_error"):
                continue
            seen[key] = r
        for r in seen.values():
            g = grade(r, qs[r["qid"]]); g["file"] = f.name; g["model"] = r["model"]
            rows.append(g)
    return pd.DataFrame(rows), qs


def boot_ci(df, col, rng):
    """Cluster bootstrap over questions of the mean of `col` (ignoring NaN)."""
    d = df.dropna(subset=[col])
    if d.empty:
        return (np.nan, np.nan)
    per_q = d.groupby("qid")[col].agg(["sum", "count"])
    s, c = per_q["sum"].to_numpy(float), per_q["count"].to_numpy(float)
    idx = rng.integers(0, len(s), size=(B, len(s)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def seed_std(df, col):
    by = df.dropna(subset=[col]).groupby("seed")[col].mean()
    return float(by.std(ddof=1)) if len(by) > 1 else np.nan


def summarise(df, label, rng, cols=("strict", "lenient", "tool_ok", "has_final")):
    out = {"label": label, "n_runs": int(len(df)), "n_questions": int(df.qid.nunique()),
           "seeds": sorted(int(s) for s in df.seed.unique())}
    for c in cols:
        out[c] = {"mean": float(df[c].mean()), "seed_std": seed_std(df, c), "ci95": boot_ci(df, c, rng)}
    for c in ["n_llm_calls", "n_tool_calls", "n_failed_tool_calls", "n_db_tool_calls", "output_tokens",
              "est_context_tokens", "ollama_prompt_tokens", "wall_s"]:
        out[c] = {"mean": float(df[c].mean()), "median": float(df[c].median()), "std": float(df[c].std(ddof=1)) if len(df) > 1 else np.nan}
    out["hit_max_iter_rate"] = float(df.hit_max_iter.mean())
    out["error_rate"] = float(df.error.mean())
    out["budget_exceeded_rate"] = float(df.budget_exceeded.mean()) if "budget_exceeded" in df else None
    return out


def paired_diff(a, b, col, rng):
    """Mean over shared questions of (b - a), bootstrap over questions."""
    pa = a.groupby("qid")[col].mean(); pb = b.groupby("qid")[col].mean()
    common = pa.index.intersection(pb.index)
    d = (pb[common] - pa[common]).to_numpy(float)
    if len(d) == 0:
        return None
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return {"n_questions": int(len(d)), "diff": float(d.mean()),
            "ci95": (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))),
            "a_mean": float(pa[common].mean()), "b_mean": float(pb[common].mean())}


def conflict_metrics(df, rng):
    c = df[df.distractor_final.notna()].copy()
    out = {"n_runs": int(len(c)), "n_questions": int(c.qid.nunique())}
    c["committed_distractor"] = ((c.distractor_final == 1) & (c.strict == 0)).astype(float)
    out["committed_distractor_rate"] = {"mean": float(c.committed_distractor.mean()),
                                        "ci95": boot_ci(c, "committed_distractor", rng)}
    e = c[c.exposed == 1].copy()
    out["n_exposed_runs"] = int(len(e))
    if len(e):
        e["missed"] = ((e.strict == 0) & (e.flagged_conflict == 0)).astype(float)
        e["committed"] = ((e.distractor_final == 1) & (e.strict == 0)).astype(float)
        out["exposed_missed_contradiction_rate"] = {"mean": float(e.missed.mean()), "ci95": boot_ci(e, "missed", rng)}
        out["exposed_committed_distractor_rate"] = {"mean": float(e.committed.mean()), "ci95": boot_ci(e, "committed", rng)}
        out["exposed_flag_rate"] = float(e.flagged_conflict.mean())
        out["exposed_accuracy"] = float(e.strict.mean())
    return out


def fmt(m):
    lo, hi = m["ci95"]
    sd = m["seed_std"]
    return f"{100*m['mean']:.1f}% ± {100*sd:.1f} [{100*lo:.1f}, {100*hi:.1f}]" if sd == sd else f"{100*m['mean']:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


def main():
    rng = np.random.default_rng(RNG_SEED)
    df, qs = load_runs()
    if df.empty:
        print("no results yet"); return
    infra = df[df.infra_error == 1]
    df.to_csv(RES / "graded.csv", index=False)
    d = df[df.infra_error == 0]
    S = {"n_infra_error_runs_excluded": int(len(infra)), "groups": {}}
    md = ["| Condition | n runs (Qs x seeds) | Accuracy strict (± seed std) [95% CI] | Accuracy lenient | Tool-selection OK | Mean LLM calls | Mean tool calls | Mean output tok | Mean est. context tok/run | Budget-exceeded |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    groups = []
    main_ = d[(d.variant == "after") & (d.ablation == "none")]
    groups.append(("after (fixed), all 56 Qs", main_))
    for cat in ["lookup", "aggregation", "multihop", "conflict"]:
        groups.append((f"after, {cat}", main_[main_.category == cat]))
    bef = d[(d.variant == "before") & (d.ablation == "none")]
    fr = sorted(bef.qid.unique())
    frq = [q for q, v in qs.items() if v["fix_relevant"]]
    groups.append(("before (base prompt + single-day velocity), all Qs", bef))
    groups.append(("before, fix-relevant Qs", bef[bef.qid.isin(frq)]))
    groups.append(("after, same fix-relevant Qs", main_[main_.qid.isin(frq)]))
    cm = main_[main_.category.isin(["conflict", "multihop"])]
    for ab in ["reconcile", "subset", "truncate"]:
        a = d[(d.variant == "after") & (d.ablation == ab)]
        if len(a):
            seeds = sorted(a.seed.unique())
            groups.append((f"ablation {ab}, conflict+multihop", a))
            groups.append((f"  baseline (none) same Qs, seeds {seeds}", cm[cm.seed.isin(seeds) & cm.qid.isin(a.qid.unique())]))
    for label, g in groups:
        if g.empty:
            continue
        s = summarise(g, label, rng); S["groups"][label] = s
        md.append(f"| {label} | {s['n_runs']} ({s['n_questions']}x{len(s['seeds'])}) | {fmt(s['strict'])} | {fmt(s['lenient'])} | "
                  f"{fmt(s['tool_ok'])} | {s['n_llm_calls']['mean']:.1f} | {s['n_tool_calls']['mean']:.1f} | "
                  f"{s['output_tokens']['mean']:.0f} | {s['est_context_tokens']['mean']:.0f} | {100*s['budget_exceeded_rate']:.0f}% |")
    S["conflict_after"] = conflict_metrics(main_[main_.category == "conflict"], rng)
    if len(bef):
        S["conflict_before"] = conflict_metrics(bef[bef.category == "conflict"], rng)
        S["before_vs_after_strict_all"] = paired_diff(bef, main_[main_.qid.isin(fr)], "strict", rng)
        S["before_vs_after_strict_fix_relevant"] = paired_diff(bef[bef.qid.isin(frq)], main_[main_.qid.isin(frq)], "strict", rng)
        S["before_vs_after_distractor_fix_relevant"] = paired_diff(bef[bef.qid.isin(frq)].dropna(subset=["distractor_final"]), main_[main_.qid.isin(frq)].dropna(subset=["distractor_final"]), "distractor_final", rng)
    for ab in ["reconcile", "subset", "truncate"]:
        a = d[(d.variant == "after") & (d.ablation == ab)]
        if len(a):
            base = cm[cm.seed.isin(a.seed.unique()) & cm.qid.isin(a.qid.unique())]
            S[f"ablation_{ab}_vs_none_strict"] = paired_diff(base, a, "strict", rng)
            S[f"ablation_{ab}_conflict"] = conflict_metrics(a[a.category == "conflict"], rng)
    # per-question accuracy table (main)
    pq = main_.groupby("qid").agg(category=("category", "first"), strict=("strict", "mean"),
                                  lenient=("lenient", "mean"), tool_ok=("tool_ok", "mean"),
                                  n=("strict", "size")).reset_index()
    pq.to_csv(RES / "per_question_main.csv", index=False)
    S["first_tool_counts_main"] = main_.first_tool.value_counts().head(15).to_dict()
    S["db_tool_call_share_main"] = float(main_.n_db_tool_calls.sum() / max(1, main_.n_tool_calls.sum()))
    S["failed_tool_call_share_main"] = float(main_.n_failed_tool_calls.sum() / max(1, main_.n_tool_calls.sum()))
    S["models"] = sorted(d.model.unique().tolist())
    (RES / "summary.json").write_text(json.dumps(S, indent=2, default=float), encoding="utf-8")
    (RES / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(json.dumps({k: v for k, v in S.items() if k != "groups"}, indent=1, default=float))


if __name__ == "__main__":
    main()
