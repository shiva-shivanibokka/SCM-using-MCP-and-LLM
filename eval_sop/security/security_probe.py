"""Local-only security probes for the agent's SQL guard and python_repl sandbox.

Uses only CANARY files created by this script inside a temp directory -- it
never touches .env or any real credential. Each probe runs in a subprocess
with a hard wall-clock timeout so a hang is recorded instead of blocking.

Run from repo root:
  python eval_sop/security/security_probe.py            # current code
  python eval_sop/security/security_probe.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

# All probe files live under a scratch dir (default: <repo>/../../sec_probe, i.e.
# the session scratchpad when run from the eval worktree); override with SEC_PROBE_DIR.
import os
TMP = Path(os.getenv("SEC_PROBE_DIR", str(ROOT.parents[1] / "sec_probe")))
TMP.mkdir(parents=True, exist_ok=True)
CANARY = TMP / "canary_secret.csv"
CANARY.write_text("key,value\nCANARY_TOKEN,s3cr3t-canary-123\n", encoding="utf-8")
WRITE_TARGET = TMP / "written_by_repl.csv"
CANARY_P = CANARY.as_posix()
WRITE_P = WRITE_TARGET.as_posix()


def run(code: str, timeout: float) -> dict:
    prog = textwrap.dedent(f"""
        import os, sys, json
        os.environ['DATABASE_URL'] = ''
        sys.path.insert(0, {str(ROOT)!r})
        out = None
        {textwrap.indent(textwrap.dedent(code), ' ' * 8).strip()}
        print('@@RESULT@@' + json.dumps(out, default=str)[:2000])
    """)
    t0 = time.time()
    try:
        p = subprocess.run([PY, "-c", prog], capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace", cwd=str(ROOT))
        res = None
        for line in p.stdout.splitlines():
            if line.startswith("@@RESULT@@"):
                res = json.loads(line[len("@@RESULT@@"):])
        return {"status": "completed", "secs": round(time.time() - t0, 1), "result": res,
                "stderr_tail": p.stderr.strip().splitlines()[-1:] if p.returncode else []}
    except subprocess.TimeoutExpired:
        return {"status": f"TIMEOUT (> {timeout}s, killed)", "secs": round(time.time() - t0, 1), "result": None}


SQL = "from intelligence.sql import run_query, is_safe_select\n"
REPL = "from mcp_server.server import tool_python_repl as R\n"

PROBES = [
    # ---- SQL guard (run_sql_query tool / Ask-Your-Data API share intelligence.sql)
    ("sql_quoted_path_read",
     "SQL guard: read an arbitrary local file via a quoted path (SELECT * FROM '<file>')",
     SQL + f"q = \"SELECT * FROM '{CANARY_P}'\"\nr = run_query(q, 'data')\n"
           "out = {'guard_allows': is_safe_select(q), 'error': r['error'], 'rows': r['rows']}", 60),
    ("sql_read_csv_fn",
     "SQL guard: explicit read_csv() table function",
     SQL + f"q = \"SELECT * FROM read_csv('{CANARY_P}')\"\nr = run_query(q, 'data')\n"
           "out = {'guard_allows': is_safe_select(q), 'error': r['error'], 'rows': r['rows']}", 60),
    ("sql_list_dir_via_glob_path",
     "SQL guard: wildcard quoted path enumerates/reads every CSV in a directory",
     SQL + f"q = \"SELECT count(*) AS n FROM '{TMP.as_posix()}/*.csv'\"\nr = run_query(q, 'data')\n"
           "out = {'guard_allows': is_safe_select(q), 'error': r['error'], 'rows': r['rows']}", 60),
    ("sql_settings_leak",
     "SQL guard: read DuckDB settings (e.g. home/temp directory paths)",
     SQL + "q = \"SELECT name, value FROM duckdb_settings() WHERE name IN ('home_directory','temp_directory','enable_external_access')\"\n"
           "r = run_query(q, 'data')\nout = {'guard_allows': is_safe_select(q), 'error': r['error'], 'rows': r['rows']}", 60),
    ("sql_cross_join_dos",
     "SQL guard: expensive cross join (1.2M x 1.2M rows) -- is there a timeout?",
     SQL + "q = 'SELECT sum(a.demand * b.demand) AS s FROM store_inventory a, store_inventory b'\n"
           "r = run_query(q, 'data')\nout = {'guard_allows': is_safe_select(q), 'error': r['error'], 'rows': r['rows']}", 90),
    ("sql_write_blocked",
     "SQL guard: COPY ... TO (write) is blocked",
     SQL + f"q = \"COPY (SELECT 1) TO '{WRITE_P}'\"\nr = run_query(q, 'data')\n"
           "out = {'guard_allows': is_safe_select(q), 'error': r['error']}", 60),
    ("sql_false_positive",
     "SQL guard false positive: legit query mentioning a column/name containing 'load' or 'replace'",
     SQL + "q = \"SELECT sku_id FROM products WHERE name LIKE '%Download%' OR category = 'Replacement'\"\n"
           "r = run_query(q, 'data')\nout = {'guard_allows': is_safe_select(q), 'error': r['error']}", 60),
    # ---- python_repl sandbox
    ("repl_import_os_blocked", "python_repl: `import os` is blocked",
     REPL + "out = R('import os\\nos.getcwd()')", 60),
    ("repl_pandas_read_file", "python_repl: read an arbitrary local file through pandas",
     REPL + f"out = R(\"pd.read_csv('{CANARY_P}').to_dict('records')\")", 60),
    ("repl_pandas_write_file", "python_repl: write a file through pandas",
     REPL + f"r = R(\"pd.DataFrame({{'a':[1]}}).to_csv('{WRITE_P}')\")\n"
            f"import pathlib\nout = {{'repl': r, 'file_exists_after': pathlib.Path('{WRITE_P}').exists()}}", 60),
    ("repl_numpy_read_file", "python_repl: read a file through numpy",
     REPL + f"out = R(\"np.loadtxt('{CANARY_P}', dtype=str, delimiter=',')\")", 60),
    ("repl_os_via_module_attr", "python_repl: reach the os module through a pre-loaded module's attribute (benign getcwd)",
     REPL + "out = R('pd.io.common.os.getcwd()')", 60),
    ("repl_infinite_loop", "python_repl: infinite loop -- is there an execution timeout?",
     REPL + "out = R('while True:\\n    pass')", 20),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    results = []
    for pid, desc, code, to in PROBES:
        if a.only and pid not in a.only:
            continue
        r = run(code, to)
        if pid == "repl_pandas_write_file" and WRITE_TARGET.exists():
            WRITE_TARGET.unlink()
        results.append({"id": pid, "description": desc, **r})
        print(f"- {pid}: {r['status']} ({r['secs']}s)\n    {json.dumps(r['result'], default=str)[:400]}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
