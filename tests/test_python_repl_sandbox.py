"""Security regression tests for the agent's python_repl tool: pandas/numpy
file I/O and module-attribute escapes must be rejected. Files live under
pytest's tmp_path only."""
import os

import pytest

os.environ.setdefault("DATABASE_URL", "")
srv = pytest.importorskip("mcp_server.server")
R = srv.tool_python_repl


def test_normal_analysis_still_works():
    out = R("df.groupby('category')['demand'].sum().sort_values().tail(1).to_dict()")
    assert "Error" not in out and "{" in out


def test_pandas_read_of_local_file_blocked(tmp_path):
    f = tmp_path / "canary.csv"
    f.write_text("k,v\nTOKEN,canary-xyz\n", encoding="utf-8")
    out = R(f"pd.read_csv('{f.as_posix()}')")
    assert "canary-xyz" not in out and "SecurityError" in out


def test_numpy_read_of_local_file_blocked(tmp_path):
    f = tmp_path / "canary.csv"
    f.write_text("k,v\nTOKEN,canary-xyz\n", encoding="utf-8")
    out = R(f"np.loadtxt('{f.as_posix()}', dtype=str, delimiter=',')")
    assert "canary-xyz" not in out and "SecurityError" in out


def test_pandas_write_blocked(tmp_path):
    f = tmp_path / "out.csv"
    out = R(f"df.head(1).to_csv('{f.as_posix()}')")
    assert not f.exists() and "SecurityError" in out


def test_os_via_module_attribute_blocked():
    out = R("pd.io.common.os.getcwd()")
    assert "SecurityError" in out
