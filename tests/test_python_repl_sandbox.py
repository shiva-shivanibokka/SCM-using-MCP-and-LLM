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
    ok = ("Error" not in out) and ("{" in out)
    assert ok


def test_pandas_read_of_local_file_blocked(tmp_path):
    f = tmp_path / "canary.csv"
    f.write_text("k,v\nTOKEN,canary-xyz\n", encoding="utf-8")
    out = R(f"pd.read_csv('{f.as_posix()}')")
    leaked, blocked = ("canary-xyz" in out), ("SecurityError" in out)  # never echo `out` (may hold secrets)
    assert not leaked and blocked


def test_numpy_read_of_local_file_blocked(tmp_path):
    f = tmp_path / "canary.csv"
    f.write_text("k,v\nTOKEN,canary-xyz\n", encoding="utf-8")
    out = R(f"np.loadtxt('{f.as_posix()}', dtype=str, delimiter=',')")
    leaked, blocked = ("canary-xyz" in out), ("SecurityError" in out)  # never echo `out` (may hold secrets)
    assert not leaked and blocked


def test_pandas_write_blocked(tmp_path):
    f = tmp_path / "out.csv"
    out = R(f"df.head(1).to_csv('{f.as_posix()}')")
    wrote, blocked = f.exists(), ("SecurityError" in out)
    assert not wrote and blocked


def test_os_via_module_attribute_blocked():
    out = R("pd.io.common.os.getcwd()")
    blocked = "SecurityError" in out
    assert blocked


# --- Bypasses found in adversarial review / enumerate_io_routes.py ----------
# Each test first asserts the attack has no effect; the canary must never leak
# and no file may be written.

import sys  # noqa: E402


def _canary(tmp_path):
    f = tmp_path / "canary.csv"
    f.write_text("k,v\nTOKEN,canary-xyz\n", encoding="utf-8")
    return f


def test_numpy_datasource_open_read_blocked(tmp_path):
    f = _canary(tmp_path)
    out = R(f"np.lib._datasource.open('{f.as_posix()}').read()")
    leaked, blocked = ("canary-xyz" in out), ("SecurityError" in out)  # never echo `out` (may hold secrets)
    assert not leaked and blocked


def test_numpy_datasource_open_write_blocked(tmp_path):
    f = tmp_path / "w.txt"
    out = R(f"np.lib._datasource.open('{f.as_posix()}', 'w').write('x')")
    wrote, blocked = f.exists(), ("SecurityError" in out)
    assert not wrote and blocked


def test_to_string_buf_keyword_write_blocked(tmp_path):
    f = tmp_path / "s.txt"
    out = R(f"df.head(1).to_string(buf='{f.as_posix()}')")
    wrote, blocked = f.exists(), ("SecurityError" in out)
    assert not wrote and blocked


def test_to_string_positional_path_write_blocked(tmp_path):
    f = tmp_path / "s2.txt"
    out = R(f"df.head(1).to_string('{f.as_posix()}')")
    wrote, blocked = f.exists(), ("SecurityError" in out)
    assert not wrote and blocked


def test_info_buf_write_blocked(tmp_path):
    f = tmp_path / "info.txt"
    out = R(f"df.head(1).info(buf='{f.as_posix()}')")
    wrote, blocked = f.exists(), ("SecurityError" in out)
    assert not wrote and blocked


def test_open_memmap_write_blocked(tmp_path):
    f = tmp_path / "m.npy"
    out = R(f"np.lib.format.open_memmap('{f.as_posix()}', mode='w+', dtype='f8', shape=(2,))")
    wrote, blocked = f.exists(), ("SecurityError" in out)
    assert not wrote and blocked


def test_ndarray_dump_write_blocked(tmp_path):
    f = tmp_path / "a.pkl"
    out = R(f"np.arange(3).dump('{f.as_posix()}')")
    wrote, blocked = f.exists(), ("SecurityError" in out)
    assert not wrote and blocked


def test_json_codecs_open_read_blocked(tmp_path):
    f = _canary(tmp_path)
    out = R(f"json.codecs.open('{f.as_posix()}').read()")
    leaked, blocked = ("canary-xyz" in out), ("SecurityError" in out)  # never echo `out` (may hold secrets)
    assert not leaked and blocked


def test_pandas_get_handle_read_blocked(tmp_path):
    f = _canary(tmp_path)
    out = R(f"pd.core.frame.get_handle('{f.as_posix()}', 'r').handle.read()")
    leaked, blocked = ("canary-xyz" in out), ("SecurityError" in out)  # never echo `out` (may hold secrets)
    assert not leaked and blocked


def test_subprocess_via_f2py_blocked():
    exe = sys.executable.replace("\\", "/")
    out = R(f"np.f2py.subprocess.check_output(['{exe}', '-c', 'print(\"canary-proc\")'])")
    ran, blocked = ("canary-proc" in out), ("SecurityError" in out)
    assert not ran and blocked


def test_format_string_attribute_traversal_env_leak_blocked(monkeypatch):
    monkeypatch.setenv("REPL_CANARY_ENV", "leak-xyz")
    out = R("'{0.__globals__[sys].modules[os].environ}'.format(pd.DataFrame.to_dict)")
    leaked, blocked = ("leak-xyz" in out), ("SecurityError" in out)  # never echo `out` (may hold secrets)
    assert not leaked and blocked


def test_legit_fstring_and_to_string_still_work():
    out = R("x = df['demand'].sum()\nprint(f'{x:,.0f}')\ndf.head(2)[['sku_id','demand']].to_string(index=False)")
    ok = ("Error" not in out) and ("sku_id" in out)
    assert ok
