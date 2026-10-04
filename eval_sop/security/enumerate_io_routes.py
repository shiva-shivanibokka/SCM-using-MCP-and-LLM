"""Research aid: enumerate attribute paths (non-underscore names, depth <= 3)
reachable from the modules pre-loaded into python_repl, and list (a) module
objects with dangerous capabilities and (b) callables whose names suggest
file/OS/process I/O. Used to build the python_repl attribute blocklist.
  python eval_sop/security/enumerate_io_routes.py
"""
import collections, datetime, inspect, json, math, re, types
import numpy as np
import pandas as pd

ROOTS = {"pd": pd, "np": np, "json": json, "re": re, "math": math,
         "collections": collections, "datetime": datetime}
DANGEROUS_MODULES = {"os", "sys", "subprocess", "shutil", "pathlib", "io", "builtins", "importlib",
                     "ctypes", "socket", "pickle", "marshal", "tempfile", "codecs", "gzip", "bz2",
                     "lzma", "zipfile", "tarfile", "mmap", "urllib", "http", "shelve", "glob",
                     "posixpath", "ntpath", "platform", "inspect", "types", "runpy", "code",
                     "threading", "multiprocessing", "signal", "ftplib", "smtplib", "sqlite3", "webbrowser"}
IO_WORDS = re.compile(r"(^|_)(open|load|loads|save|savez|read|write|dump|dumps|tofile|fromfile|memmap|"
                      r"system|popen|exec|eval|compile|import|remove|unlink|rmdir|mkdir|rename|"
                      r"get_handle|urlopen|run|call)($|_)|^read_|^to_(csv|excel|parquet|pickle|json|sql|hdf|"
                      r"feather|stata|html|xml|latex|clipboard|orc|gbq|string|markdown)$", re.I)

seen, mods, funcs = set(), {}, {}
q = collections.deque((name, obj, 0) for name, obj in ROOTS.items())
while q:
    path, obj, depth = q.popleft()
    if id(obj) in seen and isinstance(obj, types.ModuleType):
        continue
    seen.add(id(obj))
    if depth >= 3:
        continue
    try:
        names = dir(obj)
    except Exception:
        continue
    for n in names:
        if n.startswith("_"):
            continue
        try:
            v = getattr(obj, n)
        except Exception:
            continue
        p = f"{path}.{n}"
        if isinstance(v, types.ModuleType):
            base = v.__name__.split(".")[0]
            if base in DANGEROUS_MODULES or n in DANGEROUS_MODULES:
                mods.setdefault(n, p)
            if base in ("numpy", "pandas") or n in DANGEROUS_MODULES:
                q.append((p, v, depth + 1))
        elif callable(v) and IO_WORDS.search(n):
            funcs.setdefault(n, p)
        elif inspect.isclass(v) and depth < 2:
            q.append((p, v, depth + 1))
print("== dangerous module objects reachable by attribute name (name -> example path)")
for n, p in sorted(mods.items()):
    print(f"  {n:20s} {p}")
print("== I/O-like callables (name -> example path)")
for n, p in sorted(funcs.items()):
    print(f"  {n:24s} {p}")
