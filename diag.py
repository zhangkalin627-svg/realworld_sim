# -*- coding: utf-8 -*-
"""诊断摘要（JSON）：随 issue 粘贴。用法: python3 diag.py"""
import json, platform, subprocess, sys, os, shutil
def g(cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: return ""
d = {"tool": "realworld_sim diag", "note": "仿真结果诊断摘要，不含任何用户数据路径以外信息"}
try:
    from rw.version import VERSION
    d["version"] = VERSION
except Exception: d["version"] = "unknown"
d["git_commit"] = g(["git", "rev-parse", "--short", "HEAD"]) if os.path.isdir(".git") else ""
d["os"] = platform.platform(); d["python"] = sys.version.split()[0]
d["deps"] = {m: (lambda: __import__(m) and getattr(__import__(m), "__version__", "ok"))() for m in ("numpy", "scipy") if True}
d["ngspice"] = g(["ngspice", "--version"]).splitlines()[0] if shutil.which("ngspice") else "NOT FOUND"
print(json.dumps(d, ensure_ascii=False, indent=2))
