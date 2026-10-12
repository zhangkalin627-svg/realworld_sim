"""validate_gui.py — app.py 命令行/截图入口的回归自检（归属：真实的世界-GUI与性能）
运行:  python3 tools/validate_gui.py
每项打印 PASS/FAIL；有失败返回非零退出码。离屏运行（QT_QPA_PLATFORM=offscreen）。
"""
import os, sys, shutil, subprocess, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, QT_QPA_PLATFORM="offscreen")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS" if ok else "FAIL"), name, detail)


def run(args, cwd, timeout=180):
    p = subprocess.run([sys.executable, "app.py"] + args, cwd=cwd, env=ENV,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


# G1：--screenshot 漏写输出文件名 → 用法提示 + 退出码 2，不出 traceback
rc, out = run(["--screenshot"], HERE)
check("G1 --screenshot 缺参数给用法提示", rc == 2 and "用法" in out and "Traceback" not in out,
      f"rc={rc}")

# G2：公开版（没有私有设备页）截 plc 页 → 退回 plcdemo，正常出图
tmp = tempfile.mkdtemp(prefix="rw_pub_")
try:
    pub = os.path.join(tmp, "rw")
    shutil.copytree(HERE, pub, ignore=shutil.ignore_patterns(
        ".git", "out", "__pycache__", "*.zip", ".venv"))
    png = os.path.join(tmp, "plc.png")
    rc, out = run(["--screenshot", png, "plc"], pub)
    ok = rc == 0 and os.path.exists(png) and os.path.getsize(png) > 10000 and "Traceback" not in out
    check("G2 公开版 --screenshot plc 不崩、退回 plcdemo", ok,
          f"rc={rc} png={os.path.exists(png)}" + ("" if ok else "\n" + out[-1500:]))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# G3：子页号越界（w3d9）→ 提示后退回 w3d0，不抛 KeyError
tmp = tempfile.mkdtemp(prefix="rw_w3d_")
try:
    png = os.path.join(tmp, "w3d.png")
    rc, out = run(["--screenshot", png, "w3d9"], HERE, timeout=300)
    ok = rc == 0 and os.path.exists(png) and "Traceback" not in out
    check("G3 --screenshot w3d9 越界退回 w3d0", ok, f"rc={rc}" + ("" if ok else "\n" + out[-1500:]))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

n = len(results); k = sum(results)
print(f"validate_gui: {k}/{n} pass")
sys.exit(0 if k == n else 1)
