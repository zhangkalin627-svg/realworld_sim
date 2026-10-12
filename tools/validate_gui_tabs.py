"""validate_gui_tabs.py — app.py --screenshot 各页名是否真的切到对应顶层页签（归属：真实的世界-GUI与性能）
运行:  python3 tools/validate_gui_tabs.py
原理：子进程里把 MainWindow.grab 包一层，截图那一刻打印当前顶层页签名，再与期望比较。
PASS/FAIL 逐项打印；有失败返回非零退出码。
"""
import os, sys, subprocess, tempfile, shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, QT_QPA_PLATFORM="offscreen")
PROBE = r'''
import sys
sys.path.insert(0, %r)
import app
_g = app.MainWindow.grab
def grab(self, *a):
    print("TOPTAB=" + self.top.tabText(self.top.currentIndex()), flush=True)
    return _g(self, *a)
app.MainWindow.grab = grab
sys.argv = ["app.py", "--screenshot", %r, %r]
app.main()
'''
# 页名 → 期望顶层页签名
CASES = [("struct", "② 结构受力"), ("structdemo", "② 结构受力"), ("plcdemo", "③ PLC 虚拟调试"),
         ("world1", "④ 物理沙盒")]

results = []
tmp = tempfile.mkdtemp(prefix="rw_tabs_")
try:
    for page, want in CASES:
        png = os.path.join(tmp, page + ".png")
        p = subprocess.run([sys.executable, "-c", PROBE % (HERE, png, page)], cwd=HERE, env=ENV,
                           capture_output=True, text=True, timeout=240)
        got = [l[7:] for l in p.stdout.splitlines() if l.startswith("TOPTAB=")]
        ok = p.returncode == 0 and got and got[-1] == want
        results.append(bool(ok))
        print(("PASS" if ok else "FAIL"), f"T {page:<10} 期望「{want}」 实际「{got[-1] if got else '无截图'}」 rc={p.returncode}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print(f"validate_gui_tabs: {sum(results)}/{len(results)} pass")
sys.exit(0 if all(results) else 1)
