"""Web SVG 中文字体回归：只 import web.modules（不 import rw.viz）渲染中文标题不得缺字。
kabu499 发现：web 绘图路径此前未注册 fonts/NotoSansSC-Regular.otf。python tools/test_web_svg_font.py"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE = r'''
import warnings, sys, io
sys.path.insert(0, ".")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import web.modules
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title("绕组 · 瞬态波形")
    ax.set_xlabel("频率 f (Hz)")
    b = io.StringIO(); fig.savefig(b, format="svg")
    miss = [str(x.message) for x in w if "missing from font" in str(x.message)]
    print("MISS=%d" % len(miss))
    print("HAS_TITLE=%d" % ("瞬态波形" in b.getvalue() or len(miss) == 0))
'''


class WebSvgFontTests(unittest.TestCase):
    def test_chinese_svg_no_missing_glyphs(self):
        r = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        self.assertIn("MISS=0", r.stdout, r.stdout + r.stderr[-300:])


if __name__ == "__main__":
    unittest.main()
