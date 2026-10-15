import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""PLC 软元件范围守卫 + 负预置守卫回归（P2，卡林 12:47 派单授权修复）。

守卫范围：T0-T255 / C0-C255 / D0-D8511 越界 -> 装载期带原行号 ValueError；
OUT T/OUT C 的 K<0 -> 装载期 ValueError；MOV 等数据搬移的负 K 合法不误伤；
边界 T255 / D8511 / C255 合法。
"""
from rw.plc.fx3u import FX3U

fails = []

def chk(name, cond):
    print(('PASS' if cond else 'FAIL'), name)
    if not cond:
        fails.append(name)

def expect_compile_error(il, needle, name):
    try:
        FX3U(il)
    except ValueError as e:
        chk(name, needle in str(e))
        return
    chk(name, False)

# ---- 1. 越界 T/C/D 装载期拒 ----
expect_compile_error(["LD M8000", "OUT T1000 K10", "END"], "T0-T255", "越界 T1000 拒")
expect_compile_error(["LD M8000", "OUT T256 K10", "END"], "T0-T255", "边界外 T256 拒")
expect_compile_error(["LD M8000", "OUT C300 K3", "END"], "C0-C255", "越界 C300 拒")
expect_compile_error(["LD M8000", "OUT C256 K3", "END"], "C0-C255", "边界外 C256 拒")
expect_compile_error(["LD M8000", "MOV K5 D99999", "END"], "D0-D8511", "越界 D99999 拒")
expect_compile_error(["LD M8000", "MOV K5 D8512", "END"], "D0-D8511", "边界外 D8512 拒")
expect_compile_error(["LD T1000", "OUT Y0", "END"], "T0-T255", "触点引用越界 T1000 拒")

# ---- 2. 负预置只在 OUT T/C 语境拒，MOV 负 K 合法 ----
expect_compile_error(["LD M8000", "OUT T0 K-5", "END"], "预置值不能为负", "OUT T0 K-5 拒")
expect_compile_error(["LD M8000", "OUT C0 K-3", "END"], "预置值不能为负", "OUT C0 K-3 拒")
try:
    p = FX3U(["LD M8000", "MOV K-100 D0", "END"])
    p.scan({})
    chk("MOV K-100 D0 合法(负数据)", p.val("D0") == -100)
except ValueError:
    chk("MOV K-100 D0 合法(负数据)", False)

# ---- 3. 边界合法 ----
try:
    p = FX3U(["LD M8000", "OUT T255 K10", "LD M8000", "MOV K7 D8511", "END"])
    p.scan({})
    chk("T255 / D8511 边界合法编译运行", p.val("D8511") == 7)
except ValueError:
    chk("T255 / D8511 边界合法编译运行", False)

print('----')
if fails:
    print('FAIL=%d FAILED: %s' % (len(fails), fails))
    sys.exit(1)
print('ALL PASS')
