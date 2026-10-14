# -*- coding: utf-8 -*-
"""② FX3U 合法 I/O 行为矩阵 v2：每条指令最小合法程序，喂固定输入序列，
逐扫描用【独立推导的期望值】做 assert（不是把本程序打印值抄成期望）。
期望依据：FX3U 指令语义（LD/OUT/SET/RST/MOV/CMP/INC/PLS/PLF/MPS-MRD-MPP/ZRST/SFTL/
ALT/ALTP/OUT T/OUT C/M8000/M8002/STL-RET）。手册文档号/页码本环境联网受限，统一标"待补"。

运行：.venv/bin/python -u test_plc_matrix.py
退出码：全部断言通过=0；任一失败=1 并打印 [FAIL]。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sys
from rw.plc.fx3u import FX3U

PASS = 0
FAIL = 0


def run_case(name, prog, inputs_seq, observe, expected, note=""):
    """observe(p)->该扫描观察值；expected 为与 inputs_seq 等长的期望列表。"""
    global PASS, FAIL
    p = FX3U([l for l in prog.strip().splitlines()], scan_ms=10)
    actual = []
    for inp in inputs_seq:
        p.scan(dict(inp))
        actual.append(observe(p))
    ok = actual == expected
    if ok:
        PASS += 1
        print("[PASS] %-28s trail=%s" % (name, actual))
    else:
        FAIL += 1
        print("[FAIL] %-28s" % name)
        print("        expected=%s" % (expected,))
        print("        actual  =%s" % (actual,))
        if note:
            print("        note: %s" % note)


# 1. LD/OUT 直通：Y0 = X0
run_case("LD/OUT 直通",
    "LD X0\nOUT Y0\nEND",
    [{'X0': 0}, {'X0': 1}, {'X0': 1}, {'X0': 0}],
    lambda p: p.get('Y0'),
    [False, True, True, False])

# 2. LDI 取反：Y0 = NOT X0
run_case("LDI 取反",
    "LDI X0\nOUT Y0\nEND",
    [{'X0': 1}, {'X0': 0}],
    lambda p: p.get('Y0'),
    [False, True])

# 3. AND 串联：Y0 = X0 AND X1
run_case("AND 串联",
    "LD X0\nAND X1\nOUT Y0\nEND",
    [{'X0': 1, 'X1': 1}, {'X0': 1, 'X1': 0}, {'X0': 0, 'X1': 1}],
    lambda p: p.get('Y0'),
    [True, False, False])

# 4. OR 并联：Y0 = X0 OR X1
run_case("OR 并联",
    "LD X0\nOR X1\nOUT Y0\nEND",
    [{'X0': 0, 'X1': 0}, {'X0': 1, 'X1': 0}, {'X0': 0, 'X1': 1}],
    lambda p: p.get('Y0'),
    [False, True, True])

# 5. ANI 常闭：Y0 = X0 AND NOT X1
run_case("ANI 常闭",
    "LD X0\nANI X1\nOUT Y0\nEND",
    [{'X0': 1, 'X1': 0}, {'X0': 1, 'X1': 1}],
    lambda p: p.get('Y0'),
    [True, False])

# 6. SET/RST 锁存：X0 置位 Y0，X1 复位 Y0
run_case("SET/RST 锁存",
    "LD X0\nSET Y0\nLD X1\nRST Y0\nEND",
    [{'X0': 1, 'X1': 0}, {'X0': 0, 'X1': 0}, {'X0': 0, 'X1': 1}, {'X0': 0, 'X1': 0}],
    lambda p: p.get('Y0'),
    [True, True, False, False])

# 7. MOV K->D：X0=1 时 D0=10
run_case("MOV K->D",
    "LD X0\nMOV K10 D0\nEND",
    [{'X0': 0}, {'X0': 1}, {'X0': 1}],
    lambda p: p.val('D0'),
    [0, 10, 10])

# 8. CMP K10 D0 M0：D0=5，K10>D0 -> M0(GT)=1,M1(EQ)=0,M2(LT)=0
run_case("CMP K10 D0 -> M0/M1/M2",
    "LD X0\nMOV K5 D0\nCMP K10 D0 M0\nEND",
    [{'X0': 1}],
    lambda p: (p.get('M0'), p.get('M1'), p.get('M2')),
    [(True, False, False)])

# 9. INC D0（电平型：rung ON 每扫描 +1）
run_case("INC D0 (电平)",
    "LD X0\nINC D0\nEND",
    [{'X0': 1}, {'X0': 1}, {'X0': 0}, {'X0': 1}],
    lambda p: p.val('D0'),
    [1, 2, 2, 3])

# 10. ALT Y0（电平型：rung ON 每扫描翻转一次）
run_case("ALT Y0 (电平翻转)",
    "LD X0\nALT Y0\nEND",
    [{'X0': 0}, {'X0': 1}, {'X0': 1}, {'X0': 0}, {'X0': 1}],
    lambda p: p.get('Y0'),
    [False, True, False, False, True],
    note="ALT 电平型：X0 持续 ON 时每扫描翻转；脉冲型翻转用 ALTP。FX 手册页码待补")

# 10b. ALTP Y0（上升沿型：仅 X0 0->1 当扫描翻转一次）
run_case("ALTP Y0 (上升沿翻转)",
    "LD X0\nALTP Y0\nEND",
    [{'X0': 0}, {'X0': 1}, {'X0': 1}, {'X0': 0}, {'X0': 1}, {'X0': 1}],
    lambda p: p.get('Y0'),
    [False, True, True, True, False, False],
    note="ALTP 仅上升沿翻转一次")

# 11. PLS M0：X0 上升沿单扫描脉冲
run_case("PLS M0 上升沿脉冲",
    "LD X0\nPLS M0\nEND",
    [{'X0': 0}, {'X0': 1}, {'X0': 0}, {'X0': 1}, {'X0': 0}],
    lambda p: p.get('M0'),
    [False, True, False, True, False])

# 11b. PLF M0：X0 下降沿单扫描脉冲
run_case("PLF M0 下降沿脉冲",
    "LD X0\nPLF M0\nEND",
    [{'X0': 1}, {'X0': 1}, {'X0': 0}, {'X0': 0}, {'X0': 1}, {'X0': 0}],
    lambda p: p.get('M0'),
    [False, False, True, False, False, True],
    note="PLF 在 X0 1->0 当扫描输出一个扫描")

# 12. MPS/MRD/MPP 栈
run_case("MPS/MRD/MPP 双支路",
    "LD X0\nMPS\nAND X1\nOUT Y0\nMRD\nAND X2\nOUT Y1\nMPP\nAND X3\nOUT Y2\nEND",
    [{'X0': 1, 'X1': 1, 'X2': 1, 'X3': 1}, {'X0': 1, 'X1': 0, 'X2': 1, 'X3': 0}],
    lambda p: (p.get('Y0'), p.get('Y1'), p.get('Y2')),
    [(True, True, True), (False, True, False)])

# 13. ZRST S0..S5（两扫描：先 X0 置位 S2/S4，再 X1 批量清）
run_case("ZRST S0..S5 (分两扫描)",
    "LD X0\nSET S2\nSET S4\nLD X1\nZRST S0 S5\nEND",
    [{'X0': 1, 'X1': 0}, {'X0': 0, 'X1': 1}],
    lambda p: [p.get('S%d' % i) for i in range(6)],
    [[False, False, True, False, True, False],
     [False, False, False, False, False, False]],
    note="扫描1 X0=1 置 S2/S4；扫描2 X1=1 ZRST 清 S0..S5")

# 14. 特殊 M：M8000 常ON / M8002 仅首扫描 ON
run_case("M8000常ON / M8002首扫描",
    "LD M8000\nOUT Y0\nLD M8002\nOUT Y1\nEND",
    [{}, {}, {}],
    lambda p: (p.get('Y0'), p.get('Y1')),
    [(True, True), (True, False), (True, False)])

# 15. STL/RET 步进（首扫描 M8002 SET S0 初始化）
run_case("STL 步进跳转 (含M8002初始化)",
    "LD M8002\nSET S0\nSTL S0\nLD X0\nSET S20\nSTL S20\nLD X1\nOUT Y0\nRET\nEND",
    [{'X0': 0, 'X1': 1}, {'X0': 1, 'X1': 1}, {'X0': 0, 'X1': 0}, {'X0': 0, 'X1': 1}],
    lambda p: (p.get('Y0'), p.get('S20'), p.get('S0')),
    [(False, False, True), (True, True, False), (False, True, False), (True, True, False)],
    note="首扫描 M8002/SET S0 初始化步；X0 上升沿转移到 S20")

# 16. OUT T0 定时器（100ms 基，scan_ms=10，K5=500ms=50 扫描）
run_case("OUT T0 K5 (100ms基, 50扫描)",
    "LD X0\nOUT T0 K5\nLD T0\nOUT Y0\nEND",
    [{'X0': 1}] * 49 + [{'X0': 1}] + [{'X0': 1}] * 11 + [{'X0': 0}],
    lambda p: p.get('Y0'),
    [False] * 49 + [True] + [True] * 11 + [False],
    note="T0 100ms基 K5=500ms=50扫描；第50扫描 T0 触点 ON；X0 OFF 当扫描复位")

# 17. OUT C0 加计数器（X0 上升沿 +1，K3=数到3输出ON；X1 RST）
run_case("OUT C0 K3 加计数",
    "LD X0\nOUT C0 K3\nLD C0\nOUT Y0\nLD X1\nRST C0\nEND",
    [{'X0': 1, 'X1': 0}, {'X0': 0, 'X1': 0}, {'X0': 1, 'X1': 0},
     {'X0': 0, 'X1': 0}, {'X0': 1, 'X1': 0}, {'X0': 0, 'X1': 1}],
    lambda p: (p.get('Y0'), p.cval.get('C0', 0)),
    [(False, 1), (False, 1), (False, 2), (False, 2), (True, 3), (True, 0)],
    note="C0 上升沿加计数，数到3输出ON；扫描6 X1=1 RST C0：RST 在 OUT Y0 之后执行，故本扫 Y0 仍=1、cval=0")

# 18. SFTL M0->M10 K4 K1（左移：src M0 1位移入 dst M10..M13 4位）
run_case("SFTL M0->M10 K4 K1",
    "LD X0\nSFTL M0 M10 K4 K1\nEND",
    [{'X0': 1, 'M0': 1}, {'X0': 1, 'M0': 0}, {'X0': 1, 'M0': 1}],
    lambda p: [p.get('M%d' % i) for i in range(10, 14)],
    [[True, False, False, False],
     [False, True, False, False],
     [True, False, True, False]],
    note="SFTL 把 src 位移入 dst 高位方向；X0=1 每扫描移一次")


print("\n==== 结果: PASS=%d FAIL=%d ====" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)