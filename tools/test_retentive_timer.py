import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""积算型定时器（T246-T255）断开保持 + 触点锁存回归。

卡布29 探活发现：旧实现 _tkeep 只初始化/读取、从不写入，
积算定时器断开后 tval 被读成 0（保持失效），触点也随 acc 断开回落。
"""
from rw.plc.fx3u import FX3U

fails = []

def chk(name, cond):
    print(('PASS' if cond else 'FAIL'), name)
    if not cond:
        fails.append(name)

# T246 K100：1ms 时基，预置 100ms；scan_ms=10 → 10 扫描到预置
prog = ['LD X000', 'OUT T246 K100', 'LD T246', 'OUT Y000', 'END']

# 1) 断开保持：计时 5 扫描(50ms)后断开，tval 应保持 50
sim = FX3U(prog)
for _ in range(5):
    sim.scan({'X000': True})
for _ in range(10):
    sim.scan({'X000': False})
chk('T246 断开后 tval 保持 50ms', sim.tval.get('T246') == 50)
chk('T246 断开期间触点未置位', sim.bits.get('T246') in (False, None) and sim.bits.get('Y000') in (False, None))

# 2) 重新 ON 继续累计到预置：再 5 扫描 → 100ms，触点锁存
for _ in range(5):
    sim.scan({'X000': True})
chk('T246 再 ON 后续计到 100ms', sim.tval.get('T246') == 100)
chk('T246 达预置触点置位', sim.bits.get('T246') is True)

# 3) 触点锁存：acc 断开后触点仍 ON
for _ in range(3):
    sim.scan({'X000': False})
chk('T246 触点断开后锁存保持', sim.bits.get('T246') is True)

# 4) RST 复位：tval 清零、触点回落
sim2 = FX3U(['LD X000', 'OUT T246 K100', 'LD X001', 'RST T246', 'END'])
for _ in range(12):
    sim2.scan({'X000': True, 'X001': False})
chk('sim2 预置到达触点 ON', sim2.bits.get('T246') is True)
for _ in range(3):
    sim2.scan({'X000': False, 'X001': True})
chk('RST 后 tval 清零', sim2.tval.get('T246') == 0)
chk('RST 后触点回落', sim2.bits.get('T246') is False)

# 5) 非积算回归：T0 断开清零、触点不锁存
sim3 = FX3U(['LD X000', 'OUT T0 K100', 'LD T0', 'OUT Y000', 'END'])
for _ in range(12):                      # 100ms 时基 ×100 = 10000ms；12 扫描 120ms 未到
    sim3.scan({'X000': True})
chk('T0 计时中触点未置位', sim3.bits.get('T246') in (False, None) and sim3.bits.get('T0') in (False, None))
v = sim3.tval.get('T0')
for _ in range(5):
    sim3.scan({'X000': False})
chk('T0 断开清零', sim3.tval.get('T0') == 0)

# 6) T250（积算 100ms 时基）也保持
sim4 = FX3U(['LD X000', 'OUT T250 K1', 'END'])
for _ in range(9):                       # 9×10=90ms < 100ms 预置
    sim4.scan({'X000': True})
for _ in range(5):
    sim4.scan({'X000': False})
chk('T250 断开保持 90ms', sim4.tval.get('T250') == 90)
sim4.scan({'X000': True})
chk('T250 第 10 扫描到预置触点 ON', sim4.bits.get('T250') is True)

print('---')
print('ALL PASS' if not fails else f'{len(fails)} FAIL: {fails}')
raise SystemExit(1 if fails else 0)
