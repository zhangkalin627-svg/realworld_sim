"""回归：①页自动重算只在参数真变时触发（家机 09:33 反馈：开下拉框/切子页误触发）。
离屏运行：QT_QPA_PLATFORM=offscreen python3 tools/test_gui_auto_recompute.py"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6 import QtWidgets, QtCore
from app import InductionPage

app = QtWidgets.QApplication([])
pg = InductionPage()
pg.sim = object()                      # 假装已算过一次（自动重算的前置条件）
pg._last_sig = pg._sig()               # 模拟 start() 记录基准快照

fails = []
# 1) 信号乱触发但值没变 → 不排重算
for _ in range(3):
    pg._kick()
if pg._deb.isActive(): fails.append("值没变却排了重算")
# 2) 弹下拉框/切子页级别的 widget 事件（不改值）→ 不排重算
pg.freq.showPopup(); pg.freq.hidePopup()
pg.tabs.setCurrentIndex(2); pg.tabs.setCurrentIndex(0)
app.processEvents()
if pg._deb.isActive(): fails.append("弹下拉框/切子页后排了重算")
# 3) 真改值 → 排重算
pg.theat.setValue(pg.theat.value() + 0.5)
app.processEvents()
if not pg._deb.isActive(): fails.append("真改值反而没排重算")
pg._deb.stop()
# 4) 改回原值 → 相对基准快照也算变化（与上次计算不同），应排重算
pg._last_sig = pg._sig()
pg.theat.setValue(pg.theat.value() - 0.5)
app.processEvents()
if not pg._deb.isActive(): fails.append("改回别的值没排重算")
pg._deb.stop()

print("误触发守卫+真变触发:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
