"""回归：工作线程→GUI 的信号必须编组到主线程（v0.32 家机 QBackingStore 崩溃根因）。
根因：PySide6 对裸 lambda 接收者不做线程编组——`wk.progress.connect(lambda: bar.setValue(...))`
的 lambda 在工作线程里直接跑，进度更新从非 GUI 线程触碰 QWidget，跑完仿真后点下拉框即段错误。
修法：app.py 的 progress 改连绑定 @Slot 方法 _on_progress。
本测试两层：
1) 动态：镜像 app.py 连接拓扑（QObject Worker 在工作线程发 progress → 页面绑定 @Slot 方法），
   断言 slot 全部落在主线程；同信号挂裸 lambda 对照，必须全落在工作线程（证明测试敏感）。
2) 静态：断言 app.py 不存在 worker 信号接裸 lambda 的写法，防回潮。
离屏运行：QT_QPA_PLATFORM=offscreen python3 tools/test_gui_progress_thread.py"""
import os, re, sys, threading
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6 import QtWidgets, QtCore

app = QtWidgets.QApplication([])
main_tid = threading.get_ident()

class FakeWorker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    done = QtCore.Signal()
    @QtCore.Slot()
    def run(self):
        for i in range(3):
            self.progress.emit(i, f"s{i}")
        self.done.emit()

res = {"bound": [], "lambda": []}
class Page(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.bar = QtWidgets.QProgressBar(self)
        self.status = QtWidgets.QLabel(self)
    @QtCore.Slot(int, str)
    def _on_progress(self, v, s):   # 与 app.py InductionPage._on_progress 同形
        res["bound"].append(threading.get_ident() == main_tid)
        self.bar.setValue(v); self.status.setText(s)

pg = Page()
wk = FakeWorker()
th = QtCore.QThread()
wk.moveToThread(th)
th.started.connect(wk.run)
wk.progress.connect(pg._on_progress)                                    # app.py 现行连接方式
wk.progress.connect(lambda v, s: res["lambda"].append(threading.get_ident() == main_tid))  # 缺陷对照
wk.done.connect(th.quit)
th.start()
t0 = QtCore.QElapsedTimer(); t0.start()
while th.isRunning() and t0.elapsed() < 10000:
    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
app.processEvents()
th.wait(2000)
dyn_ok = res["bound"] and all(res["bound"]) and res["lambda"] and not any(res["lambda"])
print(f"动态: 绑定@Slot方法落主线程={res['bound']} 对照裸lambda落主线程={res['lambda']}")

src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"), encoding="utf-8").read()
bad = re.findall(r"wk\.progress\.connect\(lambda", src)
good = "wk.progress.connect(self._on_progress)" in src
static_ok = not bad and good
print(f"静态: app.py 绑定方法连接存在={good} 裸lambda残留={len(bad)} 处")

if dyn_ok and static_ok:
    print("PASS"); sys.exit(0)
print("FAIL"); sys.exit(1)
