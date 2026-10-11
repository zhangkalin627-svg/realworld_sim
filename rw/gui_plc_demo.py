"""PLC 编程演示页：自己写 FX 指令表 → 装载 → 带着设备模型实时运行。"""
import re
from PySide6 import QtCore, QtGui, QtWidgets
from .plc.demo import DEMO_IL, DemoSim
def glow(T):
    """温度 → 火色。"""
    if T < 450:
        g = int(140 - 60 * T / 450)
        return QtGui.QColor(g, g, g + 10)
    pts = [(450, (90, 20, 10)), (600, (150, 20, 5)), (750, (210, 50, 10)), (850, (250, 110, 20)),
           (950, (255, 180, 60)), (1100, (255, 240, 180))]
    for (t0, c0), (t1, c1) in zip(pts, pts[1:]):
        if T <= t1:
            f = (T - t0) / (t1 - t0)
            return QtGui.QColor(*[int(a + (b - a) * f) for a, b in zip(c0, c1)])
    return QtGui.QColor(255, 250, 230)




class DemoView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); self.sim = None; self.setMinimumSize(360, 420)

    def paintEvent(self, e):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor("#1e2329"))
        s = self.sim
        if not s:
            p.setPen(QtGui.QColor("#ccc")); p.drawText(self.rect(), QtCore.Qt.AlignCenter, "写好程序后点「装载并运行」"); return
        W, H = self.width(), self.height(); cx = W / 2; base = H * 0.78
        pl = s.plant
        # 工件
        part = pl.part
        if part:
            col = glow(part.T) if part.heat > 0 and part.quench < 0.3 else (QtGui.QColor("#4a5058") if part.quench >= 0.3 else QtGui.QColor("#8a8f96"))
            p.setBrush(col); p.setPen(QtGui.QPen(QtGui.QColor("#222"), 2))
            p.drawRect(QtCore.QRectF(cx - 30, base - 150, 60, 150))
            p.drawRect(QtCore.QRectF(cx - 15, base - 230, 30, 80))
            p.setPen(QtGui.QColor("#fff")); p.drawText(QtCore.QPointF(cx + 40, base - 20), f"工件 P{part.id}  {part.T:.0f}°C")
        p.setBrush(QtGui.QColor("#666")); p.setPen(QtCore.Qt.NoPen); p.drawRect(QtCore.QRectF(cx - 80, base, 160, 14))
        # 感应圈
        y = base - 330 + 110 * pl.pos
        hot = s.Y.get("Y001")
        p.setBrush(QtGui.QColor("#ffb000") if hot else QtGui.QColor("#c87533")); p.setPen(QtCore.Qt.NoPen)
        p.drawRect(QtCore.QRectF(cx - 42, y, 18, 30)); p.drawRect(QtCore.QRectF(cx + 24, y, 18, 30))
        p.setBrush(QtGui.QColor("#5a6470")); p.drawRect(QtCore.QRectF(cx + 42, y + 8, 120, 14))
        if s.Y.get("Y002"):
            p.setPen(QtGui.QPen(QtGui.QColor("#5ac8fa"), 2, QtCore.Qt.DotLine))
            for k in range(6):
                p.drawLine(QtCore.QPointF(cx - 50 + 20 * k, y + 32), QtCore.QPointF(cx - 40 + 16 * k, y + 70))
        p.setPen(QtGui.QColor("#fff"))
        w = s.plc.words
        p.drawText(10, 20, f"t = {s.t:6.1f} s   完成 D0={w.get('D0', 0)}   测温不合格 D1={w.get('D1', 0)}   气压 {pl.p_air:.2f} MPa")
        act = [d for d in sorted(s.plc.bits) if d.startswith("S") and s.plc.bits[d]]
        p.drawText(10, 40, "当前步: " + " ".join(act))


class PLCDemoPage(QtWidgets.QWidget):
    def __init__(self, thermal=None):
        super().__init__()
        self.th = thermal
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("三菱 FX 指令表（可以直接改，; 后面是注释）"))
        self.ed = QtWidgets.QPlainTextEdit(DEMO_IL)
        self.ed.setFont(QtGui.QFont("monospace", 10))
        left.addWidget(self.ed, 1)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("装载并运行"); b.clicked.connect(self.load); row.addWidget(b)
        b = QtWidgets.QPushButton("恢复示例程序"); b.clicked.connect(lambda: self.ed.setPlainText(DEMO_IL)); row.addWidget(b)
        left.addLayout(row)
        self.msg = QtWidgets.QLabel(""); self.msg.setWordWrap(True); left.addWidget(self.msg)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(430); lay.addWidget(lw)
        mid = QtWidgets.QVBoxLayout()
        self.view = DemoView(); mid.addWidget(self.view, 1)
        g = QtWidgets.QGridLayout()
        for k, (t, x) in enumerate([("启动 X000", "X000"), ("停止 X001", "X001")]):
            bb = QtWidgets.QPushButton(t); bb.clicked.connect(lambda _=False, x=x: self.sim and self.sim.press(x)); g.addWidget(bb, 0, k)
        self.es = QtWidgets.QPushButton("急停 X002"); self.es.setCheckable(True)
        self.es.toggled.connect(lambda v: self.sim and self.sim.plant.ext.__setitem__("X002", not v)); g.addWidget(self.es, 0, 2)
        g.addWidget(QtWidgets.QLabel("气压 MPa"), 1, 0)
        self.air = QtWidgets.QDoubleSpinBox(); self.air.setRange(0.1, 0.8); self.air.setSingleStep(0.05); self.air.setValue(0.5)
        self.air.valueChanged.connect(lambda v: self.sim and self.sim.plant.set_air(v)); g.addWidget(self.air, 1, 1)
        self.spd = QtWidgets.QComboBox(); self.spd.addItems(["1×", "3×", "10×"]); self.spd.setCurrentIndex(1); g.addWidget(self.spd, 1, 2)
        mid.addLayout(g)
        mw = QtWidgets.QWidget(); mw.setLayout(mid); lay.addWidget(mw, 1)
        self.io = QtWidgets.QLabel(); self.io.setAlignment(QtCore.Qt.AlignTop); self.io.setFixedWidth(170)
        self.io.setTextFormat(QtCore.Qt.RichText); lay.addWidget(self.io)
        self.sim = None
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(50)

    def load(self):
        try:
            self.sim = DemoSim(self.ed.toPlainText(), self.air.value(), self.th)
            self.view.sim = self.sim
            self.es.setChecked(False)
            devs = set(re.findall(r"\b([XY][0-7]{3})\b", self.ed.toPlainText()))
            self.devs = sorted(devs)
            self.msg.setText(f"装载成功：{len(self.sim.plc.prog)} 条指令。点「启动 X000」。")
        except Exception as e:
            self.sim = None; self.view.sim = None
            self.msg.setText(f"<span style='color:#c62828'>程序有错：{e}</span>")

    def tick(self):
        s = self.sim
        if not s:
            return
        try:
            for _ in range([1, 3, 10][self.spd.currentIndex()] * 5):
                s.scan()
        except Exception as e:
            self.msg.setText(f"<span style='color:#c62828'>运行出错：{e}</span>"); self.sim = None; return
        inp = s.plant.inputs(s.Y)
        rows = []
        for d in self.devs:
            on = inp.get(d, False) if d[0] == "X" else s.plc.get(d)
            rows.append(f"<div style='background:{'#2ecc40' if on else 'transparent'};color:{'black' if on else '#888'}'>{d}</div>")
        self.io.setText("".join(rows))
        self.view.update()
