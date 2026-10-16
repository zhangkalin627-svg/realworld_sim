# -*- coding: utf-8 -*-
"""淬火过程动态回放组件：场快照 + 温度曲线随仿真时间逐步画出。
回放的是已算好的仿真结果（不是实时重算），画面与标题均标"仿真结果"。"""
import bisect
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from . import viz


class QuenchReplay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.sim = None
        self.tr = 0.0          # 回放时间 s
        self.playing = False
        lay = QtWidgets.QVBoxLayout(self)
        # ---- 控制条
        bar = QtWidgets.QHBoxLayout()
        self.btn = QtWidgets.QPushButton("▶ 播放"); self.btn.clicked.connect(self.toggle)
        self.btn.setEnabled(False)
        rb = QtWidgets.QPushButton("⟲ 重播"); rb.clicked.connect(self.replay)
        self.speed = QtWidgets.QComboBox(); self.speed.addItems(["0.5×", "1×", "2×", "4×", "8×"]); self.speed.setCurrentIndex(1)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, 1000)
        self.slider.sliderPressed.connect(self._pause)
        self.slider.sliderMoved.connect(self._seek)
        self.lab = QtWidgets.QLabel("先点「开始模拟」，算完这里自动回放。")
        bar.addWidget(self.btn); bar.addWidget(rb); bar.addWidget(QtWidgets.QLabel("速度")); bar.addWidget(self.speed)
        bar.addWidget(self.slider, 1); bar.addWidget(self.lab)
        lay.addLayout(bar)
        self.fig = Figure(figsize=(10, 4.6)); self.cv = FigureCanvasQTAgg(self.fig)
        lay.addWidget(self.cv, 1)
        self.timer = QtCore.QTimer(self); self.timer.setInterval(50); self.timer.timeout.connect(self._tick)
        self._clock = QtCore.QElapsedTimer()

    # ---------------------------------------------------------- 数据
    def set_data(self, sim):
        self.sim = sim
        self.tr = 0.0
        self.btn.setEnabled(bool(sim.frames))
        h = sim.hist
        self._ft = [f[0] for f in sim.frames]
        self._setup_axes()
        self._draw()
        self.replay()          # 算完自动播放一遍

    def _setup_axes(self):
        sim = self.sim
        self.fig.clear()
        self.ax_f, self.ax_c = self.fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.5]))
        # 左：温度场（tripcolor 建一次，之后只换数据）
        vmax = max(1000.0, float(sim.T_end_heat.max()))
        self.tc = viz.plot_field(self.ax_f, sim, sim.frames[0][1], "inferno", "°C", 25, vmax, self.fig)
        self.ax_f.set_title("温度场（仿真结果）", fontsize=10)
        # 右：温度曲线骨架（全程淡显 + 已回放段实显）
        h = sim.hist
        ax = self.ax_c
        ax.axvspan(0, sim.t_heat, color="#ffe0c0", alpha=0.5, lw=0)
        q0 = sim.t_heat + sim.t_delay
        ax.axvspan(q0, q0 + sim.t_quench, color="#cfe8ff", alpha=0.6, lw=0)
        for y, s in ((830, "Ac3"), (340, "Ms")):
            ax.axhline(y, ls="--", lw=0.8, color="gray")
            ax.text(h["t"][-1], y, s, ha="right", va="bottom", fontsize=8, color="gray")
        for k, c, n in [("Tsurf", "#d62728", "外表面"), ("Tcore", "#ff7f0e", "壁厚中心"), ("Tin", "#1f77b4", "内孔表面")]:
            ax.plot(h["t"], h[k], color=c, alpha=0.15, lw=0.8)
        self.l_s, = ax.plot([], [], color="#d62728", lw=1.6, label="外表面")
        self.l_m, = ax.plot([], [], color="#ff7f0e", lw=1.6, label="壁厚中心")
        self.l_i, = ax.plot([], [], color="#1f77b4", lw=1.6, label="内孔表面")
        self.vline = ax.axvline(0, color="#333", lw=1.0)
        self.dot_s, = ax.plot([], [], "o", color="#d62728", ms=5)
        self.dot_m, = ax.plot([], [], "o", color="#ff7f0e", ms=5)
        self.dot_i, = ax.plot([], [], "o", color="#1f77b4", ms=5)
        ax.set_xlim(0, h["t"][-1]); ax.set_ylim(0, max(1000, h["Tsurf"].max() * 1.05))
        ax.set_xlabel("时间 (s)"); ax.set_ylabel("温度 (°C)")
        ax.set_title("淬火区中部温度（橙=加热，蓝=淬火；仿真结果）", fontsize=10)
        ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.3)
        self.fig.tight_layout()

    # ---------------------------------------------------------- 播放控制
    def toggle(self):
        if not self.sim or not self.sim.frames:
            return
        if self.playing:
            self._pause()
        else:
            if self.tr >= self.sim.hist["t"][-1] - 1e-9:
                self.tr = 0.0
            self.playing = True
            self.btn.setText("⏸ 暂停")
            self._clock.start()
            self.timer.start()

    def _pause(self):
        self.playing = False
        self.btn.setText("▶ 播放")
        self.timer.stop()

    def replay(self):
        if not self.sim or not self.sim.frames:
            return
        self.tr = 0.0
        self._draw()
        if not self.playing:
            self.toggle()

    def _seek(self, v):
        if not self.sim:
            return
        self.tr = v / 1000.0 * self.sim.hist["t"][-1]
        self._draw()

    def _tick(self):
        sp = float(self.speed.currentText().rstrip("×"))
        self.tr += self._clock.restart() / 1000.0 * sp
        end = self.sim.hist["t"][-1]
        if self.tr >= end:
            self.tr = end
            self._draw()
            self._pause()
            return
        self._draw()

    # ---------------------------------------------------------- 画
    def _draw(self):
        sim = self.sim
        if sim is None:
            return
        tr = self.tr
        # 场：取 ≤tr 最近快照
        i = min(bisect.bisect_right(self._ft, tr), len(sim.frames)) - 1
        i = max(i, 0)
        self.tc.set_array(viz._mir(sim.frames[i][1]))
        # 曲线：已回放段
        h = sim.hist
        n = bisect.bisect_right(h["t"], tr)
        self.l_s.set_data(h["t"][:n], h["Tsurf"][:n])
        self.l_m.set_data(h["t"][:n], h["Tcore"][:n])
        self.l_i.set_data(h["t"][:n], h["Tin"][:n])
        if n:
            self.dot_s.set_data([h["t"][n-1]], [h["Tsurf"][n-1]])
            self.dot_m.set_data([h["t"][n-1]], [h["Tcore"][n-1]])
            self.dot_i.set_data([h["t"][n-1]], [h["Tin"][n-1]])
        self.vline.set_xdata([tr, tr])
        # 阶段标签
        if tr < sim.t_heat:
            ph = "加热"
        elif tr < sim.t_heat + sim.t_delay:
            ph = "延时（断电空冷）"
        elif tr < sim.t_heat + sim.t_delay + sim.t_quench:
            ph = f"淬火（{sim.medium}）"
        else:
            ph = "空冷"
        self.lab.setText(f"t = {tr:.2f} s · {ph} · 外表面 {h['Tsurf'][max(n-1,0)]:.0f}°C")
        self.slider.blockSignals(True)
        self.slider.setValue(int(tr / h["t"][-1] * 1000))
        self.slider.blockSignals(False)
        self.cv.draw_idle()
