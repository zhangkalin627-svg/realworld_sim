"""⑨ 三维装配世界：摆零件 → 重力/碰撞/摩擦下的刚体物理 → 稳定、压强、梁应力体检。"""
import html as _h
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from . import viz  # noqa: F401  注册中文字体
from .world3d.scene import Scene, MATS, KINDS
from .world3d.presets import PRESETS
from .world3d.physics import simulate, check, float_check
from .world3d.render import SceneRenderer, Camera

COLOR = {"OK": "#1a7f37", "WARN": "#b26a00", "FAIL": "#c62828"}
COLS = ["名称", "类型", "尺寸 mm", "材料", "X", "Y", "Z", "转角 rx,ry,rz°", "固定", "组"]
KIND_ZH = {k: v["zh"] for k, v in KINDS.items()}
ZH_KIND = {v: k for k, v in KIND_ZH.items()}


class View(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(520, 380)
        self.cam = Camera(); self.rend = None; self.frame = None; self.mark = {}
        self._drag = None

    def set_scene(self, scene, keep_cam=False):
        self.rend = SceneRenderer(scene) if scene.parts else None
        if self.rend and not keep_cam:
            self.cam = self.rend.auto_camera()
        self.frame = None; self.mark = {}; self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        if self.rend is None:
            p.fillRect(self.rect(), QtGui.QColor(246, 247, 250)); p.drawText(20, 30, "（场景为空）")
        else:
            self.rend.paint(p, self.width(), self.height(), self.cam, self.frame, self.mark)

    def mousePressEvent(self, e): self._drag = (e.position(), e.button())
    def mouseReleaseEvent(self, e): self._drag = None

    def mouseMoveEvent(self, e):
        if self._drag is None: return
        d = e.position() - self._drag[0]; self._drag = (e.position(), self._drag[1])
        if self._drag[1] == QtCore.Qt.LeftButton:
            self.cam.az -= d.x() * 0.5; self.cam.el = float(np.clip(self.cam.el + d.y() * 0.5, -5, 89))
        else:
            _, right, up, _ = self.cam.basis()
            k = self.cam.dist / 900
            self.cam.target = self.cam.target - right * d.x() * k + up * d.y() * k
        self.update()

    def wheelEvent(self, e):
        self.cam.dist *= 0.9 ** (e.angleDelta().y() / 120); self.update()


class World3DPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.scene = Scene(); self.res = None; self.chk = None
        lay = QtWidgets.QHBoxLayout(self)
        # ---- 左：零件表
        left = QtWidgets.QVBoxLayout()
        self.preset = QtWidgets.QComboBox(); self.preset.addItems(list(PRESETS))
        self.preset.activated.connect(self.load_preset)
        left.addWidget(QtWidgets.QLabel("示例场景（也可以自己加零件）：")); left.addWidget(self.preset)
        self.tbl = QtWidgets.QTableWidget(0, len(COLS)); self.tbl.setHorizontalHeaderLabels(COLS)
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tbl.itemChanged.connect(lambda *_: self._dirty())
        left.addWidget(self.tbl, 1)
        hb = QtWidgets.QHBoxLayout()
        for txt, fn in (("＋ 加零件", self.add_row), ("－ 删选中", self.del_row), ("刷新画面", self.refresh_view)):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); hb.addWidget(b)
        left.addLayout(hb)
        self.btn = QtWidgets.QPushButton("▶ 放手，让重力起作用（物理仿真 + 体检）"); self.btn.clicked.connect(self.run)
        self.bbreak = QtWidgets.QPushButton("💥 演示断裂后果（把断的梁分成两段再仿真）"); self.bbreak.clicked.connect(self.run_break); self.bbreak.setVisible(False)
        self.btn.setStyleSheet("font-weight:bold;padding:6px")
        left.addWidget(self.btn); left.addWidget(self.bbreak)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(600)
        lay.addWidget(lw)
        # ---- 右：视图 + 结果
        right = QtWidgets.QVBoxLayout()
        self.view = View(); right.addWidget(self.view, 3)
        ctl = QtWidgets.QHBoxLayout()
        self.play = QtWidgets.QPushButton("播放"); self.play.clicked.connect(self.toggle_play); ctl.addWidget(self.play)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.set_frame); ctl.addWidget(self.slider, 1)
        self.tlabel = QtWidgets.QLabel("t = 0.00 s"); ctl.addWidget(self.tlabel)
        right.addLayout(ctl)
        self.tabs = QtWidgets.QTabWidget()
        self.out = QtWidgets.QTextBrowser(); self.tabs.addTab(self.out, "体检结果")
        bw = QtWidgets.QWidget(); bl = QtWidgets.QVBoxLayout(bw)
        self.bsel = QtWidgets.QComboBox(); self.bsel.currentIndexChanged.connect(self.plot_beam); bl.addWidget(self.bsel)
        self.fig = Figure(figsize=(6, 3)); self.cv = FigureCanvasQTAgg(self.fig); bl.addWidget(self.cv)
        self.tabs.addTab(bw, "梁的弯矩/挠度")
        right.addWidget(self.tabs, 2)
        lay.addLayout(right, 1)
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.step_play)
        self._loading = False
        self.load_preset(0)

    # ---------------------------------------------------------- 表 <-> 场景
    def fill_table(self):
        self._loading = True
        self.tbl.setRowCount(0)
        for p in self.scene.parts:
            r = self.tbl.rowCount(); self.tbl.insertRow(r)
            vals = [p.name, KIND_ZH[p.kind], ",".join(f"{d:g}" for d in p.dims), None, f"{p.pos[0]:g}", f"{p.pos[1]:g}", f"{p.pos[2]:g}",
                    ",".join(f"{x:g}" for x in p.rot), None, p.group]
            for c, v in enumerate(vals):
                if v is not None: self.tbl.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
            cb = QtWidgets.QComboBox()
            for k, m in MATS.items(): cb.addItem(m["name"], k)
            cb.setCurrentIndex(cb.findData(p.mat)); cb.currentIndexChanged.connect(lambda *_: self._dirty())
            self.tbl.setCellWidget(r, 3, cb)
            fx = QtWidgets.QCheckBox(); fx.setChecked(p.fixed); fx.toggled.connect(lambda *_: self._dirty())
            self.tbl.setCellWidget(r, 8, fx)
        self._loading = False

    def read_table(self):
        s = Scene(self.scene.name); s.fluid = self.scene.fluid; s.floor_mat = self.scene.floor_mat
        for r in range(self.tbl.rowCount()):
            g = lambda c: (self.tbl.item(r, c).text() if self.tbl.item(r, c) else "")
            kind = ZH_KIND.get(g(1), g(1))
            dims = [float(x) for x in g(2).replace("，", ",").split(",") if x.strip()]
            need = len(KINDS[kind]["dims"])
            if len(dims) < need: raise ValueError(f"第 {r+1} 行尺寸需要 {need} 个数：{KINDS[kind]['dims']}")
            rot = [float(x) for x in g(7).replace("，", ",").split(",") if x.strip()] or [0, 0, 0]
            rot = (rot + [0, 0, 0])[:3]
            s.add(g(0) or f"件{r+1}", kind, dims[:need], self.tbl.cellWidget(r, 3).currentData(),
                  pos=[float(g(4) or 0), float(g(5) or 0), float(g(6) or 0)], rot=rot,
                  fixed=self.tbl.cellWidget(r, 8).isChecked(), group=g(9).strip())
        return s

    def _dirty(self):
        if self._loading: return
        self.res = None

    def refresh_view(self):
        try:
            self.scene = self.read_table()
        except Exception as e:
            self.out.setHtml(f'<span style="color:#c62828">表格有误：{_h.escape(str(e))}</span>'); return
        self.view.set_scene(self.scene, keep_cam=True)
        self.slider.setEnabled(False); self.res = None

    def add_row(self):
        p = self.scene.add(f"件{len(self.scene.parts)+1}", "box", [200, 200, 200], "Q235", pos=[0, 0, 100 + 210 * len(self.scene.parts)])
        self.fill_table(); self.refresh_view()

    def del_row(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        for r in rows: self.tbl.removeRow(r)
        self.refresh_view()

    def load_preset(self, i):
        self.scene = list(PRESETS.values())[i]()
        self.fill_table(); self.view.set_scene(self.scene); self.res = None
        self.slider.setEnabled(False); self.slider.setValue(0)
        m, c = self.scene.total_mass()
        fl = float_check(self.scene) if self.scene.fluid else []
        self.out.setHtml(f"<b>{_h.escape(self.scene.name)}</b>：{len(self.scene.parts)} 个零件，总重 {m:.1f} kg。点「放手」开始仿真。"
                         + (self.float_html(fl) if fl else ""))

    # ---------------------------------------------------------- 运行
    # 仿真挪到工作线程（v0.17，测试反馈窗口操作 segfault 排查：本页是唯一在界面线程跑重计算的页，
    # 仿真期间界面冻结，此时拖拽窗口/双击标题栏会触发 Qt 绘制异常；线程化后界面始终可响应）
    def _spawn(self, job, on_done):
        import traceback as _tb
        class _J(QtCore.QObject):
            done = QtCore.Signal(object)
            failed = QtCore.Signal(str)
            @QtCore.Slot()
            def run(self):
                try: self.done.emit(job())
                except Exception: self.failed.emit(_tb.format_exc())
        if getattr(self, "_th3", None) is not None and self._th3.isRunning():
            raise RuntimeError("仿真仍在运行")
        self._on_done3 = on_done
        self._th3 = QtCore.QThread(self); self._j3 = _J()
        self._j3.moveToThread(self._th3)
        self._th3.started.connect(self._j3.run)
        # A Python closure is not a QObject receiver: use this GUI-thread slot.
        self._j3.done.connect(self._dispatch_done, QtCore.Qt.QueuedConnection)
        self._j3.failed.connect(self._dispatch_error, QtCore.Qt.QueuedConnection)
        self._j3.done.connect(self._th3.quit); self._j3.failed.connect(self._th3.quit)
        self._j3.done.connect(self._j3.deleteLater)
        self._j3.failed.connect(self._j3.deleteLater)
        self._th3.start()

    @QtCore.Slot(object)
    def _dispatch_done(self, result):
        self._on_done3(result)

    @QtCore.Slot(str)
    def _dispatch_error(self, tb):
        self._sim_err(tb)

    def _sim_err(self, tb):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.btn.setEnabled(True); self.bbreak.setEnabled(True)
        self.out.setHtml(f'<span style="color:#c62828">仿真出错：<pre>{_h.escape(tb)}</pre></span>')

    def _after_sim(self):
        n = len(self.res.frames)
        self.slider.setEnabled(True); self.slider.setMaximum(max(0, n - 1)); self.slider.setValue(n - 1)
        self.show_report()
        self.bsel.clear()
        for b in self.chk["beams"]: self.bsel.addItem(b["name"])
        self.plot_beam()

    def run(self):
        try:
            self.scene = self.read_table()
        except Exception as e:
            self.out.setHtml(f'<span style="color:#c62828">表格有误：{_h.escape(str(e))}</span>'); return
        if not self.scene.parts: return
        self.view.set_scene(self.scene, keep_cam=True)
        self.btn.setEnabled(False); self.bbreak.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        sc = self.scene
        def job():
            res = simulate(sc, tmax=8.0)
            return res, check(res)
        def done(rc):
            QtWidgets.QApplication.restoreOverrideCursor()
            self.btn.setEnabled(True)
            self.res, self.chk = rc
            self.bbreak.setEnabled(True); self.bbreak.setVisible(bool(self.chk.get("broken")))
            self._after_sim()
        self._spawn(job, done)

    def run_break(self):
        """断裂后果：断口分裂 → 再仿真 → 再体检（v0.13）。"""
        from .world3d.fracture import break_and_rerun
        if not self.chk or not self.chk.get("broken"): return
        self.btn.setEnabled(False); self.bbreak.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        res, chk = self.res, self.chk
        def done(out):
            QtWidgets.QApplication.restoreOverrideCursor()
            self.btn.setEnabled(True)
            if out is None: return
            self.scene, self.res, self.chk = out
            self.view.set_scene(self.scene, keep_cam=True)
            self.bbreak.setVisible(False)
            self._after_sim()
        self._spawn(lambda: break_and_rerun(res, chk, tmax=6.0), done)

    def float_html(self, fl):
        rows = "".join(f'<li><b>{_h.escape(f["name"])}</b>：密度 {f["rho_part"]:.0f} vs 液体 {f["rho_fluid"]:.0f} kg/m³ → '
                       + (f'<span style="color:#1a7f37">浮，吃水约 {f["draft"]*100:.0f}%</span>' if f["floats"] else
                          f'<span style="color:#c62828">沉，净重 {f["net_N"]:.1f} N</span>') + "</li>" for f in fl)
        return f"<p><b>液体中（阿基米德，静态；实心件）：</b></p><ul>{rows}</ul>"

    def show_report(self):
        c = self.chk; r = self.res
        parts = []
        bad = c["unstable"]
        verdict = "FAIL" if bad else "OK"
        mark = {}
        for s_ in c["stability"]:
            if s_["status"] == "倾倒/滑落":
                for nm in s_["body"].split(","): mark[nm] = (230, 70, 70)
        beams = c["beams"]
        for b in beams:
            st = b.get("status", "正常" if b["ok"] else "断裂")
            if st == "断裂" or b.get("ok_dyn") is False:
                mark[b["name"]] = (230, 70, 70); verdict = "FAIL" if st == "断裂" else verdict
                if verdict == "OK": verdict = "WARN"
            elif st == "屈服" or not b["ok"]:
                mark[b["name"]] = (230, 120, 40)
                if verdict == "OK": verdict = "WARN"
        for j in c.get("bolts", []):
            if not j["ok"] and verdict == "OK": verdict = "WARN"
        self.view.mark = mark
        head = {"OK": "全部稳定，受力在材料极限内", "WARN": "稳定，但有构件应力（含冲击）超限或连接件利用率偏高", "FAIL": "有物体倾倒/滑落/断裂"}[verdict]
        parts.append(f'<h3 style="color:{COLOR[verdict]}">{head}</h3>')
        parts.append(f"<p>仿真 {r.times[-1]:.2f} s" + (f"，约 {r.t_settle:.2f} s 后静止" if r.settled else "，<b>到时间仍未静止（在滚动/滑动）</b>") + "。</p>")
        parts.append("<p><b>稳定性：</b></p><ul>")
        for s_ in c["stability"]:
            col = {"稳定": "OK", "铰接": "OK", "有位移/下落": "WARN", "倾倒/滑落": "FAIL"}[s_["status"]]
            parts.append(f'<li><span style="color:{COLOR[col]}"><b>{_h.escape(s_["body"])}</b>：{s_["status"]}</span>'
                         f'（水平位移 {s_["disp_mm"]:.1f} mm，下落 {-s_["dz_mm"]:.1f} mm，倾转 {s_["tilt_deg"]:.1f}°）</li>')
        parts.append("</ul>")
        if c["floor"]:
            m, _ = self.scene.total_mass()
            fixed_m = sum(p.mass for p in self.scene.parts if p.fixed)
            tail = f"（自由件总重 {(m-fixed_m)*9.81:.0f} N，两者应相等）" if fixed_m == 0 else "（部分重量由固定的墩/底座承担）"
            parts.append(f'<p>地面直接承受 {c["floor"]["total_N"]:.0f} N{tail}。</p>')
        bad_b = [b for b in c["bearing"] if not b["ok"]]
        parts.append("<p><b>接触压强：</b>" + (f'<span style="color:{COLOR["FAIL"]}">{len(bad_b)} 处超过材料强度</span>' if bad_b else
                     f'<span style="color:{COLOR["OK"]}">都在材料屈服/蠕变极限内</span>') + "</p><ul>")
        for b in sorted(c["bearing"], key=lambda x: -x["p"] / x["lim"])[:6]:
            col = "OK" if b["ok"] else "FAIL"
            note = "" if b["reliable"] else "（点/线接触，面积估不准，仅供参考）"
            parts.append(f'<li><span style="color:{COLOR[col]}">{_h.escape(b["pair"])}：{b["F"]:.0f} N，面积≈{b["area"]:.0f} mm²，{b["p"]:.3f} MPa / 极限 {b["lim"]:.0f} MPa</span>{note}</li>')
        parts.append("</ul>")
        if beams:
            parts.append("<p><b>梁/板弯曲（接触力 + 自重，沿长轴一维 Euler-Bernoulli）：</b></p><ul>")
            for b in beams:
                col = "OK" if b["ok"] else "FAIL"
                dyn = ""
                if "n_dyn" in b:
                    c2 = "OK" if b["ok_dyn"] else "FAIL"
                    dyn = f'；<span style="color:{COLOR[c2]}">从 {b["drop_h"]:.0f} mm 高处落下的冲击（放大 ×{b["n_dyn"]:.0f}，保守上界）σ≈{b["sigma_dyn"]:.0f} MPa</span>'
                st = b.get("status", "正常" if b["ok"] else "断裂")
                col = {"正常": "OK", "屈服": "WARN", "断裂": "FAIL"}[st]
                sec = f'，{b["section"]}' if b.get("section") else ""
                parts.append(f'<li><span style="color:{COLOR[col]}"><b>{_h.escape(b["name"])}</b>：跨距 {b["span"]:.0f} mm，最大弯矩 {b["Mmax"]/1e3:.1f} N·m，'
                             f'σ={b["sigma"]:.2f} MPa / 屈服 {b["fy"]:.0f} / 抗拉 {b.get("fu", b["fy"]):.0f} → <b>{st}</b>，挠度 {b["defl"]:.3f} mm（L/{b["defl_ratio"]:.0f}）{sec}</span>{dyn}</li>')
            parts.append("</ul>")
        for j in c.get("bolts", []):
            col = "OK" if j["ok"] else "FAIL"
            if j.get("weld"):
                parts.append(f'<p><b>焊缝群（{_h.escape(j["part"])}，角焊缝 hf={j["hf"]} mm，ΣL={j["L"]:.0f} mm）：</b>'
                             f'<span style="color:{COLOR[col]}">接头力 {np.linalg.norm(j["F"]):.0f} N、力矩 {np.linalg.norm(j["M"])/1e3:.2f} N·m；'
                             f'σ={j["sig"]+j["sig_b"]:.1f} MPa，τ={j["tau"]:.1f} MPa，利用率 {j["util"]:.3f}</span>'
                             f'（简化弹性法，喉厚 0.7hf；粗估，不含应力集中与焊根缺陷）</p>')
                continue
            bt = max(j["bolts"], key=lambda x: -x["T"]); bv = max(j["bolts"], key=lambda x: -x["V"])
            parts.append(f'<p><b>螺栓组（{_h.escape(j["part"])}，{len(j["bolts"])}×M{j["size"]} {j["grade"]}）：</b>'
                         f'<span style="color:{COLOR[col]}">接头力 {np.linalg.norm(j["F"]):.0f} N、力矩 {np.linalg.norm(j["M"])/1e3:.2f} N·m；'
                         f'最大拉 {bt["T"]:.0f} N（{bt["sig"]:.0f} MPa）/ 最大剪 {bv["V"]:.0f} N（{bv["tau"]:.0f} MPa），'
                         f'利用率 {j["worst"]:.3f}</span>（bearing 型弹性分配；不含撬力放大与预紧）</p>')
        for h in c.get("hinges", []):
            life = h.get("life")
            if life and "L10h" in life:
                ltxt = f'L10 寿命 ≈ {life["L10h"]:.0f} h'
            elif life:
                ltxt = f'静安全 S0 = {life["S0"]:.1f}'
            else:
                ltxt = "未选轴承型号"
            parts.append(f'<p><b>轴承/铰链（{_h.escape(h["part"])}，{h.get("bearing") or "—"}）：</b>'
                         f'反力 {h["F_radial"]:.0f} N，不平衡力幅值 {h["F_unbalance"]:.0f} N，P={h["P"]:.0f} N，{ltxt}</p>')
        for hz in c.get("hertz", []):
            a_b = f'a={hz["a"]:.2f} mm' if "a" in hz else f'b={hz["b"]:.2f} mm'
            parts.append(f'<p><b>赫兹{hz["kind"]}（{_h.escape(hz["part"])}↔{_h.escape(str(hz["with_"]))}）：</b>'
                         f'F={hz["F"]:.0f} N，p0={hz["p0"]:.0f} MPa，{a_b}（{hz["note"]}）</p>')
        if self.scene.fluid:
            parts.append(self.float_html(float_check(self.scene)))
        parts.append('<p style="color:#666">说明：摩擦系数按两种材料中较小者取典型值（估算）；面接触压强的面积由接触点包络估算（球/圆柱另有赫兹应力）；'
                     '型材按尖角分解（忽略圆角，质量偏轻 ~1-5%）；螺栓组为 bearing 型弹性分配（不含撬力放大、预紧与被连接件变形）；'
                     '轴承额定为典型样本值；断裂判据按 σ>抗拉强度（不含裂纹扩展与碎裂动画）；仿真未含浮力/空气阻力。</p>')
        self.out.setHtml("".join(parts))
        self.set_frame(self.slider.value())

    def plot_beam(self):
        self.fig.clear()
        if not self.chk or not self.chk["beams"] or self.bsel.currentIndex() < 0:
            self.cv.draw(); return
        b = self.chk["beams"][self.bsel.currentIndex()]
        ax = self.fig.add_subplot(211); ax.plot(b["s_grid"], b["M"] / 1e3, color="#1565c0")
        ax.axhline(0, color="#999", lw=0.6); ax.set_ylabel("弯矩 N·m"); ax.set_title(f'{b["name"]}  (支点 {b["s0"]:.0f} / {b["s1"]:.0f} mm)', fontsize=9)
        for x in (b["s0"], b["s1"]): ax.axvline(x, color="#c62828", ls=":", lw=1)
        ax2 = self.fig.add_subplot(212); ax2.plot(b["s_grid"], -b["w"], color="#2e7d32")
        ax2.set_ylabel("下垂 mm"); ax2.set_xlabel("沿长轴位置 mm"); ax2.invert_yaxis()
        for x in (b["s0"], b["s1"]): ax2.axvline(x, color="#c62828", ls=":", lw=1)
        self.fig.tight_layout(); self.cv.draw()

    # ---------------------------------------------------------- 回放
    def set_frame(self, i):
        if self.res is None: return
        i = int(np.clip(i, 0, len(self.res.frames) - 1))
        self.view.frame = self.res.frames[i]
        self.tlabel.setText(f"t = {self.res.times[i]:.2f} s"); self.view.update()

    def toggle_play(self):
        if self.timer.isActive(): self.timer.stop(); self.play.setText("播放"); return
        if self.res is None: return
        if self.slider.value() >= self.slider.maximum(): self.slider.setValue(0)
        self.timer.start(40); self.play.setText("暂停")

    def step_play(self):
        v = self.slider.value() + 1
        if v > self.slider.maximum(): self.timer.stop(); self.play.setText("播放"); return
        self.slider.setValue(v)
