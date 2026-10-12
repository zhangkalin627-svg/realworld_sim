# -*- coding: utf-8 -*-
"""MCU ↔ FX PLC ↔ 电路 联合仿真（统一时间轴 = MCU 时钟周期；外部事件按时间排序放在一个堆里）。

时间模型（全部可在构造参数里改）：
- MCU：16 MHz，指令按周期数推进；睡眠时直接跳到下一外设事件（定时器/ADC/UART）或下一外部事件。
- PLC：固定扫描周期 scan_ms（相当于 FX3U 恒定扫描 D8039）；扫描开始读输入映像，扫描结束（END）刷新输出。
- PLC 输入：X 端子 → 输入滤波 x_filter_ms（FX3U 默认 10 ms，D8020 可改）→ 输入映像。按纯延时建模。
- PLC 输出：晶体管输出 ON/OFF 响应 y_delay_ms（按 0.2 ms 假设）→ MCU 引脚。
- 电路：plant.HalfBridgeRL，PWM 边沿精确生效；ADC 在采样时刻读放大器输出。
"""
import bisect, heapq, math
import numpy as np

from .asm import assemble
from .avr import AVR
from .plant import HalfBridgeRL
from . import firmware as FW
from ..plc.fx3u import FX3U


def level_at(hist, t, default=0):
    """hist: [(t, level)] 按时间排序 → t 时刻的电平。"""
    if not hist:
        return default
    k = bisect.bisect_right([h[0] for h in hist], t) - 1
    return hist[k][1] if k >= 0 else default


def edges_on_grid(hist, t, default=0):
    tt = np.array([h[0] for h in hist]) if hist else np.array([])
    vv = np.array([h[1] for h in hist]) if hist else np.array([])
    k = np.searchsorted(tt, t, side="right") - 1
    out = np.where(k >= 0, vv[np.clip(k, 0, max(len(vv) - 1, 0))] if len(vv) else default, default)
    return out.astype(float)


class Waves:
    """给 v0.6 示波器（Scope.load）用：w.x 时间，w['名字'] 取信号，w.nets() 列名字。"""

    def __init__(self, t, sig, xname="time", plotname="联合仿真"):
        self.x = np.asarray(t, float); self.sig = sig; self.xname = xname; self.plotname = plotname

    def nets(self):
        return list(self.sig)

    def __getitem__(self, k):
        k = k.strip()
        if k in self.sig:
            return self.sig[k]
        for i in range(1, len(k)):
            if k[i] == "-" and k[:i] in self.sig and k[i + 1:] in self.sig:
                return self.sig[k[:i]] - self.sig[k[i + 1:]]
        raise KeyError(f"没有信号 {k}；可选：{', '.join(self.sig)}")


class CoSim:
    X_FROM_MCU = {"X002": "PD4", "X003": "PD7", "X004": "PB0"}
    Y_TO_MCU = {"Y000": "PD2", "Y003": "PD3"}
    BUTTONS = {"X000": 0, "X001": 1, "X005": 0}      # 启动(常开)  停止(常闭)  复位(常开)
    Y_LIST = ["Y000", "Y001", "Y002", "Y003"]

    def __init__(self, asm=FW.HEATER_ASM, plc_il=FW.PLC_IL, plant=None, f_cpu=16e6, scan_ms=2,
                 x_filter_ms=10.0, y_delay_ms=0.2):
        if hasattr(asm, "handlers"):                 # Python 行为级固件
            from .behav import PyMCU
            self.prog = None
            self.mcu = PyMCU(asm, f_cpu=f_cpu)
        else:
            self.prog = assemble(asm)
            self.mcu = AVR(self.prog, f_cpu=f_cpu)
        self.plant = plant or HalfBridgeRL()
        self.plc_il = plc_il
        self.plc = FX3U(plc_il.splitlines(), scan_ms=scan_ms)
        self.scan = scan_ms / 1e3; self.xf = x_filter_ms / 1e3; self.yd = y_delay_ms / 1e3
        self.q = []; self.seq = 0; self._tnow = None
        self.mcu.listeners.append(self._on_pin)
        self.mcu.analog[0] = self.plant.v_sense
        self.mcu.analog[1] = lambda t: self.plant.Vbus / 11.0      # 母线电压分压 1/11
        self.btn_hist = {x: [(0.0, v)] for x, v in self.BUTTONS.items()}
        self.y_hist = {y: [(0.0, 0)] for y in self.Y_LIST}          # PLC 输出端子（END 刷新时刻）
        self.y_last = {y: 0 for y in self.Y_LIST}
        self.mcu_in_hist = {p: [(0.0, 0)] for p in self.Y_TO_MCU.values()}   # MCU 实际看到的（含输出延时）
        self.ximg_hist = {}                                         # PLC 输入映像（扫描时刻）
        self.events_log = []                                        # (t, 文字)
        self.load_hist = [(0.0, self.plant.R0)]
        for p in self.Y_TO_MCU.values():
            self.mcu.set_input(p, 0)
        import re as _re
        self.plc_devs = sorted({a[0] for op, a in self.plc.prog if a and _re.fullmatch(r"[XYMSTC]\d+", a[0])})
        self.plc_snap = [(0.0, {d: False for d in self.plc_devs})]     # 每次扫描后（有变化才记）的软元件状态
        self.at(0.0, self._scan, "PLC 扫描")

    # ---------------------------------------------------------- 事件
    def at(self, t, fn, label=""):
        heapq.heappush(self.q, (self.mcu.cycles(t), self.seq, fn, label)); self.seq += 1

    def now(self):
        return self._tnow if self._tnow is not None else self.mcu.t()

    def press(self, x, t, dur=0.05):
        """按钮：t 时刻按下，dur 秒后松开（常闭按钮 = 断开）。"""
        nc = self.BUTTONS[x] == 1
        self.at(t, lambda: self._btn(x, 0 if nc else 1), f"{x} 按下")
        self.at(t + dur, lambda: self._btn(x, 1 if nc else 0), f"{x} 松开")

    def _btn(self, x, v):
        self.btn_hist[x].append((self.now(), v))
        self.events_log.append((self.now(), f"按钮 {x} → {v}"))

    def inject_fault(self, t, R):
        def f():
            self.plant.set_load(self.now(), R)
            self.load_hist.append((self.now(), R))
            self.events_log.append((self.now(), f"负载短路：R → {R} Ω"))
        self.at(t, f, "注入故障")

    def _on_pin(self, name, level, t, is_out):
        self.plant.on_pin(name, level, t, is_out)

    def _x_level(self, x, t):
        """t 时刻 PLC 输入映像看到的 X（输入滤波 = 纯延时）。"""
        ts = t - self.xf + 1e-12          # 防止浮点把“正好 10 ms”算成差一点
        if x in self.btn_hist:
            return level_at(self.btn_hist[x], ts, 0)
        pin = self.X_FROM_MCU.get(x)
        if pin:
            hist = [(c / self.mcu.f, v) for c, v in self.mcu.pin_log.get(pin, [])]
            return level_at(hist, ts, 0)
        return 0

    def _scan(self):
        t = self.now()
        inputs = {x: self._x_level(x, t) for x in list(self.BUTTONS) + list(self.X_FROM_MCU)}
        for x, v in inputs.items():
            h = self.ximg_hist.setdefault(x, [(0.0, 0)])
            if h[-1][1] != v:
                h.append((t, v))
        self.plc.scan(inputs)
        snap = {d: bool(self.plc.get(d)) for d in self.plc_devs}
        if snap != self.plc_snap[-1][1]:
            self.plc_snap.append((t + self.scan, snap))
        t_out = t + self.scan
        for y in self.Y_LIST:
            v = int(self.plc.get(y))
            if v != self.y_last[y]:
                self.y_last[y] = v
                self.at(t_out, (lambda y=y, v=v: self._y_out(y, v)), f"{y} 输出刷新")
        self.at(t + self.scan, self._scan, "PLC 扫描")

    def _y_out(self, y, v):
        self.y_hist[y].append((self.now(), v))
        self.events_log.append((self.now(), f"PLC {y} → {v}"))
        pin = self.Y_TO_MCU.get(y)
        if pin:
            self.at(self.now() + self.yd, lambda: self._mcu_in(pin, v), f"{pin} ← {y}")

    def _mcu_in(self, pin, v):
        self.mcu.set_input(pin, v)
        self.mcu_in_hist[pin].append((self.now(), v))

    def run(self, t_end):
        end = self.mcu.cycles(t_end)
        while True:
            nxt = self.q[0][0] if self.q else end
            stop = min(nxt, end)
            self.mcu.run_until(stop)
            if self.q and self.q[0][0] <= end and self.q[0][0] <= self.mcu.cyc:
                c, _, fn, _ = heapq.heappop(self.q)
                self._tnow = c / self.mcu.f          # 外部事件按它的标称时刻记录
                fn()
                self._tnow = None
            elif stop >= end:
                break

    # ---------------------------------------------------------- 结果
    def pin_hist(self, pin):
        return [(c / self.mcu.f, v) for c, v in self.mcu.pin_log.get(pin, [])]

    def uart_text(self):
        return [(e / self.mcu.f, chr(b)) for s, e, b in self.mcu.uart.tx_log]

    def waves(self, t0, t1, n=4000):
        t = np.linspace(t0, t1, n)
        i, vsw = self.plant.waveform(t)
        sig = {"i_load": i, "v_sense": np.clip(i * self.plant.Rsh * self.plant.gain, 0, self.plant.v_rail), "sw": vsw}
        for pin, nm in (("PD5", "PWM_IN"), ("PD6", "EN"), ("PD7", "MCU_故障"), ("PB0", "MCU_到达"), ("PD4", "MCU_就绪"), ("PD1", "TXD")):
            sig[nm] = 5.0 * edges_on_grid(self.pin_hist(pin), t)
        for y in self.Y_LIST:
            sig["PLC_" + y] = 24.0 * edges_on_grid(self.y_hist[y], t)
        for x in ("X000", "X003", "X004"):
            sig["PLC_" + x] = 24.0 * edges_on_grid(self.ximg_hist.get(x, [(0.0, 0)]), t)
        return Waves(t, sig)

    def adc_samples(self):
        f = self.mcu.f
        return [(s / f, d / f, ch, v, code) for (st, s, d, ch, v, vr, code) in self.mcu.adc.log]

    # ---------------------------------------------------------- 关键时刻
    def first_edge(self, hist, level, after=0.0):
        for t, v in hist:
            if t > after and v == level:
                return t
        return None

    def summary(self, i_th=None):
        """从仿真记录里量出握手和保护的关键时间（全部是仿真量出来的，不是设定值）。"""
        P = self.plant
        if i_th is None:
            i_th = 328 * 5.0 / 1024 * P.amps_per_volt()
        s = {"i_th": i_th}
        x0 = self.first_edge(self.btn_hist["X000"], 1)
        s["t_btn"] = x0
        s["t_X000_img"] = self.first_edge(self.ximg_hist.get("X000", []), 1)
        s["t_Y000_on"] = self.first_edge(self.y_hist["Y000"], 1)
        s["t_PD2_on"] = self.first_edge(self.mcu_in_hist["PD2"], 1)
        s["t_EN_on"] = self.first_edge(self.pin_hist("PD6"), 1)
        s["t_reached"] = self.first_edge(self.pin_hist("PB0"), 1)
        fl = [t for t, R in self.load_hist[1:]]
        if fl:
            tf = fl[0]; s["t_fault"] = tf
            # 电流第一次超过阈值的时刻（解析波形上 50 ns 分辨率）
            tt = np.arange(tf, tf + 2e-3, 50e-9); ii, _ = P.waveform(tt)
            k = np.nonzero(ii >= i_th)[0]
            s["t_cross"] = float(tt[k[0]]) if len(k) else None
            s["t_EN_off"] = self.first_edge(self.pin_hist("PD6"), 0, after=tf)
            s["t_PD7_on"] = self.first_edge(self.pin_hist("PD7"), 1, after=tf)
            s["t_X003_img"] = self.first_edge(self.ximg_hist.get("X003", []), 1, after=tf)
            s["t_Y000_off"] = self.first_edge(self.y_hist["Y000"], 0, after=tf)
            s["t_Y001_on"] = self.first_edge(self.y_hist["Y001"], 1, after=tf)
            s["t_PD2_off"] = self.first_edge(self.mcu_in_hist["PD2"], 0, after=tf)
            over = [(ts, code) for (ts, td, ch, v, code) in self.adc_samples() if ts >= tf and code >= 328 and ch == 0]
            s["t_sample_over"] = over[0][0] if over else None
            tt2 = np.arange(tf, tf + 3e-3, 50e-9); ii2, _ = P.waveform(tt2)
            s["i_peak"] = float(ii2.max())
            s["i_before"] = float(P.current(tf))
        return s


def heater_demo(t_end=1.6, fault_at=0.45, fault_R=0.4, reset_at=0.75, restart_at=0.85, firmware=None, plc_il=None):
    """标准演示：启动 → 软启动 → 运行 → 负载短路 → MCU 过流封锁 → 报警给 PLC → PLC 停机 → 复位 → 再启动 → 计时到停机。
    firmware：None=汇编固件（HEATER_ASM）；字符串=自定义汇编；HeaterFirmwarePy() 等对象=行为级 Python 固件。"""
    cs = CoSim(asm=firmware if firmware is not None else FW.HEATER_ASM, plc_il=plc_il or FW.PLC_IL)
    cs.press("X000", 0.020, 0.05)
    cs.inject_fault(fault_at, fault_R)
    cs.at(fault_at + 0.15, lambda: (cs.plant.set_load(cs.now(), cs.plant.R0), cs.load_hist.append((cs.now(), cs.plant.R0)),
                                    cs.events_log.append((cs.now(), "排除短路：R 恢复"))), "排除故障")
    cs.press("X005", reset_at, 0.05)
    cs.press("X000", restart_at, 0.05)
    cs.run(t_end)
    return cs


# ---------------------------------------------------------------- ngspice 回放（v0.6 电路引擎）
def replay_ngspice(cs, t0, t1, step=20e-9):
    """把联合仿真里 MCU 驱动出来的门极时序（t0..t1）原样送进 ngspice：两个通用 VDMOS 半桥 + RL 负载 + 采样电阻。
    返回 (ngspice 波形, 同一时段快速电路对象的电流, 对比数字)。负载电阻取 t0 时刻的值（窗口里不要有故障切换）。"""
    from ..circuit.spice import Circuit
    P = cs.plant
    R = level_at(cs.load_hist, t0, P.R0)
    edges = [(t, k, v) for (t, k, v) in P.gate_edges(t1) if t0 <= t <= t1]
    hs0 = ls0 = 0
    for t, k, v in P.gate_edges(t0):
        if k == "hs": hs0 = v
        else: ls0 = v
    Vg, tr = 12.0, 20e-9

    def pwl(kind, init):
        pts = [(0.0, Vg * init)]; cur = init
        for t, k, v in edges:
            if k != kind or v == cur:
                continue
            tt = t - t0
            pts.append((max(tt, pts[-1][0] + 1e-12), Vg * cur)); pts.append((tt + tr, Vg * v)); cur = v
        pts.append((t1 - t0, Vg * cur))
        return pts

    c = Circuit("MCU 门极时序回放：半桥 + RL 负载")
    c.V("bus", "bus", "0", dc=P.Vbus)
    c.C("bus", "bus", "0", 100e-6)
    c.V("gh", "gh_drv", "sw", pwl=pwl("hs", hs0)); c.V("gl", "gl_drv", "0", pwl=pwl("ls", ls0))
    c.R("gh", "gh_drv", "gh", 4.7); c.R("gl", "gl_drv", "gl", 4.7)
    c.M("1", "bus", "gh", "sw", "NPWR"); c.M("2", "sw", "gl", "0", "NPWR")
    i0 = P.current(t0)
    c.L("load", "sw", "a", P.L, ic=i0)
    c.R("load", "a", "b", R)
    c.R("sh", "b", "0", P.Rsh)
    w = c.tran(step, t1 - t0, tmax=step * 5, uic=True)
    t = w.x + t0
    i_sp = np.real(w["b"]) / P.Rsh
    i_fast, _ = P.waveform(t)
    # ADC：在 MCU 实际采样时刻，用 ngspice 的采样电阻电压算出 ADC 码，和联合仿真里 MCU 读到的码比较
    adc = [(ts, code) for (ts, td, ch, v, code) in cs.adc_samples() if t0 <= ts <= t1 and ch == 0]
    vs = np.interp([a[0] for a in adc], t, np.clip(i_sp * P.Rsh * P.gain, 0, P.v_rail))
    codes_sp = [cs.mcu.adc.quantize(v, 5.0) for v in vs]
    d = dict(i_mean_sp=float(np.mean(i_sp)), i_mean_fast=float(np.mean(i_fast)),
             ripple_sp=float(i_sp.max() - i_sp.min()), ripple_fast=float(i_fast.max() - i_fast.min()),
             adc_n=len(adc), adc_code_diff_max=int(max((abs(a - b[1]) for a, b in zip(codes_sp, adc)), default=0)),
             adc_code_diff_mean=float(np.mean([a - b[1] for a, b in zip(codes_sp, adc)])) if adc else 0.0)
    return w, t, i_sp, i_fast, d
