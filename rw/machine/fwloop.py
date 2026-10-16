# -*- coding: utf-8 -*-
"""真实固件在环（v0.32）：指令级 AVR 固件（rw.mcu.fw_quench）驱动 machine 功率级。

桥接纪律（"真实"二字的边界）：
- 固件只能看引脚：V/I 过零比较器（PD2/PD3 脉冲）、电流包络（ADC0）——和真实控制器一样的传感器。
- 固件只能动手脚：PD5 门极方波（T0 PWM）、PD6 EN、PD7 LOCK、PB0 FAULT。
- 桥从 PD5 实际边沿测周期（含固件 DSM 抖动的真实频率），按 Stage 基波解算 V1/I/φ，
  合成比较器过零脉冲喂回固件；负载剧本（T_wp/k/C_res）按 validate_pll 同款时间轴推进。
- 频率、相位、电流的"物理真值"取 Stage.solve 在实测平均频率下的解（每门极周期刷新）。
"""
import bisect, heapq, math
from ..mcu.asm import assemble
from ..mcu.avr import AVR
from ..mcu.periph import INF
from ..mcu.fw_quench import QUENCH_PLL_ASM
from .power import Spec, Stage

TWO_PI = 2.0 * math.pi


class PinInjector:
    """周期精确的引脚注入器（比较器/传感器的诚实通道）：挂在 MCU 的 timed 外设链上，
    在 AVR 主循环的 _sync_all 里按确切周期施加电平 —— 不用外部事件堆（那会把边沿压到弹出时刻）。"""

    def __init__(self, mcu):
        self.mcu = mcu
        self.ev = []            # 按周期排序的 (cyc, pin, level)
        self.addrs = ()

    def add(self, cyc, pin, level):
        i = bisect.bisect(self.ev, (cyc, chr(0x10FFFF), 2))
        self.ev.insert(i, (cyc, pin, level))
        if cyc < self.mcu.next_evt:
            self.mcu.next_evt = cyc          # 比已知下一事件更早：让主循环及时 sync

    def next_event(self):
        return self.ev[0][0] if self.ev else INF

    def sync(self, cyc):
        while self.ev and self.ev[0][0] <= cyc:
            c, pin, lvl = self.ev.pop(0)
            self.mcu.set_input(pin, lvl, cyc=c)

    def pending(self):
        return ()

    def ack(self, vec):
        pass


class FWLoop:
    def __init__(self, sp=None, asm=None, Vbus=None):
        self.sp = sp or Spec()
        self.st = Stage(self.sp)
        self.mcu = AVR(assemble(asm if asm is not None else QUENCH_PLL_ASM))
        self.Vbus = Vbus if Vbus is not None else self.sp.V_src()
        # 负载状态（剧本驱动）
        self.T_wp = 20.0; self.k = 0.949173; self.T_coil = 40.0; self.m = 1.0
        self.q = []; self._seq = 0; self._tnow = None
        self.I_env = 0.0
        self.mcu.analog[0] = lambda t: min(self.I_env * 0.01, 5.0)   # 电流包络 0.01 V/A rms
        self.inj = PinInjector(self.mcu)
        self.mcu.timed.append(self.inj)
        self.mcu.listeners.append(self._on_pin)
        self.gate_rise = None       # 上次 PD5 上升沿时刻
        self.per_avg = None         # 门极周期 EMA(1/16)
        self.solve = None           # 最近 Stage 解（实测平均频率下）
        self.hist = []              # 每门极周期：(t, f_cyc, f_avg, phi, I, P_out)

    # ---------------------------------------------------------- 时间与事件
    def at(self, t, fn):
        heapq.heappush(self.q, (self.mcu.cycles(t), self._seq, fn)); self._seq += 1

    def now(self):
        return self._tnow if self._tnow is not None else self.mcu.t()

    def _pulse(self, pin, t):
        c = self.mcu.cycles(t)
        self.inj.add(c, pin, 1)
        self.inj.add(c + 16, pin, 0)         # 1 µs 比较器脉冲

    def pin_hist(self, pin):
        return [(c / self.mcu.f, v) for c, v in self.mcu.pin_log.get(pin, [])]

    def first_edge(self, pin, level=1, after=0.0):
        for t, v in self.pin_hist(pin):
            if v == level and t > after:
                return t
        return None

    # ---------------------------------------------------------- 门极 → 功率级 → 比较器
    def _on_pin(self, name, level, t, is_out):
        if name != "PD5" or not is_out or level != 1:
            return
        prev = self.gate_rise
        self.gate_rise = t
        if prev is None:
            return
        per = t - prev
        if per <= 0:
            return
        self.per_avg = per if self.per_avg is None else self.per_avg + (per - self.per_avg) / 16.0
        f = 1.0 / self.per_avg
        s = self.st.solve(self.Vbus, f, self.T_wp, self.k, self.T_coil, self.m)
        self.solve = s
        self.I_env = s["I"]
        phi = math.atan2(s["X"], s["R"])
        self.hist.append((t, 1.0 / per, f, phi, s["I"], s["P_out"]))
        # 比较器：V 基波与门极方波同相（上升沿即过零）；I 滞后 φ
        self._pulse("PD2", t)
        frac = (phi % TWO_PI) / TWO_PI
        self._pulse("PD3", t + frac * self.per_avg)

    # ---------------------------------------------------------- 运行
    def run(self, t_end):
        end = self.mcu.cycles(t_end)
        while True:
            nxt = self.q[0][0] if self.q else end
            stop = min(nxt, end)
            self.mcu.run_until(stop)
            if self.q and self.q[0][0] <= end and self.q[0][0] <= self.mcu.cyc:
                c, _, fn = heapq.heappop(self.q)
                self._tnow = c / self.mcu.f
                fn()
                self._tnow = None
            elif stop >= end:
                break
