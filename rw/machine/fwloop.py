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
        # 模拟鉴相器（桥侧，CD4046 式）：v 方波取门极本身；i 方波 = v 延迟 φ；XOR 占空 ∝ |φ|/π
        self.v_sq = 0
        self.i_sq = 0
        self.xor_t = 0.0            # RC 状态上次更新时刻
        self.phase_v = 0.0          # RC 后相位直流（V）
        self.rc_tau = 200e-6        # 鉴相器 RC（1 ms 级响应，远快于环、远慢于载波）
        self.lead = False           # D 触发器：i 沿时 v 电平（1=容性超前）
        self.mcu.analog[1] = lambda t: self.phase_v
        self.at(0.0, lambda: self.mcu.set_input("PD4", 0))

    # ---------------------------------------------------------- 时间与事件
    def at(self, t, fn):
        heapq.heappush(self.q, (self.mcu.cycles(t), self._seq, fn)); self._seq += 1

    def now(self):
        return self._tnow if self._tnow is not None else self.mcu.t()

    def _rc_update(self, t):
        """XOR 当前值向 5 V/0 V 目标一阶指数推进到 t。"""
        dt = t - self.xor_t
        if dt <= 0:
            return
        tgt = 5.0 if (self.v_sq != self.i_sq) else 0.0
        self.phase_v += (tgt - self.phase_v) * (1.0 - math.exp(-dt / self.rc_tau))
        self.xor_t = t

    def pin_hist(self, pin):
        return [(c / self.mcu.f, v) for c, v in self.mcu.pin_log.get(pin, [])]

    def first_edge(self, pin, level=1, after=0.0):
        for t, v in self.pin_hist(pin):
            if v == level and t > after:
                return t
        return None

    # ---------------------------------------------------------- 门极 → 功率级 → 比较器
    def _on_pin(self, name, level, t, is_out):
        if name == "PD3" and not is_out:        # i 方波（注入器驱动）按真实时序到达
            self._rc_update(t)
            self.i_sq = level
            self.xor_t = t
            return
        if name != "PD5" or not is_out:
            return
        # PD5（门极 = v 方波）：先跟踪电平，再给这个 v 沿调度对应 i 沿（两个边沿都调度）；
        # i 沿经 PinInjector 打到 PD3，与 v 沿严格同序到达（堆事件会在分片内乱序，毁掉 RC 时间轴）
        self._rc_update(t)
        self.v_sq = level
        self.xor_t = t
        if self.solve is not None and self.per_avg:
            # i(t) = v(t - d)：i 沿复刻 v 沿的方向（不能 1-i_sq——初始未对齐会自锁反相）
            frac = (math.atan2(self.solve["X"], self.solve["R"]) % TWO_PI) / TWO_PI
            self.inj.add(self.mcu.cycles(t + frac * self.per_avg), "PD3", level)
        if level != 1:
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
        # D 触发器：i 沿（= v 沿延迟 φ）处 v 的电平 —— 滞后时 i 上升沿落在 v 高电平
        lead = phi < 0
        if lead != self.lead:
            self.lead = lead
            self.mcu.set_input("PD4", 1 if lead else 0)

    # ---------------------------------------------------------- 运行
    def run(self, t_end):
        end = self.mcu.cycles(t_end)
        chunk = self.mcu.cycles(20e-6)      # 20 µs 分片：堆事件（i 沿）及时弹出，RC 时间轴不失真
        while True:
            nxt = self.q[0][0] if self.q else end
            stop = min(nxt, self.mcu.cyc + chunk, end)
            self.mcu.run_until(stop)
            if self.q and self.q[0][0] <= end and self.q[0][0] <= self.mcu.cyc:
                c, _, fn = heapq.heappop(self.q)
                self._tnow = c / self.mcu.f
                fn()
                self._tnow = None
            elif stop >= end:
                break
