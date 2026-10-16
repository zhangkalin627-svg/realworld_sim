# -*- coding: utf-8 -*-
"""联合仿真用的“快速电路对象”：通用半桥（带死区的驱动器）+ RL 负载（电阻加热元件/线圈）+ 采样电阻和放大器。

开关之间按 RL 电路的解析解逐段精确积分（不是固定步长），所以 MCU 每一个 PWM 边沿都精确生效。
半桥：HS 导通 → 开关节点 = Vbus；LS 导通 → 0 V；两管都关（死区/封锁）→ 电流经体二极管续流（-Vf 或 Vbus+Vf），到零后保持 0。
驱动器：通用两线驱动（IN + EN），类似常见半桥驱动芯片的行为：EN=0 两管关；EN=1 时 HS=IN、LS=!IN，开通延时 = 死区。
参数是通用演示值，不是任何具体产品。同一拓扑可以导出到 ngspice（v0.6）做器件级回放，见 cosim.replay_ngspice。
"""
import bisect, math
import numpy as np


class HalfBridgeRL:
    def __init__(self, Vbus=48.0, L=1e-3, R=4.0, Rds=0.02, Rsh=0.01, gain=20.0, Vf=0.8, dead=0.5e-6,
                 v_rail=5.0, pin_in="PD5", pin_en="PD6"):
        self.Vbus, self.L, self.R0, self.Rds, self.Rsh, self.gain, self.Vf, self.dead = Vbus, L, R, Rds, Rsh, gain, Vf, dead
        self.v_rail = v_rail
        self.pin_in, self.pin_en = pin_in, pin_en
        self.events = []          # 排好序的 (t, seq, kind, val)；kind: hs / ls / R
        self.seq = 0
        self.segs = [(0.0, 0.0, 0, 0, R)]   # (t0, i0, hs, ls, R)
        self.built = 0.0           # segs 已建到的时间（下一事件之前）
        self.n_applied = 0         # 已经并入 segs 的事件数
        self.in_lvl = 0; self.en = 0
        self.pending = {"hs": None, "ls": None}

    # ---- 驱动器：引脚 → 门极事件
    def _add(self, t, kind, val):
        ev = (t, self.seq, kind, val); self.seq += 1
        i = bisect.bisect(self.events, ev)
        self.events.insert(i, ev)
        if i < self.n_applied:            # 来晚了：从这个时间点之前重建
            self._rebuild_from(t)
        return ev

    def _cancel(self, ev):
        i = bisect.bisect_left(self.events, ev)
        if i < len(self.events) and self.events[i] == ev:
            self.events.pop(i)
            if i < self.n_applied:
                self._rebuild_from(ev[0])

    def _rebuild_from(self, t):
        starts = [sg[0] for sg in self.segs]
        k = bisect.bisect_left(starts, t)
        if k == 0:
            self.segs = [(0.0, 0.0, 0, 0, self.R0)]; self.n_applied = 0; self.built = 0.0
            return
        self.segs = self.segs[:k]
        t0 = self.segs[-1][0]
        self.n_applied = bisect.bisect_right(self.events, (t0, float("inf")))
        self.built = t0

    def on_pin(self, name, level, t, is_out=True):
        if name not in (self.pin_in, self.pin_en):
            return
        if name == self.pin_in:
            self.in_lvl = level
        else:
            self.en = level
        want_hs = self.en and self.in_lvl
        want_ls = self.en and not self.in_lvl
        for sw, want in (("hs", want_hs), ("ls", want_ls)):
            p = self.pending[sw]
            if p is not None and p[0] <= t:          # 开通事件已经发生
                self.pending[sw] = p = None
            if not want:
                if p is not None:
                    self._cancel(p); self.pending[sw] = None
                self._add(t, sw, 0)                       # 关断：立即
            elif self.pending[sw] is None:
                self.pending[sw] = self._add(t + self.dead, sw, 1)   # 开通：延时一个死区

    def set_load(self, t, R):
        self._add(t, "R", R)

    # ---- 解析解
    def _seg_eval(self, seg, t):
        t0, i0, hs, ls, R = seg
        Rt = R + self.Rds + self.Rsh
        tau = self.L / Rt
        dt = t - t0
        if hs or ls:
            v = self.Vbus if hs else 0.0
            ii = v / Rt
            return ii + (i0 - ii) * math.exp(-dt / tau)
        if i0 == 0.0:
            return 0.0
        v = -self.Vf if i0 > 0 else self.Vbus + self.Vf
        ii = v / Rt
        ratio = (0.0 - ii) / (i0 - ii)
        tz = -tau * math.log(ratio) if 0 < ratio < 1 else float("inf")
        if dt >= tz:
            return 0.0
        return ii + (i0 - ii) * math.exp(-dt / tau)

    def _build(self, t):
        while self.n_applied < len(self.events) and self.events[self.n_applied][0] <= t:
            te, _, kind, val = self.events[self.n_applied]
            last = self.segs[-1]
            i_e = self._seg_eval(last, te)
            hs, ls, R = last[2], last[3], last[4]
            if kind == "hs": hs = val
            elif kind == "ls": ls = val
            else: R = val
            if te == last[0]:
                self.segs[-1] = (te, last[1], hs, ls, R)
            else:
                self.segs.append((te, i_e, hs, ls, R))
            self.n_applied += 1
        self.built = max(self.built, t)

    def current(self, t):
        self._build(t)
        k = bisect.bisect_right(self.segs, (t, float("inf"))) - 1
        return self._seg_eval(self.segs[max(k, 0)], t)

    def v_sense(self, t):
        """电流采样放大器输出（单电源，0…v_rail 削顶）→ 接 MCU ADC。"""
        return min(self.v_rail, max(0.0, self.current(t) * self.Rsh * self.gain))

    def amps_per_volt(self):
        return 1.0 / (self.Rsh * self.gain)

    # ---- 波形（画图/送示波器）
    def waveform(self, t):
        t = np.asarray(t, float)
        self._build(float(t[-1]))
        starts = np.array([s[0] for s in self.segs])
        idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(self.segs) - 1)
        i = np.empty_like(t); vsw = np.empty_like(t)
        for k in np.unique(idx):
            m = idx == k
            seg = self.segs[k]
            i[m] = [self._seg_eval(seg, x) for x in t[m]]
            if seg[2]:
                vsw[m] = self.Vbus
            elif seg[3]:
                vsw[m] = 0.0
            else:
                vsw[m] = np.where(i[m] > 0, -self.Vf, np.where(i[m] < 0, self.Vbus + self.Vf, 0.0))
        return i, vsw

    def gate_edges(self, t_end):
        """HS/LS 门极事件列表 [(t, kind, val)]（已应用、按时间排序），给 ngspice 回放用。"""
        self._build(t_end)
        return [(t, k, v) for (t, _, k, v) in self.events if t <= t_end and k in ("hs", "ls")]
