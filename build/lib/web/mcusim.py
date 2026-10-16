"""单片机/PLC 联合仿真 Web 模块 (v0.21)：⑦ 页移植——AVR 子集(16 MHz) + FX PLC + 半桥电路统一时间轴。
标准演示：启动握手 → 软启动 → 负载短路 → MCU 过流封锁 → 报警给 PLC → 停机 → 复位 → 再启动。
内核与桌面版完全一致（rw.mcu）；理论对照见 rw/mcu/selftest.py（validate.py §9）。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .modules import _clean, _svg


def _plot_window(cs, s, t0, t1, title):
    """桌面版 redraw() 的 4 面板时序图（去掉光标/表格交互）。"""
    w = cs.waves(t0, t1, 6000)
    t = w.x; tm = t * 1e3
    fig = plt.figure(figsize=(8.6, 7.6))
    ax = fig.subplots(4, 1, sharex=True, gridspec_kw=dict(height_ratios=[2.2, 2.2, 1.2, 2.4], hspace=0.22))
    fig.subplots_adjust(left=0.15, right=0.965, top=0.94, bottom=0.07)

    def dig(a, names, ttl):
        for i, (key, lab, col) in enumerate(names):
            y = (w[key] > 0.5 * w[key].max() if w[key].max() > 0 else w[key] * 0).astype(float)
            a.fill_between(tm, i * 1.4, i * 1.4 + y, step="post", color=col, alpha=0.35, lw=0)
            a.step(tm, i * 1.4 + y, where="post", color=col, lw=1.1)
        a.set_yticks([i * 1.4 + 0.5 for i in range(len(names))]); a.set_yticklabels([n[1] for n in names], fontsize=8)
        a.set_ylim(-0.3, len(names) * 1.4); a.set_title(ttl, fontsize=9, loc="left", pad=2)
    dig(ax[0], [("PLC_X000", "X000 启动", "#1e88e5"), ("PLC_Y000", "Y000 启动", "#43a047"), ("PLC_X003", "X003 报警", "#e53935"),
                ("PLC_Y001", "Y001 报警灯", "#c62828"), ("PLC_Y002", "Y002 加热中", "#8e24aa")], "PLC（FX 指令表，扫描 2 ms）")
    dig(ax[1], [("EN", "PD6 驱动 EN", "#43a047"), ("PWM_IN", "PD5 PWM", "#fb8c00"), ("MCU_故障", "PD7 报警", "#e53935"),
                ("MCU_到达", "PB0 到达", "#8e24aa"), ("TXD", "PD1 TXD", "#546e7a")], "单片机引脚")
    e = np.array(cs.pin_hist("PD5")) if cs.pin_hist("PD5") else np.zeros((0, 2))
    if len(e) > 2:
        r = e[e[:, 1] == 1, 0]; f = e[e[:, 1] == 0, 0]
        dd, tt = [], []
        for a0, a1 in zip(r[:-1], r[1:]):
            ff = f[(f > a0) & (f <= a1)]
            if len(ff) and a1 - a0 < 1e-3:
                dd.append((ff[0] - a0) / (a1 - a0) * 100); tt.append(a0)
        if tt:
            tt = np.array(tt); dd = np.array(dd); mk = (tt >= t0) & (tt <= t1)
            ax[2].step(tt[mk] * 1e3, dd[mk], where="post", color="#fb8c00", lw=1.2)
    ax[2].set_ylabel("占空比 %", fontsize=8); ax[2].set_ylim(-3, 60); ax[2].grid(alpha=0.3)
    ax[3].plot(tm, w["i_load"], color="#1565c0", lw=0.9, label="负载电流（快速电路模型）")
    ax[3].axhline(s["i_th"], color="#e53935", ls="--", lw=1, label=f"过流阈值 {s['i_th']:.2f} A")
    ad = [(ts, code) for (ts, td, ch, v, code) in cs.adc_samples() if t0 <= ts <= t1 and ch == 0]
    if ad and len(ad) < 3000:
        ax[3].plot([a[0] * 1e3 for a in ad], [a[1] * 5 / 1024 * cs.plant.amps_per_volt() for a in ad], ".", ms=3,
                   color="#2e7d32", label="ADC 采样（换算成 A）")
    ax[3].set_ylabel("A", fontsize=8); ax[3].grid(alpha=0.3); ax[3].legend(fontsize=7, loc="upper right")
    ax[3].set_xlabel("时间 (ms)")
    for a in ax:
        a.tick_params(labelsize=8)
    ax[3].set_xlim(tm[0], tm[-1])
    fig.suptitle(title, fontsize=11)
    return {"title": title, "svg": _svg(fig)}


def run_mcu(params, progress):
    from rw.mcu.cosim import heater_demo
    from rw.mcu import firmware as FW
    from rw import viz
    t_end = min(max(float(params.get("t_end", 1.6)), 0.5), 3.0)
    fault_at = float(params.get("fault_at", 0.45))
    fault_R = min(max(float(params.get("fault_R", 0.4)), 0.05), 5.0)
    reset_at = float(params.get("reset_at", 0.75))
    restart_at = float(params.get("restart_at", 0.85))
    if not (0.05 < fault_at < reset_at < restart_at < t_end):
        raise ValueError(f"时刻必须满足 0.05 < 短路 {fault_at} < 复位 {reset_at} < 再启动 {restart_at} < 时长 {t_end}")
    custom_asm = (params.get("custom_asm") or "").strip()
    custom_plc = (params.get("custom_plc") or "").strip()
    sel = params.get("fw", "asm")
    if sel == "behav":
        from rw.mcu.behav import HeaterFirmwarePy
        fw = HeaterFirmwarePy()
    elif sel == "custom":
        if not custom_asm:
            raise ValueError("选择了自定义汇编固件，但程序框是空的")
        fw = custom_asm
    else:
        fw = None
    progress(0.05)
    cs = heater_demo(t_end=t_end, fault_at=fault_at, fault_R=fault_R,
                     reset_at=reset_at, restart_at=restart_at,
                     firmware=fw, plc_il=custom_plc or None)
    progress(0.6)
    s = cs.summary()
    m = cs.mcu

    def ms(x): return "—" if x is None else f"{x*1e3:.4f} ms"
    def d(a, b, unit="ms"):
        if a is None or b is None: return "—"
        return f"{(b-a)*1e3:.4f} ms" if unit == "ms" else f"{(b-a)*1e6:.3f} µs"
    lines = [f"仿真 {m.t():.3f} s：执行 {m.instr_count} 条指令，{len(m.irq_log)} 次中断，PLC {len(cs.plc_snap)} 次状态变化",
             "",
             "关键时刻（全部从仿真波形量出，不是设定值）：",
             f"· 按下启动 X000：{ms(s.get('t_btn'))}",
             f"· PLC Y000 输出（含 10 ms 输入滤波 + 扫描）：{d(s.get('t_btn'), s.get('t_Y000_on'))}",
             f"· Y000 → MCU 打开 EN（0.2 ms 输出延时 + 中断）：{d(s.get('t_Y000_on'), s.get('t_EN_on'))}",
             f"· 软启动到 50%（PB0 到达）：{d(s.get('t_EN_on'), s.get('t_reached'))}"]
    if "t_fault" in s:
        lines += ["",
                  f"· 短路时刻：{ms(s['t_fault'])}（R → {fault_R:g} Ω），短路前电流 {s['i_before']:.2f} A",
                  f"· 电流越过阈值 {s['i_th']:.2f} A：{d(s['t_fault'], s.get('t_cross'), 'us')}",
                  f"· ADC 采到超限 → 关 EN：{d(s.get('t_sample_over'), s.get('t_EN_off'), 'us')}",
                  f"· 越过阈值 → 关 EN（保护响应）：{d(s.get('t_cross'), s.get('t_EN_off'), 'us')}",
                  f"· 峰值电流：{s['i_peak']:.2f} A",
                  f"· MCU 报警 PD7 → PLC 撤销 Y000：{d(s.get('t_PD7_on'), s.get('t_Y000_off'))}",
                  f"· PLC 报警灯 Y001：{ms(s.get('t_Y001_on'))}"]
    ev = getattr(cs, "events_log", [])
    if ev:
        lines += ["", "事件日志："] + [f"· t={t*1e3:9.3f} ms  {msg}" for t, msg in ev[:14]]
    ut = list(cs.uart_text())
    if ut:
        lines += ["", "串口终端（末 8 行）："] + [f"· {t*1e3:9.3f} ms  {ch!r}" for t, ch in ut[-8:]]

    warns = []
    if "t_fault" in s:
        if s.get("t_EN_off") is None:
            warns.append("短路后 MCU 没有关 EN——过流保护没动作！检查固件 ADC/比较逻辑或阈值")
        if s.get("t_Y000_off") is None:
            warns.append("MCU 报警后 PLC 没有撤销 Y000——检查 PLC 程序里 X003 联锁")
    progress(0.75)

    T = m.t()
    tf = s.get("t_fault")
    plots = [_plot_window(cs, s, 0.0, T, "全程：启动 → 软启动 → 短路保护 → 复位重启"),
             _plot_window(cs, s, 0.025, 0.040, "特写：启动握手（PLC 滤波+扫描 → MCU 中断）")]
    if tf:
        plots.append(_plot_window(cs, s, tf - 0.1e-3, tf + 0.3e-3, "特写：过流保护（短路 ±0.3 ms）"))
    raw = {k: v for k, v in s.items() if isinstance(v, (int, float)) or v is None}
    raw["instr"] = m.instr_count; raw["irq"] = len(m.irq_log)
    progress(1.0)
    return _clean({"lines": lines, "warnings": warns, "plots": plots, "raw": raw})
