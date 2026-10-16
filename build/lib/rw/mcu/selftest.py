# -*- coding: utf-8 -*-
"""v0.7 自检：理论值 vs 仿真值。validate.py 第 9 段调用 run()；也可以 python3 -m rw.mcu.selftest。
每一项都写明“验证”还是“估算”：
- 验证（内部一致性）：理论值来自 AVR 指令手册周期数 / ATmega328P 数据手册公式，仿真值从引脚波形量出来。
  这证明模拟器按手册实现，不证明和真芯片逐周期一致（没有真片对照）。
- 估算：行为级固件、ngspice 回放、unicorn 计时。
"""
import numpy as np

from .asm import assemble
from .avr import AVR
from . import firmware as FW

F = 16e6


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def _edges(m, pin):
    return np.array(m.pin_log.get(pin, []), dtype=float)


def _pwm_meas(m, pin):
    e = _edges(m, pin)
    e = e[len(e) // 3:]                       # 去掉启动段
    r = e[e[:, 1] == 1, 0]; f = e[e[:, 1] == 0, 0]
    per = np.mean(np.diff(r))
    f = f[f > r[0]]; n = min(len(r), len(f))
    high = np.mean(f[:n] - r[:n])
    return F / per, high / per


def run(verbose=True):
    out, res = [], {}

    def say(s):
        out.append(s)
        if verbose:
            print(s)

    # ---- 9a 指令计时
    m = AVR(assemble(FW.TIMING_ASM)); m.run_until(60000)
    d = np.diff(_edges(m, "PB5")[1:, 0])
    th = FW.TIMING_CYCLES
    res["timing"] = (th, float(d.mean()), int(d.min()), int(d.max()))
    say("[MCU] 指令计时：一次循环（sbi+200×dec/brne+1000×sbiw/brne+rcall/ret+rjmp）理论 %d 周期 = %.4f µs；仿真 %.0f 周期（min %d / max %d）= %.4f µs" % (
        th, th / F * 1e6, d.mean(), d.min(), d.max(), d.mean() / F * 1e6))
    assert d.min() == d.max() == th
    # 中断响应
    m = AVR(assemble(FW.INT_ASM)); m.run_until(2000); c0 = m.cyc
    m.set_input("PD2", 1); m.run_until(c0 + 200)
    e = _edges(m, "PB1"); lat = int(e[e[:, 1] == 1, 0][0] - c0)
    res["int_latency"] = (11, lat)
    say("[MCU] 外部中断响应（睡眠中）：理论 4 响应 + 4 唤醒 + 2 RJMP + 1（SBI 第 2 周期生效）= 11 周期；仿真 %d 周期 = %.4f µs" % (lat, lat / F * 1e6))
    assert lat == 11

    # ---- 9b 定时器溢出 / CTC
    m = AVR(assemble(FW.timer_asm(0, 3, 0))); m.run_for(0.03)
    e = _edges(m, "PB0")[2:, 0]; fo = F / np.mean(np.diff(e))
    th = F / (64 * 256)
    res["ovf"] = (th, fo)
    say("[MCU] 定时器0 Normal 溢出频率（预分频 64）：理论 f/(64·256) = %.4f Hz；仿真 %.4f Hz（%+.4f%%）" % (th, fo, pct(fo, th)))
    assert abs(pct(fo, th)) < 1e-6
    m = AVR(assemble(FW.timer_asm(2, 3, 124, com="(1<<COM0A0)"))); m.run_for(0.03)
    e = _edges(m, "PB1")[2:, 0]; fc = F / np.mean(np.diff(e))
    fpin, duty = _pwm_meas(m, "PD6")
    res["ctc"] = (2000.0, fc, 1000.0, fpin)
    say("[MCU] 定时器0 CTC（OCR0A=124，预分频 64）：比较中断 理论 2000 Hz / 仿真 %.4f Hz；OC0A 翻转输出 理论 1000 Hz / 仿真 %.4f Hz（占空比 %.4f）" % (fc, fpin, duty))
    assert abs(fc - 2000) < 1e-6 and abs(fpin - 1000) < 1e-6

    # ---- 9c PWM
    rows = []
    for mode, cs, ocra, ocrb, com, pin, fth, dth, txt in (
            (7, 2, 99, 29, "(1<<COM0B1)", "PD5", F / (8 * 100), 30 / 100, "Fast PWM TOP=OCR0A=99，预分频 8，OCR0B=29 非反相"),
            (3, 1, 63, 0, "(1<<COM0A1)", "PD6", F / 256, 64 / 256, "Fast PWM TOP=0xFF，预分频 1，OCR0A=63 非反相"),
            (3, 1, 0, 191, "(1<<COM0B1)|(1<<COM0B0)", "PD5", F / 256, 1 - 192 / 256, "Fast PWM TOP=0xFF，OCR0B=191 反相"),
            (3, 1, 0, 0, "(1<<COM0B1)", "PD5", F / 256, 1 / 256, "Fast PWM OCR0B=0 非反相（手册：窄脉冲 1/256）")):
        m = AVR(assemble(FW.timer_asm(mode, cs, ocra, ocrb, com))); m.run_for(0.004)
        fp, dp = _pwm_meas(m, pin)
        rows.append((txt, fth, fp, dth, dp))
        say("[MCU] PWM %s：频率 理论 %.3f Hz / 仿真 %.3f Hz；占空比 理论 %.5f / 仿真 %.5f" % (txt, fth, fp, dth, dp))
        assert abs(pct(fp, fth)) < 1e-6 and abs(dp - dth) < 1e-9
    res["pwm"] = rows
    # 同一个 PWM 送进 v0.6 示波器的自动测量
    from ..circuit.scope import measure
    m = AVR(assemble(FW.timer_asm(7, 2, 99, 29, "(1<<COM0B1)"))); m.run_for(0.002)
    e = _edges(m, "PD5"); t = np.arange(0.0005, 0.002, 1 / F * 4)
    k = np.searchsorted(e[:, 0] / F, t, side="right") - 1; y = 5.0 * e[k, 1]
    ms = measure(t, y)
    res["pwm_scope"] = (ms["频率"], ms["占空比"])
    say("[MCU] 同一路 PWM 送进 ⑥ 示波器自动测量（0.25 µs 采样）：频率 %.2f Hz、占空比 %.4f（理论 20000 Hz、0.30；差别来自示波器采样点分辨率）" % (ms["频率"], ms["占空比"]))

    # ---- 9d ADC 量化
    errs = []; tconv = None
    vins = np.linspace(0.0, 5.0, 401)
    for v in vins:
        m = AVR(assemble(FW.ADC_ASM)); m.analog[0] = float(v); m.run_until(10000)
        code = m.mem[0x100] | m.mem[0x101] << 8
        errs.append(code - min(v * 1024 / 5.0, 1023.0))
        st, sm, dn, ch, vi, vr, cd = m.adc.log[0]; tconv = dn - st
    errs = np.array(errs)
    res["adc"] = (float(np.abs(errs).max()), float(errs.mean()), float(np.std(errs)), tconv)
    say("[MCU] ADC 量化（10 位，Vref 5 V，401 个输入 0–5 V）：误差 max|e| = %.4f LSB（理论 <1）、平均 %.4f LSB（截断量化理论 -0.5）、标准差 %.4f LSB（理论 1/√12 = 0.2887）" % (
        np.abs(errs).max(), errs.mean(), errs.std()))
    say("[MCU] ADC 首次转换时间：理论 25 个 ADC 时钟 × 128 / 16 MHz = 200.000 µs；仿真 %.3f µs" % (tconv / F * 1e6))
    assert np.abs(errs).max() < 1.0 and tconv == 25 * 128

    # ---- 9e UART
    m = AVR(assemble(FW.UART_ASM)); m.run_for(0.006)
    e = _edges(m, "PD1"); bit = np.diff(e[2:, 0]).min()          # e[0] 复位、e[1] TXEN 拉高空闲
    baud = F / bit; th = F / (16 * 104)
    # 独立解码 TX 引脚波形（在每一位中间采样）
    txt = ""; i = 1
    while i < len(e):
        if e[i, 1] == 0:
            t0 = e[i, 0]; b = 0
            for k in range(8):
                ts = t0 + (1.5 + k) * bit
                j = np.searchsorted(e[:, 0], ts, side="right") - 1
                b |= int(e[j, 1]) << k
            txt += chr(b)
            tend = t0 + 9.5 * bit
            i = np.searchsorted(e[:, 0], tend, side="right")
            while i < len(e) and e[i, 1] != 0:
                i += 1
        else:
            i += 1
    res["uart"] = (th, baud, txt)
    say("[MCU] UART：UBRR=103 → 理论 f/(16·104) = %.2f 波特（相对 9600 偏 %+.2f%%）；仿真位宽 %d 周期 → %.2f 波特；从 TX 引脚波形独立解码得到 %r（应为 'UOK\\n'）" % (
        th, pct(th, 9600), bit, baud, txt))
    assert abs(baud - th) < 1e-6 and txt == "UOK\n"

    # ---- 9f/9g 联合仿真：握手时序 + 过流保护
    from .cosim import heater_demo, replay_ngspice
    cs = heater_demo(); s = cs.summary()
    scan, xf, yd = cs.scan, cs.xf, cs.yd
    a = s["t_Y000_on"] - s["t_btn"]
    say("[联合] 启动按钮 X000 → PLC Y000 输出：理论窗口 [滤波 %.0f + 扫描 %.0f, 滤波 + 2×扫描] = [%.1f, %.1f] ms；仿真 %.3f ms" % (
        xf * 1e3, scan * 1e3, (xf + scan) * 1e3, (xf + 2 * scan) * 1e3, a * 1e3))
    assert xf + scan - 1e-9 <= a <= xf + 2 * scan + 1e-9
    b = s["t_EN_on"] - s["t_Y000_on"]; thb = yd + 21 / F
    say("[联合] PLC Y000 → MCU 打开驱动 EN：理论 输出延时 %.1f ms + INT0 中断 21 周期（1.3125 µs）= %.4f ms；仿真 %.4f ms" % (yd * 1e3, thb * 1e3, b * 1e3))
    assert abs(b - thb) < 1e-9
    c = s["t_Y000_off"] - s["t_PD7_on"]
    say("[联合] MCU 报警 PD7 → PLC 撤销 Y000：理论窗口 [%.1f, %.1f] ms；仿真 %.3f ms；再 +%.1f ms 到达 MCU PD2" % (
        (xf + scan) * 1e3, (xf + 2 * scan) * 1e3, c * 1e3, (s["t_PD2_off"] - s["t_Y000_off"]) * 1e3))
    assert xf + scan - 1e-9 <= c <= xf + 2 * scan + 1e-9
    resp_s = s["t_EN_off"] - s["t_sample_over"]
    th_s = (13 - 1.5) * 32 / F + 25 / F      # 唤醒+响应 8、RJMP 2、ISR 到 CBI 前 14、CBI 第 2 周期生效 1
    resp_c = s["t_EN_off"] - s["t_cross"]
    say("[联合] 过流保护：ADC 采样到超限 → 关 EN 理论 11.5 个 ADC 时钟（23 µs）+ 中断与判断 25 周期（1.5625 µs）= %.3f µs；仿真 %.3f µs" % (th_s * 1e6, resp_s * 1e6))
    assert abs(resp_s - th_s) < 1e-9
    say("[联合] 过流保护：电流真正越过 %.2f A → 关 EN 理论窗口 [%.1f, %.1f] µs（采样间隔 = 1 个 PWM 周期 50 µs）；仿真 %.1f µs；短路前 %.2f A，峰值 %.2f A" % (
        s["i_th"], th_s * 1e6, (th_s + 50e-6) * 1e6, resp_c * 1e6, s["i_before"], s["i_peak"]))
    assert th_s - 1e-9 <= resp_c <= th_s + 50e-6 + 1e-9
    res["cosim"] = s
    say("[联合] 串口记录：" + "".join(ch for t, ch in cs.uart_text()) + "（S 启动 R 到达 OC 过流 r 复位）")

    # ---- 估算项
    from .behav import HeaterFirmwarePy
    cb = heater_demo(firmware=HeaterFirmwarePy()); sb = cb.summary()
    say("[估算] 行为级 Python 固件（每个中断按 60 周期估）：过流响应 %.2f µs（汇编版 %.2f µs）；软启动到达 %.2f ms（汇编版 %.2f ms）" % (
        (sb["t_EN_off"] - sb["t_cross"]) * 1e6, resp_c * 1e6, sb["t_reached"] * 1e3, s["t_reached"] * 1e3))
    try:
        from ..circuit.spice import find_ngspice
        if find_ngspice():
            w, t, isp, ifa, dd = replay_ngspice(cs, 0.30, 0.302)
            res["replay"] = dd
            say("[估算] ngspice 回放（MCU 门极时序 2 ms → 通用 VDMOS 半桥 + RL）：平均电流 ngspice %.4f A / 快速模型 %.4f A（%+.2f%%）；纹波 %.3f / %.3f A；%d 次 ADC 采样的码值差 最大 %d LSB" % (
                dd["i_mean_sp"], dd["i_mean_fast"], pct(dd["i_mean_fast"], dd["i_mean_sp"]), dd["ripple_sp"], dd["ripple_fast"], dd["adc_n"], dd["adc_code_diff_max"]))
    except Exception as ex:   # noqa: BLE001
        say("[估算] ngspice 回放失败：%s" % ex)
    from .cortexm import HAVE_UNICORN
    if HAVE_UNICORN:
        from .cortexm import CortexM
        cm = CortexM(); cm.run(3000)
        tog = len(cm.pin_log)
        say("[可选] unicorn Cortex-M（Thumb）冒烟：执行 %d 条指令，GPIO 写 %d 次（循环 3 条指令写一次，应为 %d）；计时按 CPI=1 估算" % (cm.n, tog, (cm.n - 4 + 2) // 3))
    else:
        say("[可选] 没装 unicorn，跳过 Cortex-M 冒烟（pip install unicorn）")
    return out, res


if __name__ == "__main__":
    run()
