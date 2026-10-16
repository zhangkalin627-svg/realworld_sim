# -*- coding: utf-8 -*-
"""行为级固件模式：用 Python 写固件逻辑（中断处理函数），跑在同一套外设层上（定时器/ADC/UART/GPIO 和汇编模式完全一样）。

适合：先验证控制思路，再写汇编/C。时间精度：外设是周期精确的，但固件代码本身的执行时间是“估计值”
（每个中断处理函数按 cost_cycles 计，默认 60 周期），所以响应时间是估算，不是指令级结果。
"""
from .avr import AVR, I_
from .defs import SYMBOLS as S


class _EmptyProg:
    words = []; labels = {}; lines = {}; source = "(Python 行为级固件)"; symbols = {}


class PyMCU(AVR):
    def __init__(self, firmware, f_cpu=16e6, cost_cycles=60):
        super().__init__(_EmptyProg(), f_cpu=f_cpu)
        self.fw = firmware
        self.cost = cost_cycles
        self.sreg = 0
        # 寄存器访问辅助（数据空间地址）
        self.fw.setup(self)
        self.sleeping = True

    # 固件里用的小工具
    def io(self, name, v=None):
        a = S[name] + (0x20 if name in _IO_NAMES else 0)
        if v is None:
            return self.rd(a)
        self.wr(a, v)

    def setb(self, name, bit): self.io(name, self.io(name) | (1 << bit))
    def clrb(self, name, bit): self.io(name, self.io(name) & ~(1 << bit))
    def sei(self): self.sreg |= I_; self.irq_dirty = True

    def run_until(self, cyc_end, max_instr=None):
        while self.cyc < cyc_end:
            if not self.irq_dirty:
                nxt = min(self.next_evt, cyc_end)
                if nxt > self.cyc:
                    self.cyc = int(nxt)
            self._sync_all()
            self.irq_dirty = False
            if not self.sreg & I_:
                continue
            best = None
            for p in self.periphs:
                for v in p.pending():
                    if best is None or v < best[0]:
                        best = (v, p)
            if best is None:
                continue
            vec, p = best
            p.ack(vec)
            self.cyc += 8 + 2                     # 中断响应 4 + 唤醒 4 + 向量 RJMP 2（与汇编模式相同）
            self.irq_log.append((self.cyc, vec))
            h = self.fw.handlers.get(vec)
            if h:
                h(self)
            self.cyc += self.cost
            self.instr_count += 1
            if self.cyc >= self.next_evt:
                self._sync_all()


from .defs import IO as _IO
_IO_NAMES = set(_IO)


class HeaterFirmwarePy:
    """与 firmware.HEATER_ASM 同样逻辑的 Python 版本。"""
    OC_TH, SOFT_TH, DUTY_MAX, RAMP_DIV = 328, 287, 49, 40

    def __init__(self):
        self.state = 0; self.duty = 0; self.tick = 0; self.adc = 0
        self.handlers = {0x20: self.t0ovf, 0x2A: self.adc_done, 0x02: self.int0, 0x04: self.int1}

    def setup(self, m):
        m.io("DDRD", 0xF0); m.setb("DDRB", 0)
        m.io("UBRR0H", 0); m.io("UBRR0L", 103); m.io("UCSR0C", 0x06); m.io("UCSR0B", 1 << 3)
        m.io("OCR0A", 99); m.io("OCR0B", 0); m.io("TCCR0A", 0x03); m.io("TCCR0B", 0x0A); m.io("TIMSK0", 1)
        m.io("ADMUX", 0x40); m.io("ADCSRA", 0x80 | 0x08 | 0x05)
        m.io("EICRA", 0x0D); m.io("EIMSK", 0x03); m.io("SMCR", 1)
        m.setb("PORTD", 4); m.sei()

    def _send(self, m, ch): m.io("UDR0", ord(ch))

    def t0ovf(self, m):
        m.io("ADCSRA", 0x80 | 0x40 | 0x08 | 0x05)
        if self.state == 1:
            self.tick += 1
            if self.tick >= self.RAMP_DIV:
                self.tick = 0
                if self.adc >= self.SOFT_TH:
                    return
                if self.duty < self.DUTY_MAX:
                    self.duty += 1; m.io("OCR0B", self.duty)
                else:
                    self.state = 2; m.setb("PORTB", 0); self._send(m, "R")

    def adc_done(self, m):
        self.adc = m.io("ADCL") | m.io("ADCH") << 8
        if self.state in (1, 2) and self.adc >= self.OC_TH:
            m.clrb("PORTD", 6); m.io("TCCR0A", 0x03); m.clrb("PORTD", 5); m.setb("PORTD", 7); m.clrb("PORTB", 0)
            self.state = 3; self._send(m, "O"); self._send(m, "C")

    def int0(self, m):
        if m.pin("PD2"):
            if self.state == 0:
                self.duty = 0; self.tick = 0; m.io("OCR0B", 0); m.io("TCCR0A", 0x23); m.setb("PORTD", 6)
                self.state = 1; self._send(m, "S")
        else:
            m.clrb("PORTD", 6); m.io("TCCR0A", 0x03); m.clrb("PORTD", 5); m.clrb("PORTB", 0)
            if self.state != 3:
                self.state = 0

    def int1(self, m):
        if self.state == 3 and not m.pin("PD2"):
            m.clrb("PORTD", 7); self.state = 0; self._send(m, "r")


PY_SOURCE_NOTE = "行为级固件（Python）：rw/mcu/behav.py 里的 HeaterFirmwarePy，逻辑与汇编版相同；固件执行时间按每个中断 60 周期估计。"
