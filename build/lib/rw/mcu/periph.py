# -*- coding: utf-8 -*-
"""通用 MCU 外设层（MIT，自写）：GPIO 端口、8 位定时器/PWM、10 位 ADC、UART、外部中断。

寄存器布局按 ATmega328P 数据手册（公开文档），行为按手册描述实现，时间以 CPU 时钟周期计（整数）。
外设是“事件驱动”的：sync(cyc) 把状态推进到 cyc，next_event() 给出下一次会发生事情的周期，
CPU 睡眠时可以直接跳到下一事件，所以几秒钟的联合仿真不需要逐周期执行。

已知简化（在 README 局限里也写了）：
- 引脚输入没有 2 个周期的同步器延迟；悬空输入读 0（PORT=1 时按上拉读 1）
- 定时器只实现 Normal / CTC / Fast PWM（WGM=0,2,3,7）；相位修正 PWM、外部时钟源未实现，写入时报错
- ADC 只做单次转换（ADATE 自动触发未实现）；转换从写 ADSC 的那个周期开始（真片对齐到下一个 ADC 时钟沿，±1 ADC 时钟）
- UART 只做异步 8N1/8N2（5–8 位数据），不做奇偶校验；发送从写 UDR0 的周期开始（真片对齐到波特率时钟）
"""
INF = float("inf")


class MCUError(RuntimeError):
    pass


# ---------------------------------------------------------------- GPIO
class Port:
    def __init__(self, mcu, letter, pin_io, nbits=8):
        self.mcu, self.L, self.n = mcu, letter, nbits
        self.a_pin, self.a_ddr, self.a_port = pin_io + 0x20, pin_io + 0x21, pin_io + 0x22
        self.addrs = (self.a_pin, self.a_ddr, self.a_port)
        self.ddr = 0; self.port = 0
        self.ext = [None] * 8            # 外部驱动电平（None = 悬空）
        self.override = [None] * 8       # 外设接管输出（定时器 OC、UART TXD）
        self.force_out = [False] * 8     # 外设强制为输出（UART TXEN 时 TXD）
        self.last = [None] * 8           # 上次通知过的电平

    def name(self, b):
        return f"P{self.L}{b}"

    def is_output(self, b):
        return bool(self.ddr >> b & 1) or self.force_out[b]

    def level(self, b):
        if self.is_output(b):
            o = self.override[b]
            return int(o) if o is not None else (self.port >> b) & 1
        e = self.ext[b]
        if e is None:
            return (self.port >> b) & 1      # PORT=1 → 内部上拉
        return int(bool(e))

    def refresh(self, cyc, bits=range(8)):
        for b in bits:
            if b >= self.n:
                continue
            v = self.level(b)
            if v != self.last[b]:
                self.last[b] = v
                self.mcu._pin_change(self.name(b), v, cyc, self.is_output(b))

    def read(self, a, cyc):
        if a == self.a_pin:
            return sum(self.level(b) << b for b in range(self.n))
        return self.ddr if a == self.a_ddr else self.port

    def write(self, a, v, cyc):
        if a == self.a_pin:          # 写 PINx 的 1 → 翻转 PORTx 对应位（ATmega 特性）
            self.port ^= v
        elif a == self.a_ddr:
            self.ddr = v
        else:
            self.port = v
        self.refresh(cyc)

    def sync(self, cyc): pass
    def next_event(self): return INF
    def pending(self): return ()
    def ack(self, vec): pass


# ---------------------------------------------------------------- 8 位定时器
class Timer8:
    PRESC0 = {0: 0, 1: 1, 2: 8, 3: 64, 4: 256, 5: 1024}             # 6/7 外部时钟：未实现
    PRESC2 = {0: 0, 1: 1, 2: 8, 3: 32, 4: 64, 5: 128, 6: 256, 7: 1024}

    def __init__(self, mcu, n, a_tccra, a_tccrb, a_tcnt, a_ocra, a_ocrb, a_timsk, a_tifr, vec_a, vec_b, vec_ovf,
                 pin_a, pin_b, presc):
        self.mcu, self.n = mcu, n
        self.A = dict(tccra=a_tccra, tccrb=a_tccrb, tcnt=a_tcnt, ocra=a_ocra, ocrb=a_ocrb, timsk=a_timsk, tifr=a_tifr)
        self.addrs = tuple(self.A.values())
        self.vec = (vec_a, vec_b, vec_ovf)
        self.pins = (pin_a, pin_b)        # (Port, bit)
        self.presc = presc
        self.tccra = self.tccrb = self.timsk = self.tifr = 0
        self.ocr = [0, 0]; self.ocr_buf = [0, 0]
        self.cnt = 0; self.tc = 0; self.N = 0
        self.oc = [0, 0]
        self.ovf_count = 0

    # ---- 模式
    @property
    def wgm(self):
        return ((self.tccrb >> 3) & 1) << 2 | (self.tccra & 3)

    def pwm(self):
        return self.wgm in (3, 7)

    def top(self):
        w = self.wgm
        return self.ocr[0] if w in (2, 7) else 0xFF

    def com(self, ch):
        return (self.tccra >> (6 - 2 * ch)) & 3

    def _connected(self, ch):
        c = self.com(ch)
        if c == 0:
            return False
        if self.pwm() and c == 1:
            return ch == 0 and self.wgm == 7     # 模式 7 下 COMxA=1 → 比较匹配翻转；其它 = 断开
        return True

    def _apply_override(self, cyc):
        for ch in (0, 1):
            port, b = self.pins[ch]
            port.override[b] = self.oc[ch] if self._connected(ch) else None
            port.refresh(cyc, (b,))

    def _set_oc(self, ch, v, cyc):
        if self.oc[ch] != v:
            self.oc[ch] = v
            if self._connected(ch):
                port, b = self.pins[ch]
                port.override[b] = v
                port.refresh(cyc, (b,))

    # ---- 寄存器
    def read(self, a, cyc):
        self.sync(cyc)
        A = self.A
        if a == A["tcnt"]: return self.cnt
        if a == A["tccra"]: return self.tccra
        if a == A["tccrb"]: return self.tccrb & 0x0F
        if a == A["ocra"]: return self.ocr_buf[0] if self.pwm() else self.ocr[0]
        if a == A["ocrb"]: return self.ocr_buf[1] if self.pwm() else self.ocr[1]
        if a == A["timsk"]: return self.timsk
        if a == A["tifr"]: return self.tifr
        return 0

    def write(self, a, v, cyc):
        self.sync(cyc)
        A = self.A
        if a == A["tccra"] or a == A["tccrb"]:
            if a == A["tccra"]:
                self.tccra = v
            else:
                self.tccrb = v & 0x0F
                cs = v & 7
                if cs not in self.presc:
                    raise MCUError(f"定时器{self.n}：外部时钟源 CS={cs} 未实现")
                newN = self.presc[cs]
                if newN != self.N:
                    self.N = newN
                    if newN:
                        self.tc = (cyc // newN) * newN      # 预分频器是自由运行的：计数沿在 N 的整数倍周期
            if self.wgm in (1, 4, 5, 6):
                raise MCUError(f"定时器{self.n}：WGM={self.wgm}（相位修正 PWM/保留）未实现，只支持 0/2/3/7")
            self._apply_override(cyc)
        elif a == A["tcnt"]:
            self.cnt = v
            if self.N:
                self.tc = (cyc // self.N) * self.N
        elif a in (A["ocra"], A["ocrb"]):
            ch = 0 if a == A["ocra"] else 1
            self.ocr_buf[ch] = v
            if not self.pwm():
                self.ocr[ch] = v
        elif a == A["timsk"]:
            self.timsk = v & 7
        elif a == A["tifr"]:
            self.tifr &= ~v & 7                    # 写 1 清标志

    # ---- 推进
    def _events(self):
        """下一次事件：(周期, 触发比较的通道, 是否回到 BOTTOM, 回绕前的最大值)。
        比较匹配/回绕都发生在“计数器离开该值”的那个计数沿（手册：OCF 在 TCNT=OCR 之后的下一个定时器时钟置位），
        所以非反相 Fast PWM 的高电平是 OCR+1 个计数，占空比 (OCR+1)/(TOP+1)。"""
        N, tc, cnt = self.N, self.tc, self.cnt
        top = self.top()
        lim = top if cnt <= top else 0xFF          # 计数器已越过 TOP（例如改小了 OCRA）→ 先数到 0xFF 再回绕
        t_wrap = tc + (lim - cnt + 1) * N
        t, fire = t_wrap, []
        for ch in (0, 1):
            v = self.ocr[ch]
            if cnt <= v <= lim:
                tt = tc + (v - cnt + 1) * N
                if tt < t:
                    t, fire = tt, [ch]
                elif tt == t:
                    fire.append(ch)
        return t, fire, t == t_wrap, lim

    def next_event(self):
        if not self.N:
            return INF
        return self._events()[0]

    def sync(self, cyc):
        while self.N:
            t, fire, wrap, lim = self._events()
            if t > cyc:
                k = (cyc - self.tc) // self.N
                if k > 0:
                    self.cnt += k; self.tc += k * self.N
                return
            pwm, w = self.pwm(), self.wgm
            want = {}
            for ch in fire:
                self.tifr |= 2 if ch == 0 else 4
                c = self.com(ch)
                if pwm:
                    if c == 2: want[ch] = 0
                    elif c == 3: want[ch] = 1
                    elif c == 1 and ch == 0 and w == 7: want[ch] = 1 - self.oc[ch]
                else:
                    if c == 1: want[ch] = 1 - self.oc[ch]
                    elif c == 2: want[ch] = 0
                    elif c == 3: want[ch] = 1
            if wrap:
                if w in (0, 3, 7) or (w == 2 and lim == 0xFF):
                    self.tifr |= 1
                    self.ovf_count += 1
                self.cnt = 0; self.tc = t
                if pwm:
                    self.ocr = list(self.ocr_buf)              # 双缓冲：在 BOTTOM 更新
                    for ch in (0, 1):
                        c = self.com(ch)
                        if c == 2: want[ch] = 1                # 非反相：BOTTOM 置 1（OCR=TOP 时与匹配清零同一时刻 → 恒高）
                        elif c == 3: want[ch] = 0
            else:
                self.cnt = self.ocr[fire[0]] + 1; self.tc = t
            for ch, v in want.items():
                self._set_oc(ch, v, t)

    # ---- 中断
    def pending(self):
        m = self.tifr & self.timsk
        out = []
        if m & 2: out.append(self.vec[0])
        if m & 4: out.append(self.vec[1])
        if m & 1: out.append(self.vec[2])
        return out

    def ack(self, vec):
        if vec == self.vec[0]: self.tifr &= ~2
        elif vec == self.vec[1]: self.tifr &= ~4
        elif vec == self.vec[2]: self.tifr &= ~1


# ---------------------------------------------------------------- ADC
class ADC:
    PS = [2, 2, 4, 8, 16, 32, 64, 128]

    def __init__(self, mcu, vec, avcc=5.0, aref=5.0):
        self.mcu, self.vec = mcu, vec
        self.addrs = (0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7E)
        self.admux = self.adcsra = self.adcsrb = self.didr = 0
        self.result = 0
        self.avcc, self.aref = avcc, aref
        self.busy = None        # dict(start, sample, done, ch)
        self.first = True
        self.log = []           # (t_start, t_sample, t_done, ch, vin, vref, code)  周期

    def vref(self):
        r = self.admux >> 6
        return {0: self.aref, 1: self.avcc, 3: 1.1}.get(r, self.avcc)

    def read(self, a, cyc):
        self.sync(cyc)
        if a == 0x78:
            return (self.result << 6) & 0xC0 if self.admux & 0x20 else self.result & 0xFF
        if a == 0x79:
            return self.result >> 2 if self.admux & 0x20 else self.result >> 8
        if a == 0x7A:
            return self.adcsra | (0x40 if self.busy else 0)
        if a == 0x7B: return self.adcsrb
        if a == 0x7C: return self.admux
        return self.didr

    def write(self, a, v, cyc):
        self.sync(cyc)
        if a == 0x7A:
            if v & 0x10:
                self.adcsra &= ~0x10            # 写 1 清 ADIF
            was_en = self.adcsra & 0x80
            self.adcsra = (self.adcsra & 0x10) | (v & 0xAF)
            if not (v & 0x80):
                self.busy = None; self.first = True
            elif not was_en:
                self.first = True
            if v & 0x40 and v & 0x80 and not self.busy:
                ck = self.PS[v & 7]
                n, s = (25, 13.5) if self.first else (13, 1.5)
                self.busy = dict(start=cyc, sample=cyc + int(round(s * ck)), done=cyc + n * ck, ch=self.admux & 0x0F, sampled=None)
                self.first = False
        elif a == 0x7C:
            self.admux = v
        elif a == 0x7B:
            self.adcsrb = v
        elif a == 0x7E:
            self.didr = v

    def quantize(self, vin, vref):
        code = int(vin * 1024.0 / vref) if vin > 0 else 0       # 手册：ADC = Vin·1024/Vref（取整）
        return max(0, min(1023, code))

    def next_event(self):
        b = self.busy
        if not b:
            return INF
        return b["sample"] if b["sampled"] is None else b["done"]

    def sync(self, cyc):
        b = self.busy
        if not b:
            return
        if b["sampled"] is None and cyc >= b["sample"]:
            b["sampled"] = self.mcu.analog_in(b["ch"], b["sample"])
        if cyc >= b["done"]:
            vr = self.vref()
            self.result = self.quantize(b["sampled"], vr)
            self.adcsra |= 0x10
            self.log.append((b["start"], b["sample"], b["done"], b["ch"], b["sampled"], vr, self.result))
            self.busy = None

    def pending(self):
        return [self.vec] if (self.adcsra & 0x18) == 0x18 else []

    def ack(self, vec):
        if vec == self.vec:
            self.adcsra &= ~0x10


# ---------------------------------------------------------------- UART
class UART:
    def __init__(self, mcu, txd, rxd, vec_rx, vec_udre, vec_tx):
        self.mcu = mcu
        self.addrs = (0xC0, 0xC1, 0xC2, 0xC4, 0xC5, 0xC6)
        self.txd, self.rxd = txd, rxd       # (Port, bit)
        self.vec = (vec_rx, vec_udre, vec_tx)
        self.ucsra = 0x20; self.ucsrb = 0; self.ucsrc = 0x06; self.ubrr = 0
        self.tx_buf = None; self.frame = None        # frame: dict(edges=[(cyc, lvl)], end, byte)
        self.rx_fifo = []; self.rx_pending = []      # (done_cyc, byte)
        self.tx_log = []                              # (start_cyc, end_cyc, byte)

    def bit_cycles(self):
        return (8 if self.ucsra & 2 else 16) * (self.ubrr + 1)

    def baud(self):
        return self.mcu.f / self.bit_cycles()

    def nbits(self):
        return 5 + ((self.ucsrc >> 1) & 3)

    def read(self, a, cyc):
        self.sync(cyc)
        if a == 0xC0:
            return (self.ucsra & ~0x80) | (0x80 if self.rx_fifo else 0)
        if a == 0xC1: return self.ucsrb
        if a == 0xC2: return self.ucsrc
        if a == 0xC4: return self.ubrr & 0xFF
        if a == 0xC5: return self.ubrr >> 8
        if a == 0xC6:
            return self.rx_fifo.pop(0) if self.rx_fifo else 0
        return 0

    def write(self, a, v, cyc):
        self.sync(cyc)
        if a == 0xC0:
            if v & 0x40: self.ucsra &= ~0x40         # 写 1 清 TXC
            self.ucsra = (self.ucsra & ~0x03) | (v & 0x03)
        elif a == 0xC1:
            self.ucsrb = v
            port, b = self.txd
            if v & 0x08:
                port.force_out[b] = True; port.override[b] = 1 if not self.frame else port.override[b]
            else:
                port.force_out[b] = False; port.override[b] = None
            port.refresh(cyc, (b,))
        elif a == 0xC2:
            if v & 0x30:
                raise MCUError("UART 奇偶校验未实现（UPM0 必须为 0）")
            self.ucsrc = v
        elif a == 0xC4:
            self.ubrr = (self.ubrr & 0xF00) | v
        elif a == 0xC5:
            self.ubrr = (self.ubrr & 0xFF) | ((v & 0x0F) << 8)
        elif a == 0xC6:
            if not self.ucsrb & 0x08:
                return
            if self.frame is None:
                self._start(v, cyc)
            else:
                self.tx_buf = v
                self.ucsra &= ~0x20                   # UDRE=0

    def _start(self, byte, cyc):
        T = self.bit_cycles(); nb = self.nbits(); stop = 2 if self.ucsrc & 0x08 else 1
        bits = [0] + [(byte >> i) & 1 for i in range(nb)] + [1] * stop
        edges = []; lvl = 1
        for i, bv in enumerate(bits):
            if bv != lvl:
                edges.append((cyc + i * T, bv)); lvl = bv
        self.frame = dict(edges=edges, end=cyc + len(bits) * T, byte=byte & ((1 << nb) - 1), start=cyc)
        self.ucsra |= 0x20

    def inject_rx(self, byte, cyc):
        """外部设备在周期 cyc 开始发送一个字节（同一波特率）。RXC 在停止位中间置位。"""
        self.rx_pending.append((cyc + int((1 + self.nbits() + 0.5) * self.bit_cycles()), byte & 0xFF))
        self.rx_pending.sort()

    def next_event(self):
        t = INF
        if self.frame:
            t = self.frame["edges"][0][0] if self.frame["edges"] else self.frame["end"]
        if self.rx_pending:
            t = min(t, self.rx_pending[0][0])
        return t

    def sync(self, cyc):
        port, b = self.txd
        while self.frame:
            f = self.frame
            if f["edges"] and f["edges"][0][0] <= cyc:
                t, lv = f["edges"].pop(0)
                port.override[b] = lv; port.refresh(t, (b,))
                continue
            if not f["edges"] and f["end"] <= cyc:
                self.tx_log.append((f["start"], f["end"], f["byte"]))
                self.frame = None
                if self.tx_buf is not None:
                    nb, self.tx_buf = self.tx_buf, None
                    self._start(nb, f["end"])
                else:
                    self.ucsra |= 0x40                # TXC
                continue
            break
        while self.rx_pending and self.rx_pending[0][0] <= cyc:
            t, byte = self.rx_pending.pop(0)
            if self.ucsrb & 0x10:
                if len(self.rx_fifo) >= 2:
                    self.ucsra |= 0x08                # DOR 溢出
                else:
                    self.rx_fifo.append(byte)

    def pending(self):
        out = []
        if self.rx_fifo and self.ucsrb & 0x80: out.append(self.vec[0])
        if self.ucsra & 0x20 and self.ucsrb & 0x20: out.append(self.vec[1])
        if self.ucsra & 0x40 and self.ucsrb & 0x40: out.append(self.vec[2])
        return out

    def ack(self, vec):
        if vec == self.vec[2]:
            self.ucsra &= ~0x40


# ---------------------------------------------------------------- 外部中断 INT0/INT1
class ExtInt:
    def __init__(self, mcu, pins, vecs):
        self.mcu, self.pins, self.vecs = mcu, pins, vecs     # pins: ["PD2", "PD3"]
        self.addrs = (0x69, 0x3C, 0x3D)                        # EICRA, EIFR, EIMSK（数据空间地址）
        self.eicra = self.eifr = self.eimsk = 0
        self.lvl = {p: None for p in pins}

    def read(self, a, cyc):
        return {0x69: self.eicra, 0x3C: self.eifr, 0x3D: self.eimsk}[a]

    def write(self, a, v, cyc):
        if a == 0x69: self.eicra = v & 0x0F
        elif a == 0x3C: self.eifr &= ~v & 3
        else: self.eimsk = v & 3

    def on_pin(self, name, level):
        if name not in self.lvl:
            return
        n = self.pins.index(name)
        old = self.lvl[name]; self.lvl[name] = level
        if old is None or old == level:
            return
        isc = (self.eicra >> (2 * n)) & 3
        if isc == 1 or (isc == 2 and level == 0) or (isc == 3 and level == 1):
            self.eifr |= 1 << n

    def sync(self, cyc): pass
    def next_event(self): return INF

    def pending(self):
        out = []
        for n, p in enumerate(self.pins):
            if not self.eimsk >> n & 1:
                continue
            isc = (self.eicra >> (2 * n)) & 3
            if isc == 0 and self.lvl[p] == 0:
                out.append(self.vecs[n])
            elif isc and self.eifr >> n & 1:
                out.append(self.vecs[n])
        return out

    def ack(self, vec):
        for n, v in enumerate(self.vecs):
            if v == vec:
                self.eifr &= ~(1 << n)
