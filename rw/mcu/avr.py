# -*- coding: utf-8 -*-
"""AVR 8 位 CPU 指令子集模拟器（真实的世界 rw.mcu，MIT，自写）。

- 指令语义与 SREG 标志按《AVR Instruction Set Manual》；周期数按经典 AVRe+ 内核（ATmega328P）：
  ALU 1，LDS/STS/LD/ST/LDD/STD/PUSH/POP/SBI/CBI/ADIW/SBIW/MUL 2，RJMP 2，JMP 3，RCALL 3，CALL 4，RET/RETI 4，
  条件分支 不跳 1 / 跳 2，跳过类（CPSE/SBRS/SBRC/SBIS/SBIC）不跳 1 / 跳过单字 2 / 跳过双字 3。
- 中断：响应至少 4 个周期（压栈 PC、清 I），从睡眠唤醒再加 4 个周期；向量表地址按 ATmega328P；
  RETI/SEI 之后至少执行一条指令才响应下一个中断；向量地址越低优先级越高。
- 存储：32 个通用寄存器、I/O 0x20–0xFF、SRAM 0x0100–0x08FF（2 KB）。
程序来自 rw.mcu.asm.assemble()（内部指令表，不是 .hex）。
"""
from .defs import F_CPU_DEFAULT, PIN_FUNCS
from .periph import Port, Timer8, ADC, UART, ExtInt, MCUError, INF

RAMEND = 0x08FF
C_, Z_, N_, V_, S_, H_, T_, I_ = (1 << i for i in range(8))


class AVR:
    def __init__(self, program, f_cpu=F_CPU_DEFAULT, avcc=5.0):
        self.prog = program
        self.words = program.words
        self.f = float(f_cpu)
        self.r = [0] * 32
        self.mem = bytearray(RAMEND + 1)
        self.pc = 0; self.cyc = 0; self.sreg = 0; self.sp = RAMEND
        self.sleeping = False; self.irq_block = 0; self.irq_dirty = True
        self.halted = None
        self.instr_count = 0
        self.smcr = 0
        self.listeners = []              # fn(name, level, t_s, is_output)
        self.pin_log = {}                # name → [(cyc, level)]
        self.analog = {}                 # ADC 通道 → 电压（常数）或 fn(t_s)
        self.irq_log = []                # (cyc, vector)
        self.trace = None                # 可选：list，记录 (cyc, pc)
        # 外设
        self.B = Port(self, "B", 0x03); self.C = Port(self, "C", 0x06, 7); self.D = Port(self, "D", 0x09)
        self.ports = {"B": self.B, "C": self.C, "D": self.D}
        self.t0 = Timer8(self, 0, 0x44, 0x45, 0x46, 0x47, 0x48, 0x6E, 0x35, 0x1C, 0x1E, 0x20,
                         (self.D, 6), (self.D, 5), Timer8.PRESC0)
        self.t2 = Timer8(self, 2, 0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0x70, 0x37, 0x0E, 0x10, 0x12,
                         (self.B, 3), (self.D, 3), Timer8.PRESC2)
        self.adc = ADC(self, 0x2A, avcc=avcc, aref=avcc)
        self.uart = UART(self, (self.D, 1), (self.D, 0), 0x24, 0x26, 0x28)
        self.eint = ExtInt(self, ["PD2", "PD3"], [0x02, 0x04])
        self.periphs = [self.B, self.C, self.D, self.t0, self.t2, self.adc, self.uart, self.eint]
        self.iomap = {}
        for p in self.periphs:
            for a in p.addrs:
                self.iomap[a] = p
        self.timed = [self.t0, self.t2, self.adc, self.uart]
        self.next_evt = INF
        for p in self.ports.values():
            p.refresh(0)
        self._build_dispatch()

    # ------------------------------------------------------------ 公共接口
    def t(self):
        return self.cyc / self.f

    def cycles(self, seconds):
        return int(round(seconds * self.f))

    def pin(self, name):
        """引脚当前电平（输出或输入都返回电平）。"""
        p = self.ports[name[1]]; return p.level(int(name[2]))

    def is_output(self, name):
        return self.ports[name[1]].is_output(int(name[2]))

    def set_input(self, name, level, cyc=None):
        """外部电路驱动引脚（默认当前周期生效；cyc=显式周期，供周期精确的注入器用）。"""
        p = self.ports[name[1]]; b = int(name[2])
        p.ext[b] = None if level is None else int(bool(level))
        p.refresh(self.cyc if cyc is None else cyc, (b,))
        self.irq_dirty = True

    def analog_in(self, ch, cyc):
        v = self.analog.get(ch, 0.0)
        return float(v(cyc / self.f)) if callable(v) else float(v)

    def _pin_change(self, name, level, cyc, is_out):
        self.pin_log.setdefault(name, []).append((cyc, level))
        self.eint.on_pin(name, level)
        self.irq_dirty = True
        for fn in self.listeners:
            fn(name, level, cyc / self.f, is_out)

    # ------------------------------------------------------------ 数据空间
    def rd(self, a):
        if a < 32:
            return self.r[a]
        if a < 0x100:
            p = self.iomap.get(a)
            if p is not None:
                return p.read(a, self.cyc)
            if a == 0x5F: return self.sreg
            if a == 0x5D: return self.sp & 0xFF
            if a == 0x5E: return self.sp >> 8
            if a == 0x53: return self.smcr
            return self.mem[a]
        if a > RAMEND:
            raise MCUError(f"PC={self.pc:#06x}：读数据地址 {a:#06x} 越界")
        return self.mem[a]

    def wr(self, a, v):
        v &= 0xFF
        if a < 32:
            self.r[a] = v; return
        if a < 0x100:
            p = self.iomap.get(a)
            if p is not None:
                p.write(a, v, self.cyc)
                self.next_evt = min(q.next_event() for q in self.timed)
                self.irq_dirty = True
                return
            if a == 0x5F:
                if v & I_ and not self.sreg & I_:
                    self.irq_dirty = True
                self.sreg = v; return
            if a == 0x5D: self.sp = (self.sp & 0xFF00) | v; return
            if a == 0x5E: self.sp = (self.sp & 0xFF) | (v << 8); return
            if a == 0x53: self.smcr = v; return
            self.mem[a] = v; return
        if a > RAMEND:
            raise MCUError(f"PC={self.pc:#06x}：写数据地址 {a:#06x} 越界")
        self.mem[a] = v

    def push(self, v):
        if self.sp < 0x100:
            raise MCUError(f"PC={self.pc:#06x}：栈溢出（SP={self.sp:#06x}）")
        self.mem[self.sp] = v & 0xFF; self.sp -= 1

    def pop(self):
        self.sp += 1
        if self.sp > RAMEND:
            raise MCUError(f"PC={self.pc:#06x}：栈下溢")
        return self.mem[self.sp]

    def push_pc(self, pc):
        self.push(pc & 0xFF); self.push(pc >> 8)

    def pop_pc(self):
        hi = self.pop(); lo = self.pop(); return hi << 8 | lo

    # ------------------------------------------------------------ 运行
    def _sync_all(self):
        c = self.cyc
        for p in self.timed:
            p.sync(c)
        self.next_evt = min(p.next_event() for p in self.timed)
        self.irq_dirty = True

    def _try_irq(self):
        self.irq_dirty = False
        if not self.sreg & I_ or self.irq_block:
            return False
        best = None
        for p in self.periphs:
            for v in p.pending():
                if best is None or v < best[0]:
                    best = (v, p)
        if best is None:
            return False
        vec, p = best
        p.ack(vec)
        self.cyc += 4 + (4 if self.sleeping else 0)
        self.sleeping = False
        self.push_pc(self.pc)
        self.sreg &= ~I_
        self.pc = vec
        self.irq_log.append((self.cyc, vec))
        if self.cyc >= self.next_evt:
            self._sync_all()
        return True

    def run_until(self, cyc_end, max_instr=None):
        """执行到周期 cyc_end（运行中可能超出最多一条指令的周期数；睡眠时正好停在 cyc_end）。"""
        words, disp = self.words, self.disp
        n = 0
        while self.cyc < cyc_end:
            if self.halted:
                self.cyc = cyc_end; break
            if self.sleeping:
                if self.irq_dirty and self._try_irq():      # 外部引脚刚触发的中断：在当前周期唤醒
                    continue
                nxt = min(self.next_evt, cyc_end)
                if nxt > self.cyc:
                    self.cyc = int(nxt)
                self._sync_all()
                self._try_irq()
                continue
            w = words[self.pc] if self.pc < len(words) else None
            if w is None:
                raise MCUError(f"PC={self.pc:#06x} 处没有指令（程序跑飞？）")
            if self.trace is not None:
                self.trace.append((self.cyc, self.pc))
            self.cyc += disp[w[0]](w[1], w[2])
            self.instr_count += 1
            if self.irq_block:
                self.irq_block -= 1
                if not self.irq_block:
                    self.irq_dirty = True
            if self.cyc >= self.next_evt:
                self._sync_all()
            if self.irq_dirty:
                self._try_irq()
            n += 1
            if max_instr and n >= max_instr:
                break

    def run_for(self, seconds):
        self.run_until(self.cyc + self.cycles(seconds))

    # ------------------------------------------------------------ 指令
    def _build_dispatch(self):
        d = {}
        for name in dir(self):
            if name.startswith("i_"):
                d[name[2:]] = getattr(self, name)
        self.disp = d

    # 标志辅助
    def _nzs(self, R, V):
        s = self.sreg & ~(N_ | Z_ | V_ | S_)
        if R & 0x80: s |= N_
        if R == 0: s |= Z_
        if V: s |= V_
        if bool(R & 0x80) != bool(V): s |= S_
        self.sreg = s

    def _add(self, d, rr, cin):
        a = self.r[d]; R = a + rr + cin; r8 = R & 0xFF
        V = (~(a ^ rr) & (a ^ r8) & 0x80)
        self._nzs(r8, V)
        s = self.sreg & ~(C_ | H_)
        if R > 0xFF: s |= C_
        if (a & 0xF) + (rr & 0xF) + cin > 0xF: s |= H_
        self.sreg = s
        return r8

    def _sub(self, a, b, cin, keepz):
        R = a - b - cin; r8 = R & 0xFF
        V = ((a ^ b) & (a ^ r8) & 0x80)
        zprev = self.sreg & Z_
        self._nzs(r8, V)
        s = self.sreg & ~(C_ | H_)
        if R < 0: s |= C_
        if (a & 0xF) - (b & 0xF) - cin < 0: s |= H_
        if keepz and r8 == 0 and not zprev:
            s &= ~Z_
        self.sreg = s
        return r8

    def _logic(self, R):
        self._nzs(R, 0)
        return R

    def _size(self, addr):
        w = self.words[addr] if addr < len(self.words) else None
        return 2 if w and w[0] in ("lds", "sts", "jmp", "call") else 1

    def _skip(self, cond):
        if cond:
            n = self._size(self.pc + 1)
            self.pc += 1 + n
            return 1 + n
        self.pc += 1
        return 1

    # 无操作数
    def i_nop(self, a, b): self.pc += 1; return 1
    def i_wdr(self, a, b): self.pc += 1; return 1
    def i_break(self, a, b): self.halted = "break"; self.pc += 1; return 1
    def i_sei(self, a, b):
        self.sreg |= I_; self.irq_block = 1; self.pc += 1; return 1
    def i_cli(self, a, b): self.sreg &= ~I_; self.pc += 1; return 1
    def i_sec(self, a, b): self.sreg |= C_; self.pc += 1; return 1
    def i_clc(self, a, b): self.sreg &= ~C_; self.pc += 1; return 1
    def i_sez(self, a, b): self.sreg |= Z_; self.pc += 1; return 1
    def i_clz(self, a, b): self.sreg &= ~Z_; self.pc += 1; return 1
    def i_set(self, a, b): self.sreg |= T_; self.pc += 1; return 1
    def i_clt(self, a, b): self.sreg &= ~T_; self.pc += 1; return 1

    def i_sleep(self, a, b):
        self.pc += 1
        if self.smcr & 1:
            self.sleeping = True
        return 1

    # 寄存器/立即数
    def i_ldi(self, a, b): self.r[a] = b; self.pc += 1; return 1
    def i_mov(self, a, b): self.r[a] = self.r[b]; self.pc += 1; return 1
    def i_movw(self, a, b): self.r[a] = self.r[b]; self.r[a + 1] = self.r[b + 1]; self.pc += 1; return 1
    def i_add(self, a, b): self.r[a] = self._add(a, self.r[b], 0); self.pc += 1; return 1
    def i_adc(self, a, b): self.r[a] = self._add(a, self.r[b], self.sreg & 1); self.pc += 1; return 1
    def i_sub(self, a, b): self.r[a] = self._sub(self.r[a], self.r[b], 0, False); self.pc += 1; return 1
    def i_sbc(self, a, b): self.r[a] = self._sub(self.r[a], self.r[b], self.sreg & 1, True); self.pc += 1; return 1
    def i_subi(self, a, b): self.r[a] = self._sub(self.r[a], b, 0, False); self.pc += 1; return 1
    def i_sbci(self, a, b): self.r[a] = self._sub(self.r[a], b, self.sreg & 1, True); self.pc += 1; return 1
    def i_cp(self, a, b): self._sub(self.r[a], self.r[b], 0, False); self.pc += 1; return 1
    def i_cpc(self, a, b): self._sub(self.r[a], self.r[b], self.sreg & 1, True); self.pc += 1; return 1
    def i_cpi(self, a, b): self._sub(self.r[a], b, 0, False); self.pc += 1; return 1
    def i_and(self, a, b): self.r[a] = self._logic(self.r[a] & self.r[b]); self.pc += 1; return 1
    def i_andi(self, a, b): self.r[a] = self._logic(self.r[a] & b); self.pc += 1; return 1
    def i_cbr(self, a, b): self.r[a] = self._logic(self.r[a] & ~b & 0xFF); self.pc += 1; return 1
    def i_or(self, a, b): self.r[a] = self._logic(self.r[a] | self.r[b]); self.pc += 1; return 1
    def i_ori(self, a, b): self.r[a] = self._logic(self.r[a] | b); self.pc += 1; return 1
    def i_sbr(self, a, b): self.r[a] = self._logic(self.r[a] | b); self.pc += 1; return 1
    def i_eor(self, a, b): self.r[a] = self._logic(self.r[a] ^ self.r[b]); self.pc += 1; return 1
    def i_ser(self, a, b): self.r[a] = 0xFF; self.pc += 1; return 1

    def i_com(self, a, b):
        R = (~self.r[a]) & 0xFF; self._nzs(R, 0); self.sreg |= C_; self.r[a] = R; self.pc += 1; return 1

    def i_neg(self, a, b):
        d = self.r[a]; R = (-d) & 0xFF
        self._nzs(R, R == 0x80)
        s = self.sreg & ~(C_ | H_)
        if R: s |= C_
        if (R | d) & 0x08: s |= H_
        self.sreg = s; self.r[a] = R; self.pc += 1; return 1

    def i_inc(self, a, b):
        d = self.r[a]; R = (d + 1) & 0xFF; self._nzs(R, d == 0x7F); self.r[a] = R; self.pc += 1; return 1

    def i_dec(self, a, b):
        d = self.r[a]; R = (d - 1) & 0xFF; self._nzs(R, d == 0x80); self.r[a] = R; self.pc += 1; return 1

    def _shift_flags(self, R, c):
        n = R >> 7; v = n ^ c
        s = self.sreg & ~(N_ | Z_ | V_ | S_ | C_)
        if n: s |= N_
        if R == 0: s |= Z_
        if v: s |= V_
        if n ^ v: s |= S_
        if c: s |= C_
        self.sreg = s

    def i_lsr(self, a, b):
        d = self.r[a]; R = d >> 1; self._shift_flags(R, d & 1); self.r[a] = R; self.pc += 1; return 1

    def i_ror(self, a, b):
        d = self.r[a]; R = (d >> 1) | ((self.sreg & 1) << 7); self._shift_flags(R, d & 1); self.r[a] = R; self.pc += 1; return 1

    def i_asr(self, a, b):
        d = self.r[a]; R = (d >> 1) | (d & 0x80); self._shift_flags(R, d & 1); self.r[a] = R; self.pc += 1; return 1

    def i_swap(self, a, b):
        d = self.r[a]; self.r[a] = ((d << 4) | (d >> 4)) & 0xFF; self.pc += 1; return 1

    def i_adiw(self, a, b):
        d = self.r[a] | self.r[a + 1] << 8; R = d + b; r16 = R & 0xFFFF
        V = (not d & 0x8000) and (r16 & 0x8000)
        s = self.sreg & ~(N_ | Z_ | V_ | S_ | C_)
        if r16 & 0x8000: s |= N_
        if r16 == 0: s |= Z_
        if V: s |= V_
        if bool(r16 & 0x8000) != bool(V): s |= S_
        if R > 0xFFFF: s |= C_
        self.sreg = s; self.r[a] = r16 & 0xFF; self.r[a + 1] = r16 >> 8; self.pc += 1; return 2

    def i_sbiw(self, a, b):
        d = self.r[a] | self.r[a + 1] << 8; R = d - b; r16 = R & 0xFFFF
        V = (d & 0x8000) and not (r16 & 0x8000)
        s = self.sreg & ~(N_ | Z_ | V_ | S_ | C_)
        if r16 & 0x8000: s |= N_
        if r16 == 0: s |= Z_
        if V: s |= V_
        if bool(r16 & 0x8000) != bool(V): s |= S_
        if R < 0: s |= C_
        self.sreg = s; self.r[a] = r16 & 0xFF; self.r[a + 1] = r16 >> 8; self.pc += 1; return 2

    def i_mul(self, a, b):
        R = self.r[a] * self.r[b]
        self.r[0] = R & 0xFF; self.r[1] = R >> 8
        s = self.sreg & ~(Z_ | C_)
        if R & 0x8000: s |= C_
        if R == 0: s |= Z_
        self.sreg = s; self.pc += 1; return 2

    def i_bst(self, a, b):
        self.sreg = (self.sreg | T_) if self.r[a] >> b & 1 else (self.sreg & ~T_); self.pc += 1; return 1

    def i_bld(self, a, b):
        self.r[a] = (self.r[a] | (1 << b)) if self.sreg & T_ else (self.r[a] & ~(1 << b)); self.pc += 1; return 1

    # I/O 与存储
    def i_in(self, a, b): self.r[a] = self.rd(b + 0x20); self.pc += 1; return 1
    def i_out(self, a, b): self.wr(a + 0x20, self.r[b]); self.pc += 1; return 1

    def i_sbi(self, a, b):
        A = a + 0x20
        self.cyc += 1                                  # 写在第 2 个周期生效
        if A in (0x23, 0x26, 0x29):                    # SBI PINx,b → 只翻转这一位
            self.wr(A, 1 << b)
        else:
            self.wr(A, self.rd(A) | (1 << b))
        self.cyc -= 1
        self.pc += 1; return 2

    def i_cbi(self, a, b):
        self.cyc += 1
        self.wr(a + 0x20, self.rd(a + 0x20) & ~(1 << b))
        self.cyc -= 1
        self.pc += 1; return 2

    def i_sbis(self, a, b): return self._skip(self.rd(a + 0x20) >> b & 1)
    def i_sbic(self, a, b): return self._skip(not (self.rd(a + 0x20) >> b & 1))
    def i_sbrs(self, a, b): return self._skip(self.r[a] >> b & 1)
    def i_sbrc(self, a, b): return self._skip(not (self.r[a] >> b & 1))
    def i_cpse(self, a, b): return self._skip(self.r[a] == self.r[b])

    def i_lds(self, a, b):
        self.cyc += 1; self.r[a] = self.rd(b); self.cyc -= 1; self.pc += 2; return 2

    def i_sts(self, a, b):
        self.cyc += 1; self.wr(a, self.r[b]); self.cyc -= 1; self.pc += 2; return 2

    def _ptr(self, p):
        base = {"X": 26, "Y": 28, "Z": 30}[p.strip("+-")]
        return base, self.r[base] | self.r[base + 1] << 8

    def _setptr(self, base, v):
        v &= 0xFFFF; self.r[base] = v & 0xFF; self.r[base + 1] = v >> 8

    def i_ld(self, a, p):
        base, addr = self._ptr(p)
        if p.startswith("-"):
            addr -= 1; self._setptr(base, addr)
        self.r[a] = self.rd(addr)
        if p.endswith("+"):
            self._setptr(base, addr + 1)
        self.pc += 1; return 2

    def i_st(self, p, b):
        base, addr = self._ptr(p)
        if p.startswith("-"):
            addr -= 1; self._setptr(base, addr)
        self.cyc += 1; self.wr(addr, self.r[b]); self.cyc -= 1
        if p.endswith("+"):
            self._setptr(base, addr + 1)
        self.pc += 1; return 2

    def i_ldd(self, a, q):
        base, addr = self._ptr(q[0]); self.r[a] = self.rd(addr + q[1]); self.pc += 1; return 2

    def i_std(self, q, b):
        base, addr = self._ptr(q[0]); self.cyc += 1; self.wr(addr + q[1], self.r[b]); self.cyc -= 1; self.pc += 1; return 2

    def i_push(self, a, b): self.push(self.r[a]); self.pc += 1; return 2
    def i_pop(self, a, b): self.r[a] = self.pop(); self.pc += 1; return 2

    # 跳转
    def i_rjmp(self, a, b): self.pc = a; return 2
    def i_jmp(self, a, b): self.pc = a; return 3
    def i_rcall(self, a, b): self.push_pc(self.pc + 1); self.pc = a; return 3
    def i_call(self, a, b): self.push_pc(self.pc + 2); self.pc = a; return 4
    def i_ret(self, a, b): self.pc = self.pop_pc(); return 4

    def i_reti(self, a, b):
        self.pc = self.pop_pc(); self.sreg |= I_; self.irq_block = 1; return 4

    def i_ijmp(self, a, b): self.pc = self.r[30] | self.r[31] << 8; return 2

    def i_icall(self, a, b):
        self.push_pc(self.pc + 1); self.pc = self.r[30] | self.r[31] << 8; return 3

    def _br(self, cond, a):
        if cond:
            self.pc = a; return 2
        self.pc += 1; return 1

    def i_breq(self, a, b): return self._br(self.sreg & Z_, a)
    def i_brne(self, a, b): return self._br(not self.sreg & Z_, a)
    def i_brcs(self, a, b): return self._br(self.sreg & C_, a)
    def i_brlo(self, a, b): return self._br(self.sreg & C_, a)
    def i_brcc(self, a, b): return self._br(not self.sreg & C_, a)
    def i_brsh(self, a, b): return self._br(not self.sreg & C_, a)
    def i_brmi(self, a, b): return self._br(self.sreg & N_, a)
    def i_brpl(self, a, b): return self._br(not self.sreg & N_, a)
    def i_brge(self, a, b): return self._br(not self.sreg & S_, a)
    def i_brlt(self, a, b): return self._br(self.sreg & S_, a)
    def i_brts(self, a, b): return self._br(self.sreg & T_, a)
    def i_brtc(self, a, b): return self._br(not self.sreg & T_, a)
    def i_brie(self, a, b): return self._br(self.sreg & I_, a)
    def i_brid(self, a, b): return self._br(not self.sreg & I_, a)
    def i_brvs(self, a, b): return self._br(self.sreg & V_, a)
    def i_brvc(self, a, b): return self._br(not self.sreg & V_, a)

    # ------------------------------------------------------------ 显示用
    def regs_text(self):
        f = "".join(ch if self.sreg >> (7 - i) & 1 else "-" for i, ch in enumerate("ITHSVNZC"))
        rows = []
        for i in range(0, 32, 8):
            rows.append("  ".join(f"r{j:<2}={self.r[j]:02X}" for j in range(i, i + 8)))
        rows.append(f"PC={self.pc:04X}  SP={self.sp:04X}  SREG={f}  周期={self.cyc}  t={self.t()*1e3:.4f} ms"
                    + ("  [睡眠]" if self.sleeping else ""))
        return "\n".join(rows)

    def pins_table(self):
        out = []
        for L, p in self.ports.items():
            for b in range(p.n):
                nm = f"P{L}{b}"
                out.append((nm, PIN_FUNCS.get(nm, ""), "输出" if p.is_output(b) else "输入", p.level(b)))
        return out
