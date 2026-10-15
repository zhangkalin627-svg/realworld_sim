# -*- coding: utf-8 -*-
"""FX3U 指令子集解释器（真实的世界 rw.plc，含 STL/RET 步进、100 ms/10 ms 定时器）：读 GX Works2 指令表文本，逐条执行。
覆盖：LD LDI LDP LDF AND ANI ANDP ANDF OR ORI ORP ORF ANB ORB MPS MRD MPP
      OUT(Y/M/T/C) SET RST PLS PLF MOV CMP ZRST SFTL INC INCP DINC DINCP ALT ALTP END
      STL RET（步进梯形图；同一 Y 可以在不同步里 OUT）
扫描模型：输入映像刷新 → 顺序执行 → 输出映像刷新；100ms 定时器按扫描时间累计；
特殊 M：M8000 常ON、M8001 常OFF、M8002 首扫描、M8012 100ms 时钟、M8013 1s 时钟。
"""
import re

OUTPUT_OPS = {'OUT', 'SET', 'RST', 'PLS', 'PLF', 'MOV', 'CMP', 'ZRST', 'SFTL', 'INC', 'INCP',
              'DINC', 'DINCP', 'ALT', 'ALTP'}
LOAD_OPS = {'LD', 'LDI', 'LDP', 'LDF'}


def split_dev(d):
    m = re.fullmatch(r'([XYMSTCD])(\d+)', d)
    if not m:
        raise ValueError('bad device ' + d)
    return m.group(1), m.group(2)


def dev_offset(d, k):
    t, n = split_dev(d)
    if t in 'XY':
        v = int(n, 8) + k
        return '%s%03o' % (t, v)
    return '%s%d' % (t, int(n) + k)


CONTACT_OPS = LOAD_OPS | {'AND', 'ANI', 'ANDP', 'ANDF', 'OR', 'ORI', 'ORP', 'ORF'}
ZERO_OPS = {'ANB', 'ORB', 'MPS', 'MRD', 'MPP', 'RET', 'END'}
BIT1_OPS = {'SET', 'RST', 'PLS', 'PLF', 'ALT', 'ALTP'}
WORD1_OPS = {'INC', 'INCP', 'DINC', 'DINCP'}


def _check_dev(tok, ln, writable=False):
    """软元件名合法性：形状 + X/Y 八进制位 + 输入写守卫，报错带原程序行号。"""
    try:
        t, n = split_dev(tok)
    except ValueError:
        raise ValueError(f"第{ln}行：非法软元件 {tok}")
    if t in 'XY' and any(c not in '01234567' for c in n):
        raise ValueError(f"第{ln}行：{t} 元件地址须为八进制（0-7），收到 {tok}")
    if writable and t == 'X':
        raise ValueError(f"第{ln}行：不能写输入 {tok}")
    # FX3U 型号范围守卫：T/C/D 编号越界即拒（dict 存储不会自动报错，真机编程工具直接拒）
    if t == 'T' and not 0 <= int(n) <= 255:
        raise ValueError(f"第{ln}行：T 元件编号须在 T0-T255，收到 {tok}")
    if t == 'C' and not 0 <= int(n) <= 255:
        raise ValueError(f"第{ln}行：C 元件编号须在 C0-C255，收到 {tok}")
    if t == 'D' and not 0 <= int(n) <= 8511:
        raise ValueError(f"第{ln}行：D 元件编号须在 D0-D8511，收到 {tok}")
    return t


def _check_word_or_const(tok, ln):
    """字元件 D 或 K/H 常数（含负数十进制）。"""
    if tok and tok[0] in 'KH':
        try:
            int(tok[1:], 16) if tok[0] == 'H' else int(tok[1:])
        except ValueError:
            raise ValueError(f"第{ln}行：非法常数 {tok}")
        return
    t = _check_dev(tok, ln)
    if t != 'D':
        raise ValueError(f"第{ln}行：此处需要字元件 D 或 K/H 常数，收到 {tok}")

class FX3U:
    def __init__(self, il_lines, scan_ms=10):
        self.scan_ms = scan_ms
        self.bits = {}
        self.words = {}
        self.tval = {}      # timer accumulated ms
        self.cval = {}      # counter value
        self.cprev = {}     # counter coil previous
        self.t_ms = 0
        self.first = True
        self.prog = []
        self.prog_lines = []   # 每条指令在原程序里的行号（1 起），报错定位用
        for n, ln in enumerate(il_lines, 1):
            ln = ln.split(';')[0].strip()
            if not ln:
                continue
            parts = ln.split()
            self.prog.append((parts[0].upper(), parts[1:]))
            self.prog_lines.append(n)
        if not self.prog:
            raise ValueError('第1行：程序为空（至少需要 END 指令）')
        if self.prog[-1][0] != 'END':
            raise ValueError('program must end with END')
        # ---- 装载期静态校验：操作数个数 / 软元件与常数格式 / 八进制位 / 输入写守卫 / 栈配平，
        #      一律带原程序行号，不再留到扫描期抛裸 IndexError/ValueError
        for (op, a), ln in zip(self.prog, self.prog_lines):
            n = len(a)
            if op in ZERO_OPS:
                if n:
                    raise ValueError(f"第{ln}行：{op} 不接受操作数")
            elif op in CONTACT_OPS or op == 'STL':
                if n != 1:
                    raise ValueError(f"第{ln}行：{op} 需要 1 个操作数，收到 {n} 个")
                _check_dev(a[0], ln)
            elif op == 'OUT':
                if n < 1:
                    raise ValueError(f"第{ln}行：OUT 缺少目的软元件")
                t = _check_dev(a[0], ln, writable=True)
                if t in 'TC':
                    if n != 2:
                        raise ValueError(f"第{ln}行：OUT {a[0]} 需要设定值（K/H 常数或 D 元件）")
                    _check_word_or_const(a[1], ln)
                    # 预置值语境拒绝负 K：负预置会让 tval>=预置 恒真、定时器/计数器首扫描即动作
                    if a[1].startswith('K') and int(a[1][1:]) < 0:
                        raise ValueError(f"第{ln}行：OUT {a[0]} 预置值不能为负，收到 {a[1]}")
                elif n != 1:
                    raise ValueError(f"第{ln}行：OUT {a[0]} 只接受 1 个操作数")
            elif op in BIT1_OPS:
                if n != 1:
                    raise ValueError(f"第{ln}行：{op} 需要 1 个操作数，收到 {n} 个")
                _check_dev(a[0], ln, writable=True)
            elif op in WORD1_OPS:
                if n != 1:
                    raise ValueError(f"第{ln}行：{op} 需要 1 个操作数，收到 {n} 个")
                if _check_dev(a[0], ln, writable=True) != 'D':
                    raise ValueError(f"第{ln}行：{op} 目的须为字元件 D，收到 {a[0]}")
            elif op == 'MOV':
                if n != 2:
                    raise ValueError(f"第{ln}行：MOV 需要 2 个操作数，收到 {n} 个")
                _check_word_or_const(a[0], ln)
                if _check_dev(a[1], ln, writable=True) != 'D':
                    raise ValueError(f"第{ln}行：MOV 目的须为字元件 D，收到 {a[1]}")
            elif op == 'CMP':
                if n != 3:
                    raise ValueError(f"第{ln}行：CMP 需要 3 个操作数，收到 {n} 个")
                _check_word_or_const(a[0], ln)
                _check_word_or_const(a[1], ln)
                _check_dev(a[2], ln, writable=True)
            elif op == 'ZRST':
                if n != 2:
                    raise ValueError(f"第{ln}行：ZRST 需要 2 个操作数，收到 {n} 个")
                t1 = _check_dev(a[0], ln, writable=True)
                t2 = _check_dev(a[1], ln, writable=True)
                if t1 != t2:
                    raise ValueError(f"第{ln}行：ZRST 起止软元件类型须一致，收到 {a[0]} {a[1]}")
            elif op == 'SFTL':
                if n != 4:
                    raise ValueError(f"第{ln}行：SFTL 需要 4 个操作数，收到 {n} 个")
                _check_dev(a[0], ln)
                _check_dev(a[1], ln, writable=True)
                for k in (2, 3):
                    if not a[k].startswith('K') or not a[k][1:].isdigit():
                        raise ValueError(f"第{ln}行：SFTL 位数须为 K 常数，收到 {a[k]}")
            else:
                raise ValueError(f"第{ln}行：未知指令 {op}")
        # ---- 栈配平：ANB/ORB 需要块内先有分支，MPS/MRD/MPP 需要配对（装载期发现，不留给扫描期）
        blk = 0
        mps = 0
        fresh = True
        for (op, a), ln in zip(self.prog, self.prog_lines):
            if op in LOAD_OPS:
                if fresh:
                    blk = 0
                    mps = 0
                    fresh = False
                else:
                    blk += 1
            elif op in ('ANB', 'ORB'):
                if blk < 1:
                    raise ValueError(f"第{ln}行：{op} 前没有可组合的触点块（检查 LD 块结构）")
                blk -= 1
            elif op == 'MPS':
                mps += 1
            elif op in ('MRD', 'MPP'):
                if mps < 1:
                    raise ValueError(f"第{ln}行：{op} 栈为空（MPS/MPP 未配平）")
                if op == 'MPP':
                    mps -= 1
            elif op in OUTPUT_OPS or op in ('STL', 'RET'):
                fresh = True
        self.edge = [False] * len(self.prog)
        self.used_ops = sorted({op for op, _ in self.prog})

    # ---------------- device access
    def get(self, d):
        if d == 'M8000':
            return True
        if d == 'M8001':
            return False
        if d == 'M8002':
            return self.first
        if d == 'M8012':
            return (self.t_ms % 100) < 50
        if d == 'M8013':
            return (self.t_ms % 1000) < 500
        return self.bits.get(d, False)

    def val(self, s):
        if s[0] == 'K':
            return int(s[1:])
        if s[0] == 'H':
            return int(s[1:], 16)
        return self.words.get(s, 0)

    def setbit(self, d, v):
        if d[0] in 'X':
            raise ValueError('cannot write input ' + d)
        self.bits[d] = v

    # ---------------- one scan
    def scan(self, inputs):
        for k, v in inputs.items():
            self.bits[k] = bool(v)
        acc = False
        blk = []
        mps = []
        prev_out = True
        gate = True          # STL 步进母线
        cur_stl = None
        stl_outs = set()     # STL 块里 OUT 的 Y/M（允许在不同步里重复使用）
        stl_true = set()
        get = self.get
        bits = self.bits
        for i, (op, a) in enumerate(self.prog):
            if op in LOAD_OPS:
                if not prev_out:
                    blk.append(acc)
                else:
                    blk = []
                    mps = []
                v = get(a[0])
                if op == 'LD':
                    acc = gate and v
                elif op == 'LDI':
                    acc = gate and not v
                elif op == 'LDP':
                    acc = gate and v and not self.edge[i]; self.edge[i] = v
                else:
                    acc = gate and (not v) and self.edge[i]; self.edge[i] = v
                prev_out = False
                continue
            if op == 'AND':
                acc = acc and get(a[0])
            elif op == 'ANI':
                acc = acc and not get(a[0])
            elif op == 'ANDP':
                v = get(a[0]); acc = acc and v and not self.edge[i]; self.edge[i] = v
            elif op == 'ANDF':
                v = get(a[0]); acc = acc and (not v) and self.edge[i]; self.edge[i] = v
            elif op == 'OR':
                acc = acc or (gate and get(a[0]))
            elif op == 'ORI':
                acc = acc or (gate and not get(a[0]))
            elif op == 'ORP':
                v = get(a[0]); acc = acc or (gate and v and not self.edge[i]); self.edge[i] = v
            elif op == 'ORF':
                v = get(a[0]); acc = acc or (gate and (not v) and self.edge[i]); self.edge[i] = v
            elif op == 'ANB':
                if not blk:
                    raise ValueError(f"第{self.prog_lines[i]}行：ANB 块栈为空（检查 LD 块结构）")
                acc = blk.pop() and acc
            elif op == 'ORB':
                if not blk:
                    raise ValueError(f"第{self.prog_lines[i]}行：ORB 块栈为空（检查 LD 块结构）")
                acc = blk.pop() or acc
            elif op == 'MPS':
                mps.append(acc)
            elif op == 'MRD':
                if not mps:
                    raise ValueError(f"第{self.prog_lines[i]}行：MRD 栈为空（MPS/MPP 未配平）")
                acc = mps[-1]
            elif op == 'MPP':
                if not mps:
                    raise ValueError(f"第{self.prog_lines[i]}行：MPP 栈为空（MPS/MPP 未配平）")
                acc = mps.pop()
            elif op == 'STL':
                cur_stl = a[0]
                gate = get(a[0])
                acc = gate
                blk = []; mps = []
                prev_out = False
                continue
            elif op == 'RET':
                gate = True; cur_stl = None
                prev_out = True
                continue
            elif op == 'END':
                break
            elif op in OUTPUT_OPS:
                if op == 'OUT' and cur_stl and a[0][0] in 'YM':
                    stl_outs.add(a[0])
                    if acc:
                        stl_true.add(a[0])
                    prev_out = True
                    continue
                if op == 'SET' and cur_stl and a[0][0] == 'S' and acc and a[0] != cur_stl:
                    self.setbit(a[0], True)          # 步转移：置下一步、复位本步
                    self.bits[cur_stl] = False
                    prev_out = True
                    continue
                self._exec_out(i, op, a, acc)
                prev_out = True
                continue
            else:
                raise NotImplementedError(f"第{self.prog_lines[i]}行：未知指令 {op}")
            prev_out = False
        for d in stl_outs:
            self.bits[d] = d in stl_true
        self.first = False
        self.t_ms += self.scan_ms

    def _rise(self, i, acc):
        r = acc and not self.edge[i]
        self.edge[i] = acc
        return r

    def _exec_out(self, i, op, a, acc):
        d = a[0]
        if op == 'OUT':
            t = d[0]
            if t == 'T':
                n = int(d[1:])
                retentive = 246 <= n <= 255          # 积算型：断开保持累计值
                if acc:
                    self.tval[d] = self.tval.get(d, 0) + self.scan_ms
                elif not retentive:
                    self.tval[d] = 0
                unit = 10 if 200 <= n <= 245 else (1 if 246 <= n <= 249 else 100)
                # 积算型触点达预置后锁存，须 RST 复位；非积算触点随 acc 断开回落
                if retentive:
                    self.bits[d] = self.tval[d] >= self.val(a[1]) * unit
                else:
                    self.bits[d] = acc and self.tval[d] >= self.val(a[1]) * unit
            elif t == 'C':
                prev = self.cprev.get(d, False)
                if acc and not prev:
                    if self.cval.get(d, 0) < self.val(a[1]):
                        self.cval[d] = self.cval.get(d, 0) + 1
                self.cprev[d] = acc
                self.bits[d] = self.cval.get(d, 0) >= self.val(a[1])
            else:
                self.setbit(d, acc)
        elif op == 'SET':
            if acc:
                self.setbit(d, True)
        elif op == 'RST':
            if acc:
                if d[0] == 'C':
                    self.cval[d] = 0; self.bits[d] = False
                elif d[0] == 'T':
                    self.tval[d] = 0; self.bits[d] = False
                elif d[0] == 'D':
                    self.words[d] = 0
                else:
                    self.setbit(d, False)
        elif op == 'PLS':
            self.setbit(d, self._rise(i, acc))
        elif op == 'PLF':
            r = (not acc) and self.edge[i]
            self.edge[i] = acc
            self.setbit(d, r)
        elif op in ('ALT', 'ALTP'):
            go = acc if op == 'ALT' else self._rise(i, acc)
            if go:
                self.setbit(d, not self.get(d))
        elif op in ('INC', 'INCP'):
            go = acc if op == 'INC' else self._rise(i, acc)
            if go:
                v = self.words.get(d, 0) + 1
                if v > 32767:
                    v = -32768
                self.words[d] = v
        elif op in ('DINC', 'DINCP'):
            go = acc if op == 'DINC' else self._rise(i, acc)
            if go:
                hi = dev_offset(d, 1)
                v = (self.words.get(hi, 0) & 0xFFFF) << 16 | (self.words.get(d, 0) & 0xFFFF)
                v = (v + 1) & 0xFFFFFFFF
                lo, h = v & 0xFFFF, v >> 16
                self.words[d] = lo - 65536 if lo > 32767 else lo
                self.words[hi] = h - 65536 if h > 32767 else h
        elif op == 'MOV':
            if acc:
                self.words[a[1]] = self.val(a[0])
        elif op == 'CMP':
            if acc:
                s1, s2 = self.val(a[0]), self.val(a[1])
                self.setbit(a[2], s1 > s2)
                self.setbit(dev_offset(a[2], 1), s1 == s2)
                self.setbit(dev_offset(a[2], 2), s1 < s2)
        elif op == 'ZRST':
            if acc:
                t1, n1 = split_dev(a[0]); t2, n2 = split_dev(a[1])
                if t1 != t2:
                    raise ValueError(f"第{self.prog_lines[i]}行：ZRST 起止软元件类型须一致")
                base = 8 if t1 in 'XY' else 10
                for k in range(int(n2, base) - int(n1, base) + 1):
                    dd = dev_offset(a[0], k)
                    if t1 == 'D':
                        self.words[dd] = 0
                    elif t1 in 'TC':
                        self.bits[dd] = False
                        (self.tval if t1 == 'T' else self.cval)[dd] = 0
                    else:
                        self.setbit(dd, False)
        elif op == 'SFTL':
            if acc:
                s, dst, n1, n2 = a[0], a[1], int(a[2][1:]), int(a[3][1:])
                vals = [self.get(dev_offset(dst, k)) for k in range(n1)]
                src = [self.get(dev_offset(s, k)) for k in range(n2)]
                new = src + vals[:n1 - n2]
                for k in range(n1):
                    self.setbit(dev_offset(dst, k), new[k])
        else:
            raise NotImplementedError(f"第{self.prog_lines[i]}行：未知指令 {op}")

    # ------------- convenience
    def dword(self, d):
        hi = dev_offset(d, 1)
        return (self.words.get(hi, 0) & 0xFFFF) << 16 | (self.words.get(d, 0) & 0xFFFF)
