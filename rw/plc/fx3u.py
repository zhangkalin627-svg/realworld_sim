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


class FX3U:
    def __init__(self, il_lines, scan_ms=10):
        self.scan_ms = scan_ms
        self.bits = {}
        self.words = {}
        self.tval = {}      # timer accumulated ms
        self.cval = {}      # counter value
        self.cprev = {}     # counter coil previous
        self.t_ms = 0
        self._tkeep = {}
        self.first = True
        self.prog = []
        for ln in il_lines:
            ln = ln.split(';')[0].strip()
            if not ln:
                continue
            parts = ln.split()
            self.prog.append((parts[0].upper(), parts[1:]))
        if self.prog[-1][0] != 'END':
            raise ValueError('program must end with END')
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
                    acc = v and not self.edge[i]; self.edge[i] = v
                else:
                    acc = (not v) and self.edge[i]; self.edge[i] = v
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
                acc = acc or get(a[0])
            elif op == 'ORI':
                acc = acc or not get(a[0])
            elif op == 'ORP':
                v = get(a[0]); acc = acc or (v and not self.edge[i]); self.edge[i] = v
            elif op == 'ORF':
                v = get(a[0]); acc = acc or ((not v) and self.edge[i]); self.edge[i] = v
            elif op == 'ANB':
                acc = blk.pop() and acc
            elif op == 'ORB':
                acc = blk.pop() or acc
            elif op == 'MPS':
                mps.append(acc)
            elif op == 'MRD':
                acc = mps[-1]
            elif op == 'MPP':
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
                raise NotImplementedError(op)
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
                if acc:
                    self.tval[d] = self.tval.get(d, 0) + self.scan_ms
                else:
                    self.tval[d] = 0
                n = int(d[1:])
                unit = 10 if 200 <= n <= 245 else (1 if 246 <= n <= 249 else 100)
                if 246 <= n <= 255 and not acc:      # 积算型：断开保持
                    self.tval[d] = self.tval.get(d, 0) if acc else self._tkeep.get(d, 0)
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
                assert t1 == t2
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
            raise NotImplementedError(op)

    # ------------- convenience
    def dword(self, d):
        hi = dev_offset(d, 1)
        return (self.words.get(hi, 0) & 0xFFFF) << 16 | (self.words.get(d, 0) & 0xFFFF)
