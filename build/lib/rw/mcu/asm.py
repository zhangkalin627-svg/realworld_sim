# -*- coding: utf-8 -*-
"""AVR 汇编器（子集，真实的世界 rw.mcu，MIT）。

语法按 Atmel/Microchip AVR 汇编器的常见写法：
    .equ NAME = 表达式      .def temp = r16      .org 地址(字)      标号:
    ldi r16, low(RAMEND)    out SPL, r16         ; 注释
表达式支持 0x.. / $.. / 0b.. / 十进制、+ - * / << >> & | ~ ( )、low() high() BV()。
内置 ATmega328P 风格的寄存器名和位名（见 defs.py），不需要 .include。

输出：Program（按字地址存放已解码的指令元组）。不生成 .hex 机器码 —— 这是一个“汇编 → 内部指令表”的
教学/仿真汇编器，不能装载 avr-gcc 编译的固件（见 README 局限）。
"""
import re
from .defs import SYMBOLS

TWO_WORD = {"lds", "sts", "jmp", "call"}
REG_RE = re.compile(r"^r(\d{1,2})$", re.I)

# 每条指令的操作数形式：r=寄存器 k=常数/标号 io=I/O 地址 b=位号 p=指针(X/Y/Z 及 + -)  q=Y+q/Z+q
FORMS = {
    "nop": "", "sei": "", "cli": "", "sleep": "", "ret": "", "reti": "", "ijmp": "", "icall": "", "wdr": "",
    "sec": "", "clc": "", "sez": "", "clz": "", "set": "", "clt": "", "break": "",
    "ldi": "rk", "mov": "rr", "movw": "rr", "add": "rr", "adc": "rr", "sub": "rr", "sbc": "rr", "and": "rr", "or": "rr",
    "eor": "rr", "cp": "rr", "cpc": "rr", "cpse": "rr", "mul": "rr",
    "subi": "rk", "sbci": "rk", "andi": "rk", "ori": "rk", "cpi": "rk", "sbr": "rk", "cbr": "rk",
    "adiw": "rk", "sbiw": "rk",
    "com": "r", "neg": "r", "inc": "r", "dec": "r", "lsr": "r", "ror": "r", "asr": "r", "swap": "r", "push": "r", "pop": "r",
    "clr": "r", "ser": "r", "tst": "r", "lsl": "r", "rol": "r",
    "in": "rk", "out": "kr", "sbi": "kk", "cbi": "kk", "sbis": "kk", "sbic": "kk", "sbrs": "rk", "sbrc": "rk",
    "bst": "rk", "bld": "rk",
    "lds": "rk", "sts": "kr", "ld": "rp", "st": "pr", "ldd": "rq", "std": "qr",
    "rjmp": "k", "rcall": "k", "jmp": "k", "call": "k",
    "breq": "k", "brne": "k", "brcs": "k", "brcc": "k", "brlo": "k", "brsh": "k", "brmi": "k", "brpl": "k",
    "brge": "k", "brlt": "k", "brts": "k", "brtc": "k", "brie": "k", "brid": "k", "brvs": "k", "brvc": "k",
}
ALIAS = {"clr": ("eor", 2), "tst": ("and", 2), "lsl": ("add", 2), "rol": ("adc", 2)}  # 同一寄存器两次


class AsmError(Exception):
    pass


class Program:
    def __init__(self, words, labels, lines, source, symbols):
        self.words = words          # list：words[addr] = (op, a, b, lineno) 或 None
        self.labels = labels        # 标号 → 字地址
        self.lines = lines          # 字地址 → 源代码行号
        self.source = source
        self.symbols = symbols

    @property
    def size_words(self):
        return sum(1 for w in self.words if w is not None) + sum(1 for w in self.words if w and w[0] in TWO_WORD)

    def addr_of_line(self, lineno):
        for a, ln in self.lines.items():
            if ln == lineno:
                return a
        return None


def _eval(expr, syms, lineno):
    e = expr.strip()
    e = re.sub(r"'(.)'", lambda m: str(ord(m.group(1))), e)          # 字符常数 'A'
    e = re.sub(r"\$([0-9a-fA-F]+)", r"0x\1", e)
    e = re.sub(r"\blow\s*\(", "_lo(", e, flags=re.I)
    e = re.sub(r"\bhigh\s*\(", "_hi(", e, flags=re.I)
    e = re.sub(r"\bBV\s*\(", "_bv(", e)
    env = {"_lo": lambda v: v & 0xFF, "_hi": lambda v: (v >> 8) & 0xFF, "_bv": lambda v: 1 << v, "__builtins__": {}}
    names = {}
    for tok in set(re.findall(r"(?<![0-9A-Za-z_])[A-Za-z_][A-Za-z0-9_]*", e)):
        if tok in env or re.fullmatch(r"0x[0-9a-fA-F]+|0b[01]+", tok):
            continue
        if tok in syms:
            names[tok] = syms[tok]
        elif tok.upper() in syms:
            names[tok] = syms[tok.upper()]
        else:
            raise AsmError(f"第 {lineno} 行：未定义的符号 '{tok}'")
    try:
        v = eval(e.replace("/", "//"), env, names)  # noqa: S307  只含数字/符号/白名单函数
    except Exception as ex:
        raise AsmError(f"第 {lineno} 行：表达式 '{expr}' 无法计算（{ex}）")
    return int(v)


def _reg(tok, defs, lineno):
    t = tok.strip()
    t = defs.get(t.lower(), t)
    m = REG_RE.match(t)
    if not m or int(m.group(1)) > 31:
        raise AsmError(f"第 {lineno} 行：'{tok}' 不是寄存器 r0..r31")
    return int(m.group(1))


def _split_ops(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def assemble(source, extra_symbols=None):
    syms = dict(SYMBOLS)
    if extra_symbols:
        syms.update(extra_symbols)
    defs = {}
    labels = {}
    stmts = []          # (addr, op, ops, lineno)
    pc = 0
    # ---------- 第一遍：地址、标号、.equ/.def
    for lineno, raw in enumerate(source.splitlines(), 1):
        line = raw.split(";")[0].split("//")[0].strip()
        if not line:
            continue
        while True:
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
            if not m:
                break
            if m.group(1) in labels:
                raise AsmError(f"第 {lineno} 行：标号 '{m.group(1)}' 重复")
            labels[m.group(1)] = pc
            line = m.group(2).strip()
        if not line:
            continue
        if line.startswith("."):
            d = line.split(None, 1)
            kw = d[0].lower(); rest = d[1] if len(d) > 1 else ""
            if kw in (".equ", ".set"):
                name, val = re.split(r"\s*[=,]\s*", rest, 1)
                syms[name.strip()] = _eval(val, {**syms, **labels}, lineno)
            elif kw == ".def":
                name, reg = re.split(r"\s*=\s*", rest, 1)
                defs[name.strip().lower()] = reg.strip()
            elif kw == ".org":
                pc = _eval(rest, {**syms, **labels}, lineno)
            elif kw in (".include", ".device", ".cseg", ".list", ".nolist", ".exit"):
                pass
            else:
                raise AsmError(f"第 {lineno} 行：不支持的伪指令 {kw}")
            continue
        parts = line.split(None, 1)
        op = parts[0].lower()
        if op not in FORMS:
            raise AsmError(f"第 {lineno} 行：不支持的指令 '{parts[0]}'（支持：{', '.join(sorted(FORMS))}）")
        ops = _split_ops(parts[1]) if len(parts) > 1 else []
        stmts.append((pc, op, ops, lineno))
        pc += 2 if op in TWO_WORD else 1
    # ---------- 第二遍：操作数求值
    allsyms = {**syms, **labels}
    words = [None] * (pc + 2)
    lines = {}
    for addr, op, ops, lineno in stmts:
        form = FORMS[op]
        if op in ALIAS:
            r = _reg(ops[0], defs, lineno) if ops else None
            if len(ops) != 1:
                raise AsmError(f"第 {lineno} 行：{op} 需要 1 个操作数")
            op = ALIAS[op][0]; a, b = r, r
            words[addr] = (op, a, b, lineno); lines[addr] = lineno
            continue
        if len(ops) != len(form):
            raise AsmError(f"第 {lineno} 行：{op} 需要 {len(form)} 个操作数，给了 {len(ops)} 个")
        vals = []
        for f, tok in zip(form, ops):
            if f == "r":
                vals.append(_reg(tok, defs, lineno))
            elif f == "k":
                vals.append(_eval(tok, allsyms, lineno))
            elif f == "p":
                t = tok.replace(" ", "").upper()
                if t not in ("X", "X+", "-X", "Y", "Y+", "-Y", "Z", "Z+", "-Z"):
                    raise AsmError(f"第 {lineno} 行：指针 '{tok}' 只能是 X X+ -X Y Y+ -Y Z Z+ -Z")
                vals.append(t)
            elif f == "q":
                m = re.fullmatch(r"\s*([YZ])\s*\+\s*(.+)", tok, re.I)
                if not m:
                    raise AsmError(f"第 {lineno} 行：'{tok}' 应写成 Y+q 或 Z+q")
                q = _eval(m.group(2), allsyms, lineno)
                if not 0 <= q <= 63:
                    raise AsmError(f"第 {lineno} 行：位移 q={q} 超出 0..63")
                vals.append((m.group(1).upper(), q))
        a = vals[0] if vals else None
        b = vals[1] if len(vals) > 1 else None
        # 范围检查（和真汇编器一样报错，避免“仿真能跑、真片不能跑”）
        if op in ("ldi", "subi", "sbci", "andi", "ori", "cpi", "sbr", "cbr") and a < 16:
            raise AsmError(f"第 {lineno} 行：{op} 只能用 r16..r31")
        if op in ("ldi", "subi", "sbci", "andi", "ori", "cpi", "sbr", "cbr"):
            if not -128 <= b <= 255:
                raise AsmError(f"第 {lineno} 行：立即数 {b} 超出 8 位")
            b &= 0xFF
        if op in ("adiw", "sbiw") and (a not in (24, 26, 28, 30) or not 0 <= b <= 63):
            raise AsmError(f"第 {lineno} 行：{op} 只能用 r24/r26/r28/r30，立即数 0..63")
        if op in ("in",) and not 0 <= b <= 63:
            raise AsmError(f"第 {lineno} 行：IN 的 I/O 地址 {b:#x} 超出 0..0x3F（扩展 I/O 要用 LDS）")
        if op in ("out",) and not 0 <= a <= 63:
            raise AsmError(f"第 {lineno} 行：OUT 的 I/O 地址 {a:#x} 超出 0..0x3F（扩展 I/O 要用 STS）")
        if op in ("sbi", "cbi", "sbis", "sbic") and (not 0 <= a <= 31 or not 0 <= b <= 7):
            raise AsmError(f"第 {lineno} 行：{op} 只能操作 I/O 地址 0..0x1F 的位 0..7")
        if op == "movw" and (a % 2 or b % 2):
            raise AsmError(f"第 {lineno} 行：movw 只能用偶数寄存器")
        if op in ("rjmp", "rcall") and not -2048 <= a - (addr + 1) <= 2047:
            raise AsmError(f"第 {lineno} 行：{op} 跳转距离超出 ±2K 字")
        if op.startswith("br") and op not in ("break",) and not -64 <= a - (addr + 1) <= 63:
            raise AsmError(f"第 {lineno} 行：{op} 跳转距离 {a - (addr + 1)} 超出 -64..+63 字")
        words[addr] = (op, a, b, lineno)
        lines[addr] = lineno
    return Program(words, labels, lines, source, allsyms)
