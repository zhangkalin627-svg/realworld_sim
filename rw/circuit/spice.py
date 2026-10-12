"""薄的 SPICE 封装（MIT）：用 Python 搭电路 → 生成 netlist → 调 ngspice -b → 读 raw 文件成 numpy 波形。

不依赖 PySpice（GPLv3）。ngspice 本身是独立的外部程序（BSD-3 许可证），这里只通过命令行调用它。

    from rw.circuit.spice import Circuit
    c = Circuit("RC")
    c.V("1", "in", "0", pulse=(0, 1, 0, 1e-9, 1e-9, 5e-3, 10e-3))
    c.R("1", "in", "out", 1e3); c.C("1", "out", "0", 1e-6)
    w = c.tran(1e-5, 20e-3)
    t, v = w.time, w["out"]
"""
import os, re, shutil, subprocess, tempfile
import numpy as np

# ---------------------------------------------------------------- 数值 ↔ SPICE 文本
_SUFFIX = [("meg", 1e6), ("mil", 25.4e-6), ("t", 1e12), ("g", 1e9), ("k", 1e3), ("m", 1e-3),
           ("u", 1e-6), ("µ", 1e-6), ("n", 1e-9), ("p", 1e-12), ("f", 1e-15)]


def parse_value(s):
    """'4k7' → 4700，'10u' → 1e-5，'2.2meg' → 2.2e6，'1e-3' → 0.001。认不出的返回 None。"""
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip().lower().replace("Ω", "").replace("ohm", "")
    m = re.fullmatch(r"([-+]?\d*\.?\d+(?:e[-+]?\d+)?)([a-zµ]*)", t)
    if m:
        num, suf = float(m.group(1)), m.group(2)
        for k, f in _SUFFIX:
            if suf.startswith(k):
                return num * f
        return num  # 'v' 'a' 'h' 'f'(法拉 F 已被当作 femto？见下)
    m = re.fullmatch(r"(\d+)([kmrunpµ])(\d+)", t)  # 4k7、2r2、4u7
    if m:
        f = dict(_SUFFIX).get(m.group(2), 1.0) if m.group(2) != "r" else 1.0
        return float(f"{m.group(1)}.{m.group(3)}") * f
    return None


def fmt(v):
    """数值 → SPICE 文本。字符串原样输出（允许 '{par}'、'1k' 等）。"""
    if isinstance(v, str):
        return v
    return "%.6g" % float(v)


# ---------------------------------------------------------------- 元件
class Element:
    def __init__(self, name, nodes, value="", params=None):
        self.name, self.nodes, self.value, self.params = name, [str(n) for n in nodes], value, dict(params or {})

    @property
    def kind(self):
        return self.name[0].upper()

    def line(self):
        p = " ".join(f"{k}={fmt(v)}" for k, v in self.params.items())
        return " ".join(x for x in [self.name, *self.nodes, str(self.value), p] if x != "")


def _src_spec(dc=None, ac=None, pulse=None, sin=None, pwl=None):
    parts = []
    if dc is not None: parts.append(f"DC {fmt(dc)}")
    if ac is not None: parts.append(f"AC {fmt(ac)}")
    if pulse is not None: parts.append("PULSE(" + " ".join(fmt(x) for x in pulse) + ")")
    if sin is not None: parts.append("SIN(" + " ".join(fmt(x) for x in sin) + ")")
    if pwl is not None: parts.append("PWL(" + " ".join(fmt(x) for xy in pwl for x in xy) + ")")
    return " ".join(parts) or "DC 0"


class _Builder:
    """Circuit 和 SubCircuit 共用的加元件方法。名字不带前缀字母时自动补上（R('1') → R1）。"""

    def _add(self, prefix, name, nodes, value="", **params):
        name = str(name)
        if not name.upper().startswith(prefix):
            name = prefix + name
        e = Element(name, nodes, value, params)
        self.elements.append(e)
        return e

    def R(self, name, n1, n2, r, **p): return self._add("R", name, [n1, n2], fmt(r), **p)
    def C(self, name, n1, n2, c, ic=None, **p):
        if ic is not None: p["ic"] = ic
        return self._add("C", name, [n1, n2], fmt(c), **p)
    def L(self, name, n1, n2, l, ic=None, **p):
        if ic is not None: p["ic"] = ic
        return self._add("L", name, [n1, n2], fmt(l), **p)
    def K(self, name, l1, l2, k): return self._add("K", name, [], f"{l1} {l2} {fmt(k)}")
    def V(self, name, np_, nm, dc=None, ac=None, pulse=None, sin=None, pwl=None):
        """pulse=(V1 V2 TD TR TF PW PER)，sin=(VO VA FREQ [TD THETA PHASE])，pwl=[(t,v),...]"""
        return self._add("V", name, [np_, nm], _src_spec(dc, ac, pulse, sin, pwl))
    def I(self, name, np_, nm, dc=None, ac=None, pulse=None, sin=None, pwl=None):
        return self._add("I", name, [np_, nm], _src_spec(dc, ac, pulse, sin, pwl))
    def D(self, name, a, k, model): return self._add("D", name, [a, k], model)
    def Q(self, name, c, b, e, model): return self._add("Q", name, [c, b, e], model)
    def M(self, name, d, g, s, model, b=None, **p):
        """MOSFET。model 是 VDMOS（功率管，3 端）时不要给 b。"""
        return self._add("M", name, [d, g, s] + ([b] if b is not None else []), model, **p)
    def E(self, name, np_, nm, cp, cm, gain): return self._add("E", name, [np_, nm, cp, cm], fmt(gain))
    def G(self, name, np_, nm, cp, cm, gm): return self._add("G", name, [np_, nm, cp, cm], fmt(gm))
    def B(self, name, np_, nm, expr): return self._add("B", name, [np_, nm], expr)
    def S(self, name, np_, nm, cp, cm, model): return self._add("S", name, [np_, nm, cp, cm], model)
    def X(self, name, nodes, subckt, **p): return self._add("X", name, list(nodes), subckt, **p)
    def raw(self, line):
        """直接加一行 SPICE 文本（导入网表时用）。"""
        e = Element(line.split()[0], [], "")
        e.line = lambda l=line: l
        self.elements.append(e)
        return e

    def nodes(self):
        out = set()
        for e in self.elements:
            out.update(e.nodes)
        return sorted(out - {"0"})


class SubCircuit(_Builder):
    def __init__(self, name, pins, params=None):
        self.name, self.pins, self.params, self.elements, self.models = name, list(pins), dict(params or {}), [], {}

    def text(self):
        p = (" params: " + " ".join(f"{k}={fmt(v)}" for k, v in self.params.items())) if self.params else ""
        body = [f".subckt {self.name} {' '.join(self.pins)}{p}"]
        body += [m for m in self.models.values()]
        body += [e.line() for e in self.elements] + [f".ends {self.name}"]
        return "\n".join(body)


# ---------------------------------------------------------------- 通用模型库（典型值，标明“通用”，不是某厂家器件的实测模型）
MODELS = {
    "DGEN": ".model DGEN D(Is=1e-14 Rs=0.01 N=1 Cjo=10p)",  # 通用小信号二极管
    "D1N4148": ".model D1N4148 D(Is=2.52n Rs=0.568 N=1.752 Cjo=4p M=0.4 Tt=20n Bv=100 Ibv=100u)",  # 广泛流传的公开参数
    "D1N4007": ".model D1N4007 D(Is=7.02767n Rs=0.0341512 N=1.80803 Cjo=10p M=0.33 Tt=4.32e-6 Bv=1000 Ibv=5u)",
    "DFAST": ".model DFAST D(Is=1e-12 Rs=0.02 N=1.3 Cjo=200p Tt=50n Bv=600)",  # 通用快恢复/体二极管
    "QNPN": ".model QNPN NPN(Is=6.7f Bf=200 Vaf=74 Br=0.7 Rb=10 Rc=1 Cjc=3.6p Cje=4.5p Tf=0.3n Tr=240n)",  # 2N3904 类
    "QPNP": ".model QPNP PNP(Is=6.5f Bf=180 Vaf=50 Br=4 Rb=10 Rc=2.5 Cjc=4.5p Cje=8p Tf=0.6n Tr=100n)",
    "NMOS_SIG": ".model NMOS_SIG NMOS(Level=1 Vto=1.6 Kp=0.1 Lambda=0.01 Cgso=50p Cgdo=10p)",  # 小信号 NMOS
    # 功率 MOSFET（ngspice VDMOS 模型，通用参数：Vth≈4 V，Rds(on)≈30 mΩ 量级，非某个具体型号）
    "NPWR": ".model NPWR VDMOS(Vto=4 Kp=20 Rd=15m Rs=10m Rg=2 Cgdmax=1.5n Cgdmin=50p Cgs=2n Cjo=800p Is=1e-11 Rb=10m Tt=100n)",
}


def opamp_subckt(name="OPAMP", gain=2e5, gbw=1e6, vsat=13.0, sr=10e6, rout=50):
    """行为级运放（通用，不是某个具体型号）。引脚：inp inn out
    差分输入 → 跨导级（电流限幅 → 压摆率 sr，V/s）→ 主极点（开环增益 gain、增益带宽 gbw）→ 内部节点钳位在 ±vsat
    （避免深饱和后恢复很慢）→ 输出电阻 rout。"""
    s = SubCircuit(name, ["inp", "inn", "out"])
    gm = 1e-3
    R = gain / gm
    C = 1 / (2 * np.pi * (gbw / gain) * R)
    imax = sr * C
    s.R("in", "inp", "inn", 10e6)
    s.B("gm", "0", "x", f"I={fmt(imax)}*tanh({fmt(gm / imax)}*(V(inp)-V(inn)))")
    s.R("1", "x", "0", R)
    s.C("1", "x", "0", C)
    v = fmt(vsat)
    s.B("clamp", "x", "0", f"I=10*(V(x)-{v})*u(V(x)-{v})+10*(V(x)+{v})*u(-V(x)-{v})")
    s.E("buf", "y", "0", "x", "0", 1)
    s.R("o", "y", "out", rout)
    return s


# ---------------------------------------------------------------- 找 ngspice
class SpiceError(RuntimeError):
    pass


def find_ngspice():
    cand = [os.environ.get("RW_NGSPICE"), shutil.which("ngspice")]
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(2, 6):  # 沙盒里解压的版本：<项目>/.sandbox/ngspice/usr/bin/ngspice
        root = os.path.abspath(os.path.join(here, *[".."] * up))
        cand.append(os.path.join(root, ".sandbox", "ngspice", "usr", "bin", "ngspice"))
    for c in cand:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _env_for(exe):
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.dirname(exe)))  # .../usr/bin/ngspice → ...
    share = os.path.join(root, "usr", "share", "ngspice")
    if not shutil.which("ngspice") or os.path.realpath(shutil.which("ngspice")) != os.path.realpath(exe):
        if os.path.isdir(share):
            env.setdefault("SPICE_LIB_DIR", share)
            env.setdefault("SPICE_SCRIPTS", os.path.join(share, "scripts"))
        lib = os.path.join(root, "usr", "lib", "aarch64-linux-gnu")
        if os.path.isdir(lib):
            env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + lib
    return env


def ngspice_version():
    exe = find_ngspice()
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-v"], capture_output=True, text=True, timeout=10, env=_env_for(exe)).stdout
        m = re.search(r"ngspice-(\S+)", out)
        return m.group(1) if m else "?"
    except Exception:
        return None


# ---------------------------------------------------------------- raw 文件
def read_raw(path):
    """读 ngspice 的 raw 文件（binary 或 ascii）。返回 (plotname, {变量名: 数组}, {变量名: 类型})。"""
    data = open(path, "rb").read()
    head_end = None
    for key in (b"Binary:\n", b"Values:\n"):
        i = data.find(key)
        if i >= 0:
            head_end, mode = i + len(key), key
            break
    if head_end is None:
        raise SpiceError("raw 文件格式不认识: " + path)
    head = data[:head_end].decode("latin-1").splitlines()
    info, names, types = {}, [], []
    it = iter(head)
    for line in it:
        if line.startswith("Variables:"):
            break
        if ":" in line:
            k, v = line.split(":", 1); info[k.strip()] = v.strip()
    nv, npnt = int(info["No. Variables"]), int(info["No. Points"])
    for _ in range(nv):
        f = next(it).split()
        names.append(f[1].lower()); types.append(f[2])
    cplx = "complex" in info.get("Flags", "")
    if mode == b"Binary:\n":
        dt = np.complex128 if cplx else np.float64
        arr = np.frombuffer(data[head_end:head_end + npnt * nv * np.dtype(dt).itemsize], dtype=dt).reshape(npnt, nv)
    else:
        tok = data[head_end:].decode("latin-1").split()
        vals = []
        k = 0
        for _ in range(npnt):
            k += 1  # 点序号
            row = []
            for _ in range(nv):
                s = tok[k]; k += 1
                if cplx:
                    a, b = s.split(","); row.append(complex(float(a), float(b)))
                else:
                    row.append(float(s))
            vals.append(row)
        arr = np.array(vals)
    out = {n: arr[:, i].copy() for i, n in enumerate(names)}
    if cplx and names:  # 复数结果里扫描变量（频率）是实数
        out[names[0]] = out[names[0]].real
    return info.get("Plotname", ""), out, dict(zip(names, types))


class CIDict(dict):
    """v0.11.1：信号名大小写不敏感的字典。存储键统一小写，self.orig 记住原始写法（显示用）。"""

    def __init__(self, src=None):
        super().__init__(); self.orig = {}
        for k, v in (src or {}).items():
            self[k] = v

    @staticmethod
    def _k(k): return k.lower() if isinstance(k, str) else k
    def __setitem__(self, k, v):
        lk = self._k(k); self.orig.setdefault(lk, k); super().__setitem__(lk, v)
    def __getitem__(self, k): return super().__getitem__(self._k(k))
    def __contains__(self, k): return super().__contains__(self._k(k))
    def __delitem__(self, k): super().__delitem__(self._k(k)); self.orig.pop(self._k(k), None)
    def get(self, k, d=None): return super().get(self._k(k), d)
    def pop(self, k, *d): self.orig.pop(self._k(k), None); return super().pop(self._k(k), *d)
    def setdefault(self, k, d=None):
        if k not in self: self[k] = d
        return self[k]
    def update(self, *a, **kw):
        for k, v in dict(*a, **kw).items(): self[k] = v
    def display(self, k): return self.orig.get(self._k(k), k)


class Waveforms:
    """仿真结果。w['out'] 取节点电压，w['a-b'] 两节点差，w['i(V1)'] 取电压源电流，w.x 扫描变量（时间/频率/直流）。
    信号名存储和查找都大小写不敏感（v0.11.1），原始名保留在 w.data.orig 里用于显示。"""

    def __init__(self, plotname, data, types, netlist="", log=""):
        data = data if isinstance(data, CIDict) else CIDict(data)
        types = types if isinstance(types, CIDict) else CIDict(types or {})
        self.plotname, self.data, self.types, self.netlist, self.log = plotname, data, types, netlist, log
        self.warnings = []
        self.xname = next(iter(data))
        self.x = np.real(data[self.xname])

    @property
    def time(self): return self.x
    @property
    def freq(self): return self.x

    def names(self):
        return [n for n in self.data if n != self.xname]

    def nets(self):
        """可选的信号名（节点电压 + 支路电流），给示波器下拉框用。"""
        out = []
        for n in self.names():
            n = self.data.display(n)
            if n.lower().endswith("#branch"):
                out.append(f"i({n[:-7]})")
            elif n.lower().startswith("v(") and n.endswith(")"):
                out.append(n[2:-1])
            elif self.types.get(n) in ("voltage",) or "#" not in n:
                out.append(n)
        return out

    def _one(self, name):
        n = name.strip().lower()
        if n in ("0", "gnd"):
            return np.zeros_like(self.x)
        for k in (n, f"v({n})"):
            if k in self.data:
                return self.data[k]
        m = re.fullmatch(r"i\((.+)\)", n)
        if m:
            k = m.group(1) + "#branch"
            if k in self.data:
                return self.data[k]
        raise KeyError(f"结果里没有信号 '{name}'。可选：{', '.join(self.nets()[:30])}")

    def __getitem__(self, expr):
        """'out' / 'a-b'（差分）/ 'i(V1)'。节点名本身带 '-'（KiCad 网名常见）时优先当作整个名字。"""
        e = expr.strip()
        try:
            return self._one(e)
        except KeyError:
            pass
        depth, cut = 0, -1
        for i, ch in enumerate(e):
            depth += ch == "("; depth -= ch == ")"
            if ch == "-" and depth == 0 and i > 0:
                try:
                    a, b = self._one(e[:i]), self._one(e[i + 1:])
                    return a - b
                except KeyError:
                    continue
        return self._one(e)


# ---------------------------------------------------------------- 电路
class Circuit(_Builder):
    def __init__(self, title="circuit"):
        self.title, self.elements, self.models, self.subckts = title, [], {}, {}
        self.options, self.includes, self.params, self.extra = {}, [], {}, []
        self.analyses = []  # 导入网表里带的 .tran/.ac/.dc 行

    def model(self, name_or_line, kind=None, **p):
        """model('D1N4007') 用内置模型库；model('MYD', 'D', Is=1e-14) 自定义；或者直接给 '.model ...' 文本。"""
        if kind is None and name_or_line.lower().startswith(".model"):
            name = name_or_line.split()[1]; self.models[name.upper()] = name_or_line
        elif kind is None:
            self.models[name_or_line.upper()] = MODELS[name_or_line.upper()]
        else:
            self.models[name_or_line.upper()] = f".model {name_or_line} {kind}(" + " ".join(f"{k}={fmt(v)}" for k, v in p.items()) + ")"
        return self

    def subckt(self, s):
        self.subckts[s.name.upper()] = s.text() if isinstance(s, SubCircuit) else str(s)
        return self

    def _auto_models(self):
        used = set()
        for e in self.elements:
            if e.kind in "DQMS" and isinstance(e.value, str):
                used.add(e.value.upper())
        for u in used:
            if u not in self.models and u in MODELS:
                self.models[u] = MODELS[u]

    def excitation(self):
        """v0.11.1：网表里有没有激励。返回 (有/没有/None=不确定, 说明)。
        独立源 V/I（值不全是 DC 0）、行为源 B、带 IC= 的储能元件、.ic 都算激励；有 .include 时看不到里面，返回 None。"""
        found, zero_src = [], []
        lines = [e.line() for e in self.elements]
        for sc in self.subckts.values():
            lines += [l for l in str(sc).splitlines()[1:] if l.strip() and not l.strip().startswith(".")]
        for l in lines:
            f = l.split()
            if not f:
                continue
            k = f[0][0].upper()
            if k in "VI":
                spec = " ".join(f[3:]).lower()
                nums = [parse_value(x) for x in re.findall(r"[-+]?[\d.]+(?:e[-+]?\d+)?[a-zµ]*", spec)]
                if re.search(r"\b(pulse|sin|pwl|exp|sffm|am|ac)\b", spec) or any(x for x in nums if x):
                    found.append(f[0])
                else:
                    zero_src.append(f[0])
            elif k == "B":
                found.append(f[0])
            elif k in "CL" and re.search(r"(?i)\bic=", l):
                found.append(f[0] + "(IC)")
        if any(x.lower().startswith(".ic") for x in self.extra):
            found.append(".ic")
        if found:
            return True, "激励：" + ", ".join(found[:8])
        if self.includes:
            return None, "网表本身没有独立源（.include 文件里可能有）"
        return False, "网表里没有任何独立源/激励" + ("（只有 0 V 电流表 %s）" % ", ".join(zero_src[:5]) if zero_src else "")

    def netlist(self, analysis, outfile=None, save=None):
        self._auto_models()
        L = [self.title]
        L += [f".include {p}" for p in self.includes]
        if self.params:
            L.append(".param " + " ".join(f"{k}={fmt(v)}" for k, v in self.params.items()))
        L += list(self.models.values()) + list(self.subckts.values())
        L += [e.line() for e in self.elements] + list(self.extra)
        o = dict(self.options)
        if o:
            L.append(".options " + " ".join(f"{k}={fmt(v)}" if v is not True else k for k, v in o.items()))
        L.append(analysis)
        if outfile:
            L += [".control", "set filetype=binary", "run",
                  f"write {outfile} " + (" ".join(save) if save else "all"), ".endc"]
        L.append(".end")
        return "\n".join(L) + "\n"

    def run(self, analysis, save=None, timeout=120):
        exe = find_ngspice()
        if not exe:
            raise SpiceError("没找到 ngspice。Ubuntu 上装：sudo apt install ngspice（或设置环境变量 RW_NGSPICE=ngspice 路径）")
        with tempfile.TemporaryDirectory(prefix="rwspice_") as d:
            raw = os.path.join(d, "out.raw")
            net = self.netlist(analysis, raw, save)
            cir = os.path.join(d, "in.cir")
            open(cir, "w").write(net)
            p = subprocess.run([exe, "-b", cir], capture_output=True, text=True, timeout=timeout, env=_env_for(exe), cwd=d)
            log = p.stdout + p.stderr
            if not os.path.exists(raw) or os.path.getsize(raw) < 50:
                err = [l for l in log.splitlines() if "rror" in l or "fatal" in l.lower()]
                raise SpiceError("ngspice 没算出结果：\n" + "\n".join(err[:15] or log.splitlines()[-15:]) + "\n\n--- netlist ---\n" + net)
            name, data, types = read_raw(raw)
        w = Waveforms(name, data, types, net, log)
        ok, why = self.excitation()
        if ok is False:
            w.warnings.append("无激励源，结果全零（%s）" % why)
        else:
            mx = max((float(np.max(np.abs(v))) for k, v in w.data.items() if k != w.xname and len(v)), default=0.0)
            if mx < 1e-20:
                w.warnings.append("所有信号都≈0（最大 %.1e），检查激励源是否接上" % mx)
        return w

    # 常用分析
    def tran(self, step, stop, start=0.0, tmax=None, uic=False, **kw):
        a = f".tran {fmt(step)} {fmt(stop)} {fmt(start)} {fmt(tmax if tmax else step)}" + (" uic" if uic else "")
        return self.run(a, **kw)

    def ac(self, fstart, fstop, points=200, variation="dec", **kw):
        return self.run(f".ac {variation} {int(points)} {fmt(fstart)} {fmt(fstop)}", **kw)

    def dc(self, src, start, stop, step, **kw):
        return self.run(f".dc {src} {fmt(start)} {fmt(stop)} {fmt(step)}", **kw)

    def op(self, **kw):
        return self.run(".op", **kw)
