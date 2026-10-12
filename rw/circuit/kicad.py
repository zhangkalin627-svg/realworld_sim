"""导入 KiCad 的网表，变成 rw.circuit.spice.Circuit。

1) SPICE 网表（.cir / .sp / .spice）：KiCad 原理图编辑器「仿真 → 导出 SPICE 网表」或 eeschema 的 Export Netlist → SPICE。
   元件行基本原样保留（ngspice 认识就能跑），.model/.subckt/.include/.param 收集起来，.tran/.ac/.dc 记到 circuit.analyses，
   .control … .endc 块丢掉（我们自己控制运行和输出）。
2) KiCad 网表（.net，S 表达式，Export Netlist → KiCad）：按位号/库元件/Sim.* 字段映射成 R/C/L/D/Q/M/V/I。
   映射不了的元件（连接器、IC 没有 SPICE 模型等）不会瞎猜，列在 circuit.warnings 里。

GND / 0 / /GND / GNDREF 等接地网络映射成 SPICE 节点 0。
"""
import os, re
from .spice import Circuit, parse_value, MODELS

GROUND = {"0", "gnd", "/gnd", "gndref", "gnda", "gndd", "gndpwr", "earth", "/0"}


def _node(n):
    n = n.strip().strip('"')
    if n.lower() in GROUND:
        return "0"
    # SPICE 节点名里不能有空格/括号；KiCad 默认网名形如 Net-(R1-Pad1)
    return re.sub(r"[\s(){}=,]", "_", n)


# ---------------------------------------------------------------- SPICE 网表
def import_spice(path, text=None):
    text = open(path, encoding="utf-8", errors="replace").read() if text is None else text
    base = os.path.dirname(os.path.abspath(path)) if path else "."
    raw_lines = text.splitlines()
    lines = []
    for l in raw_lines:  # 续行 '+'
        if l.startswith("+") and lines:
            lines[-1] += " " + l[1:]
        else:
            lines.append(l)
    title = lines[0].strip() if lines else "netlist"
    if title.lower().startswith(".title"):
        title = title[6:].strip()
    c = Circuit(title or "netlist")
    c.source_file = path
    c.warnings = []
    in_ctrl, sub = False, None
    for l in lines[1:]:
        s = l.strip()
        if not s or s.startswith("*") or s.startswith(";"):
            continue
        s = re.split(r"\s;", s)[0].strip()  # 行尾注释
        low = s.lower()
        if low.startswith(".control"):
            in_ctrl = True; continue
        if in_ctrl:
            if low.startswith(".endc"):
                in_ctrl = False
            continue
        if sub is not None:
            sub.append(s)
            if low.startswith(".ends"):
                name = sub[0].split()[1]
                c.subckts[name.upper()] = "\n".join(sub); sub = None
            continue
        if low.startswith(".subckt"):
            sub = [s]; continue
        if low.startswith(".model"):
            c.models[s.split()[1].upper()] = s; continue
        if low.startswith((".include", ".inc", ".lib")):
            parts = s.split(None, 1)
            p = parts[1].strip().strip('"') if len(parts) > 1 else ""
            if p and not os.path.isabs(p):
                p = os.path.join(base, p)
            if not os.path.exists(p.split()[0]):
                c.warnings.append(f"找不到 include 文件：{p}")
            c.includes.append(f'"{p}"' if " " in p else p)
            continue
        if low.startswith((".tran", ".ac", ".dc", ".op", ".noise", ".tf")):
            c.analyses.append(s); continue
        if low.startswith(".end"):
            break
        if low.startswith((".save", ".probe", ".print", ".plot", ".meas", ".width")):
            continue
        if low.startswith(".title"):
            c.title = s[6:].strip(); continue
        if s.startswith("."):
            c.extra.append(s); continue  # .param .options .ic .nodeset .global .func .temp …
        f = s.split()
        name = f[0]
        # 接地网名统一成 0：只替换节点位置（按元件类型的节点数）
        nn = {"R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "D": 2, "B": 2, "Q": 3, "J": 3, "M": 4, "E": 4, "G": 4,
              "F": 2, "H": 2, "S": 4, "W": 2, "Z": 3, "T": 4}.get(name[0].upper())
        if name[0].upper() == "X":
            k = len(f) - 1
            while k > 1 and "=" in f[k]:
                k -= 1
            nn = k - 1  # 最后一个非参数字段是子电路名
        if nn:
            f[1:1 + nn] = [_node(x) for x in f[1:1 + nn]]
        c.raw(" ".join(f))
    return c


# ---------------------------------------------------------------- S 表达式
def parse_sexpr(text):
    tok = re.findall(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+', text)
    stack, cur = [], []
    for t in tok:
        if t == "(":
            stack.append(cur); cur = []
        elif t == ")":
            done = cur; cur = stack.pop(); cur.append(done)
        else:
            cur.append(t[1:-1].replace('\\"', '"') if t.startswith('"') else t)
    return cur[0] if cur else []


def _find(node, key):
    return [x for x in node if isinstance(x, list) and x and x[0] == key]


def _get(node, key, default=None):
    r = _find(node, key)
    return r[0][1] if r and len(r[0]) > 1 else default


def import_kicad_net(path, text=None):
    text = open(path, encoding="utf-8", errors="replace").read() if text is None else text
    tree = parse_sexpr(text)
    if not tree or tree[0] != "export":
        raise ValueError("不是 KiCad 网表（应以 (export 开头）")
    design = (_find(tree, "design") or [[]])[0]
    c = Circuit(f"KiCad: {os.path.basename(path) if path else _get(design, 'source', 'netlist')}")
    c.source_file = path
    c.warnings = []
    # 元件 → 字段
    comps = {}
    for comp in _find((_find(tree, "components") or [[]])[0], "comp"):
        ref = _get(comp, "ref")
        fields = {}
        for fs in _find(comp, "fields"):
            for fl in _find(fs, "field"):
                nm = next((x[1] for x in fl if isinstance(x, list) and x[0] == "name"), None)
                val = fl[-1] if isinstance(fl[-1], str) else ""
                if nm:
                    fields[nm] = val
        for pr in _find(comp, "property"):
            fields[_get(pr, "name")] = _get(pr, "value", "")
        lib = (_find(comp, "libsource") or [[]])[0]
        comps[ref] = dict(ref=ref, value=_get(comp, "value", ""), lib=_get(lib, "lib", ""), part=_get(lib, "part", ""),
                          fields=fields, pins={})
    # 网络 → 每个元件每个引脚接哪个节点
    for net in _find((_find(tree, "nets") or [[]])[0], "net"):
        name = _node(_get(net, "name", ""))
        for nd in _find(net, "node"):
            ref, pin = _get(nd, "ref"), _get(nd, "pin")
            if ref in comps:
                comps[ref]["pins"][pin] = dict(net=name, name=_get(nd, "pinfunction", ""))
    # 库元件的引脚名（老版本网表 node 里没有 pinfunction）
    libpins = {}
    for lp in _find((_find(tree, "libparts") or [[]])[0], "libpart"):
        key = (_get(lp, "lib"), _get(lp, "part"))
        libpins[key] = {_get(p, "num"): _get(p, "name") for p in _find((_find(lp, "pins") or [[]])[0], "pin")}
    for ref, cp in sorted(comps.items()):
        try:
            _map_comp(c, cp, libpins.get((cp["lib"], cp["part"]), {}))
        except Exception as e:  # noqa
            c.warnings.append(f"{ref}（{cp['part'] or cp['value']}）：{e}")
    return c


def _pins(cp, libpins, order):
    """按引脚名（或 Sim.Pins 映射）排出 SPICE 节点顺序。order 如 ['C','B','E']。"""
    names = {}
    for num, d in cp["pins"].items():
        nm = (d.get("name") or libpins.get(num) or "").upper()
        names[num] = nm
    simpins = cp["fields"].get("Sim.Pins") or cp["fields"].get("Sim_Pins")
    if simpins:  # 例："1=D 2=G 3=S" 或 "1=+ 2=-"
        mp = dict(x.split("=", 1) for x in simpins.split() if "=" in x)
        names = {num: mp.get(num, names.get(num, "")).upper() for num in cp["pins"]}
    out = []
    for o in order:
        hit = [num for num, nm in names.items() if nm in (o, o.replace("+", "A").replace("-", "K"))]
        if not hit:
            hit = [num for num, nm in names.items() if nm.startswith(o)]
        if not hit:
            raise ValueError(f"找不到引脚 {o}（引脚名：{names}）")
        out.append(cp["pins"][hit[0]]["net"])
    return out


def _two(cp):
    nums = sorted(cp["pins"], key=lambda x: (len(x), x))
    if len(nums) != 2:
        raise ValueError(f"两端元件却有 {len(nums)} 个引脚连了网络")
    return [cp["pins"][n]["net"] for n in nums]


def _map_comp(c, cp, libpins):
    ref, val, part, F = cp["ref"], cp["value"], (cp["part"] or "").upper(), cp["fields"]
    if F.get("Sim.Enable", F.get("Spice_Netlist_Enabled", "1")).upper() in ("0", "N", "FALSE"):
        return
    dev = (F.get("Sim.Device") or F.get("Spice_Primitive") or ref.rstrip("0123456789")).upper()
    typ = (F.get("Sim.Type") or "").upper()
    params = F.get("Sim.Params") or ""
    model = F.get("Sim.Name") or F.get("Spice_Model") or ""
    if not cp["pins"]:
        return
    if dev in ("R", "C", "L"):
        n1, n2 = _two(cp)
        v = params.split("=", 1)[1] if params.lower().startswith(("r=", "c=", "l=")) else val
        x = parse_value(v.split()[0]) if v else None
        if x is None:
            raise ValueError(f"值 '{val}' 认不出")
        getattr(c, dev)(ref, n1, n2, x)
    elif dev in ("V", "I"):
        n1, n2 = _pins_src(cp, libpins)
        spec = _src_from_fields(val, typ, params)
        c.raw(f"{ref if ref.upper().startswith(dev) else dev + ref} {n1} {n2} {spec}")
    elif dev == "D":
        a, k = _pins(cp, libpins, ["A", "K"])
        m = (model or val or "DGEN").upper()
        if m not in MODELS and m not in c.models:
            c.warnings.append(f"{ref}：没有模型 '{m}'，暂用通用二极管 DGEN")
            m = "DGEN"
        c.D(ref, a, k, m)
    elif dev in ("Q", "NPN", "PNP"):
        cc, b, e = _pins(cp, libpins, ["C", "B", "E"])
        pnp = "PNP" in part or "PNP" in typ or dev == "PNP"
        m = (model or "").upper() or ("QPNP" if pnp else "QNPN")
        if m not in MODELS and m not in c.models:
            c.warnings.append(f"{ref}：没有模型 '{m}'，暂用通用 {'PNP' if pnp else 'NPN'}")
            m = "QPNP" if pnp else "QNPN"
        c.Q(ref, cc, b, e, m)
    elif dev in ("M", "NMOS", "PMOS"):
        d, g, s = _pins(cp, libpins, ["D", "G", "S"])
        if "PMOS" in part or "PMOS" in typ or dev == "PMOS":
            c.models.setdefault("PPWR", ".model PPWR VDMOS(pchan Vto=-4 Kp=10 Rd=30m Rs=10m Rg=2 Cgdmax=1.5n Cgdmin=50p Cgs=2n Cjo=800p Is=1e-11)")
            m = (model or "PPWR").upper()
        else:
            m = (model or "NPWR").upper()
        if m not in MODELS and m not in c.models:
            c.warnings.append(f"{ref}：没有模型 '{m}'，暂用通用功率管")
            m = "NPWR"
        c.M(ref, d, g, s, m)
    else:
        raise ValueError("没有 SPICE 映射（可在 KiCad 里给它加 Sim.Device/Sim.Params，或从 SPICE 网表导入）")


def _pins_src(cp, libpins):
    try:
        return _pins(cp, libpins, ["+", "-"])
    except ValueError:
        return _two(cp)


def _src_from_fields(val, typ, params):
    """KiCad 7/8 的 Sim.Type=DC/SIN/PULSE + Sim.Params='dc=5' / 'ampl=1 f=1k' …；老版本直接写在 Value 里。"""
    p = dict(x.split("=", 1) for x in params.split() if "=" in x)
    g = lambda *ks, d="0": next((p[k] for k in ks if k in p), d)
    if typ == "SIN":
        return f"SIN({g('dc')} {g('ampl')} {g('f', 'freq')} {g('td')} {g('theta')} {g('phase')})"
    if typ == "PULSE":
        return f"PULSE({g('y1')} {g('y2')} {g('td')} {g('tr', d='1n')} {g('tf', d='1n')} {g('tw', 'pw')} {g('per')})"
    if typ in ("DC", "") and "dc" in p:
        return f"DC {p['dc']}"
    v = val.strip()
    if re.match(r"(?i)^(dc|ac|sin|pulse|pwl|exp)", v):
        return v
    x = parse_value(v.split()[0]) if v else None
    if x is None:
        raise ValueError(f"源的值 '{val}' 认不出")
    return f"DC {x:g}"


def load(path):
    """按扩展名自动选择导入方式。"""
    ext = os.path.splitext(path)[1].lower()
    head = open(path, encoding="utf-8", errors="replace").read(200).lstrip()
    if ext == ".net" and head.startswith("(export"):
        return import_kicad_net(path)
    return import_spice(path)


def default_analysis(c, fallback=".tran 1u 10m"):
    """网表自带的 .tran 优先；没有就用 fallback。"""
    for a in c.analyses:
        if a.lower().startswith(".tran"):
            return a
    return fallback
