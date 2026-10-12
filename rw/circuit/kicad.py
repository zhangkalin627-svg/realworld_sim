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
        # v0.11.1：按完整关键字匹配（".op" 后必须是空白或行尾），否则 ".options"/".option" 会被当成 .op 分析而丢掉
        if re.match(r"\.(options?|opt)(\s|$)", low):
            c.extra.append(s); continue  # 选项原样传给 ngspice
        if re.match(r"\.(tran|ac|dc|op|noise|tf|pz|sens|disto|sp)(\s|$)", low):
            c.analyses.append(s); continue
        if re.match(r"\.end(\s|$)", low):
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
    _check_excitation(c)
    return c


def _check_excitation(c):
    """v0.11.1：没有任何独立源时在导入阶段就警告（否则仿真结果全零却显示“完成”）。"""
    ok, why = c.excitation()
    if ok is False:
        c.warnings.append("无激励源，结果全零：" + why + "。请加电源/信号源（KiCad：Simulation_SPICE 库的 VDC/VPULSE 等）")


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
                if len(_find(net, "node")) == 1 and (name.lower().startswith("unconnected") or "no_connect" in (_get(nd, "pintype", "") or "")):
                    continue  # v0.11.1：KiCad 的 "unconnected-(…)" 悬空脚不算连接
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
    _check_excitation(c)
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


# ---------------------------------------------------------------- v0.11.1：器件类型判断
# 优先级：Sim.Device / Spice_Primitive → Sim.Type → 库符号（libsource lib/part）→ value（型号）→ 引脚名 → 位号首字母（最后兜底）。
# 位号在不同标准里含义不同（IEC/DIN：V=半导体、D=集成电路、X=端子），所以不能先看位号。
_SIMDEV = {"R": "R", "C": "C", "L": "L", "D": "D", "NPN": "NPN", "PNP": "PNP", "Q": "Q", "NMOS": "NMOS", "PMOS": "PMOS",
           "M": "M", "V": "V", "I": "I", "SUBCKT": "X", "X": "X", "SPICE": "X"}
_PART_RE = [  # (正则 对 lib:part 大写, 类型)
    (r"(^|:)(POT|R_POT[\w\-.]*|RV[\w\-.]*|TRIM[\w\-.]*)$", "POT"),
    (r"(^|:)(FUSE[\w\-.]*|POLYFUSE[\w\-.]*|PTC[\w\-.]*|F_[\w\-.]*)$", "FUSE"),
    (r"(^|:)(R|R_[\w\-.]*|RESISTOR[\w\-.]*|R_SMALL[\w\-.]*)$", "R"),
    (r"(^|:)(C|CP|C_[\w\-.]*|CP_[\w\-.]*|CAP[\w\-.]*)$", "C"),
    (r"(^|:)(L|L_[\w\-.]*|INDUCTOR[\w\-.]*|FERRITE[\w\-.]*)$", "L"),
    (r"(^|:)(Q_NPN[\w\-.]*|NPN[\w\-.]*)$", "NPN"), (r"(^|:)(Q_PNP[\w\-.]*|PNP[\w\-.]*)$", "PNP"),
    (r"(^|:)(Q_NMOS[\w\-.]*|NMOS[\w\-.]*|Q_NIGBT[\w\-.]*)$", "NMOS"), (r"(^|:)(Q_PMOS[\w\-.]*|PMOS[\w\-.]*)$", "PMOS"),
    (r"(^|:)(D|D_[\w\-.]*|DIODE[\w\-.]*|LED[\w\-.]*|Z_[\w\-.]*|ZENER[\w\-.]*|TVS[\w\-.]*|SCHOTTKY[\w\-.]*)$", "D"),
    (r"^SIMULATION_SPICE:(V[\w\-.]*)$", "V"), (r"^SIMULATION_SPICE:(I[\w\-.]*)$", "I"),
    (r"(^|:)(VDC|VSIN|VPULSE|VPWL|VEXP|VAC|VSOURCE|BATTERY[\w\-.]*)$", "V"), (r"(^|:)(IDC|ISIN|IPULSE|IPWL|ISOURCE)$", "I"),
    (r"(^|:)(TRANSFORMER[\w\-.]*|TR_[\w\-.]*)$", "XFMR"),
    (r"^(CONNECTOR[\w\-.]*|MECHANICAL|TESTPOINT[\w\-.]*|MOUNTINGHOLE[\w\-.]*):", "PASSIVE"),
    (r"(^|:)(CONN[\w\-.]*|TERM[\w\-.]*|TP[\w\-.]*|TESTPOINT[\w\-.]*|MOUNTINGHOLE[\w\-.]*|JUMPER[\w\-.]*|SOLDERJUMPER[\w\-.]*|HEADER[\w\-.]*|PIN_HEADER[\w\-.]*)$", "PASSIVE"),
    (r"^RELAY[\w\-.]*:|(^|:)(RELAY[\w\-.]*|G5V[\w\-.]*|G6[\w\-.]*)$", "RELAY"),
    (r"^(ISOLATOR|INTERFACE[\w\-.]*|AMPLIFIER[\w\-.]*|MCU[\w\-.]*|REGULATOR[\w\-.]*|LOGIC[\w\-.]*|74XX|4XXX|ANALOG[\w\-.]*|DRIVER[\w\-.]*|POWER_MANAGEMENT|SENSOR[\w\-.]*|TIMER[\w\-.]*|OPTOCOUPLER[\w\-.]*|MEMORY[\w\-.]*):", "IC"),
    (r"^DIODE:", "D"), (r"^TRANSISTOR_BJT:", "BJT"), (r"^TRANSISTOR_FET:", "FET"),
]
_VAL_RE = [  # 按型号（value 字段开头）
    (r"^(1N\d|BAT\d|BAS\d|BAV\d|BZX|BZT|MMSZ|SMBJ|SMAJ|SMCJ|P6KE|1\.5KE|PMEG|MBR|SS\d|BYV|BYW|UF\d|ES\d|US\d|RB\d|STPS|SB\d|1SS|LL4|FR\d|HER\d|SMD?Z|ZMM|SM\d{4})", "D"),
    (r"^(BC\d|2N[23456]\d|MMBT|ZXT[NP]?\d|BD\d|TIP\d|PZT|MJE|MJD|2SC|2SA|2SD|2SB|BCP|BCX|FMMT|DTC|DTA)", "BJT"),
    (r"^(IRF|IRL|BSS\d|BSP\d|BSC\d|SI\d{4}|AO\d|IRLML|2N7002|STP\d|STD\d|IPP|IPD|IPB|FDN|FDS|NTD|NTR|DMN|DMP|AON|IXF)", "FET"),
    (r"^(ADUM|AMC\d|MC3\d|LM\d|TL0|TL4|NE5|HCNR|CPC\d|ISO\d|UCC|IR2\d|TLP|PC8\d|ACPL|HCPL|SI8\d|74|CD4|ATMEGA|ATTINY|STM32|OPA\d|AD\d|MAX\d|LT\d|TPS\d|UC3|UC2|SN\d|ULN|L78|L79|78\d|79\d|NCP|ACS\d|INA\d|XTR|HCPL)", "IC"),
    (r"^(CONN|HEADER|IDC|TERMINAL|PIN_?HEADER)", "PASSIVE"),
]
_FALLBACK_NOTE = "（按位号首字母兜底判断；建议在 KiCad 里给它加 Sim.Device 字段）"


def _pin_names(cp, libpins):
    out = {}
    for num, d in cp["pins"].items():
        out[num] = (d.get("name") or libpins.get(num) or "").upper()
    simpins = cp["fields"].get("Sim.Pins") or cp["fields"].get("Sim_Pins")
    if simpins:
        mp = dict(x.split("=", 1) for x in simpins.split() if "=" in x)
        out = {num: mp.get(num, out.get(num, "")).upper() for num in cp["pins"]}
    return out


def _by_pins(names):
    st = {re.sub(r"[\d_()]|TAB", "", n) for n in names.values() if n}
    st.discard("")
    if st and st <= {"A", "K"} and {"A", "K"} <= st:
        return "D"
    if {"C", "B", "E"} <= st and st <= {"C", "B", "E", "S", "SUB"}:
        return "BJT"
    if {"D", "G", "S"} <= st and st <= {"D", "G", "S", "B", "BULK"}:
        return "FET"
    return None


def classify(cp, libpins=None):
    """返回 (类型, 依据)。类型：R C L POT FUSE D NPN PNP BJT NMOS PMOS FET V I X XFMR PASSIVE RELAY IC 或 None（判断不出）。"""
    libpins = libpins or {}
    F, ref, val = cp["fields"], cp["ref"], (cp["value"] or "").strip()
    dev = (F.get("Sim.Device") or F.get("Spice_Primitive") or "").strip().upper()
    typ = (F.get("Sim.Type") or "").strip().upper()
    if dev:
        k = _SIMDEV.get(dev)
        if k == "Q":
            k = "PNP" if "PNP" in typ else ("NPN" if "NPN" in typ else "BJT")
        if k == "M":
            k = "PMOS" if "PMOS" in typ or "PCHAN" in typ else ("NMOS" if "NMOS" in typ or "NCHAN" in typ else "FET")
        if k:
            return k, f"Sim.Device={dev}"
    for t, k in (("NPN", "NPN"), ("PNP", "PNP"), ("NMOS", "NMOS"), ("PMOS", "PMOS")):
        if typ == t:
            return k, f"Sim.Type={typ}"
    libpart = f"{cp['lib'] or ''}:{cp['part'] or ''}".upper()
    for rx, k in _PART_RE:
        if re.search(rx, libpart):
            if k in ("BJT", "FET"):  # 极性看 part 名
                pol = "PNP" if "PNP" in libpart else "NPN" if "NPN" in libpart else "PMOS" if ("PMOS" in libpart or "P-CH" in libpart) else "NMOS" if "NMOS" in libpart else None
                k = pol or k
            return k, f"库符号 {cp['lib']}:{cp['part']}"
    v = val.upper().replace(" ", "")
    for rx, k in _VAL_RE:
        if re.search(rx, v):
            return k, f"型号 {val.split()[0]}"
    k = _by_pins(_pin_names(cp, libpins))
    if k:
        return k, "引脚名"
    letters = re.match(r"[A-Za-z]+", ref)
    L = letters.group(0).upper() if letters else ""
    n = len(cp["pins"])
    if L in ("R", "C", "L") and n == 2:
        return L, "位号" + _FALLBACK_NOTE
    if L in ("RV", "VR", "POT") and n == 3:
        return "POT", "位号" + _FALLBACK_NOTE
    if L == "Q" and n >= 3:
        return "BJT", "位号" + _FALLBACK_NOTE
    if L == "D" and n == 2:
        return "D", "位号" + _FALLBACK_NOTE
    if L in ("V", "I") and n == 2 and re.match(r"(?i)^(dc|ac|sin|pulse|pwl|exp|[-+]?\d)", val or ""):
        return L, "位号+值" + _FALLBACK_NOTE
    if L in ("J", "P", "X", "TP", "CN", "CON") :
        return "PASSIVE", "位号" + _FALLBACK_NOTE
    return None, "位号/库符号/型号/引脚名都判断不出"


def _pins_by(cp, libpins, order, ref):
    """多脚器件：按引脚名找每个功能脚；同名脚接在不同网络时给警告（只用第一个）。"""
    names = _pin_names(cp, libpins)
    out, warn = [], []
    for o in order:
        base = lambda nm: re.sub(r"\d+$|\(TAB\)|_TAB", "", nm)
        hit = [num for num, nm in names.items() if nm == o or base(nm) == o]
        if not hit:
            raise ValueError(f"找不到引脚 {o}（引脚名：{names}）")
        nets = {cp["pins"][h]["net"] for h in hit}
        if len(nets) > 1:
            warn.append(f"{ref}：{len(hit)} 个 {o} 脚接在不同网络 {sorted(nets)}，只用了第一个")
        out.append(cp["pins"][sorted(hit, key=lambda x: (len(x), x))[0]]["net"])
    used = {num for num, nm in names.items() if any(nm == o or re.sub(r"\d+$|\(TAB\)|_TAB", "", nm) == o for o in order)}
    return out, warn, [n for n in cp["pins"] if n not in used]


def _map_comp(c, cp, libpins):
    ref, val, part, F = cp["ref"], cp["value"], (cp["part"] or "").upper(), cp["fields"]
    if F.get("Sim.Enable", F.get("Spice_Netlist_Enabled", "1")).upper() in ("0", "N", "FALSE"):
        return
    if not cp["pins"]:
        return
    typ = (F.get("Sim.Type") or "").upper()
    params = F.get("Sim.Params") or ""
    model = F.get("Sim.Name") or F.get("Spice_Model") or ""
    kind, why = classify(cp, libpins)
    c.kinds = getattr(c, "kinds", {}); c.kinds[ref] = (kind, why)
    if why.startswith("位号") and kind not in ("PASSIVE", None):
        c.warnings.append(f"{ref}（{val}）：判为 {kind}{_FALLBACK_NOTE}")
    if kind is None:
        raise ValueError(f"判断不出器件类型（{why}），已跳过；请加 Sim.Device 字段")
    if kind in ("PASSIVE",):
        raise ValueError("连接器/端子/测试点，没有电气模型，跳过")
    if kind in ("IC", "XFMR", "RELAY"):
        nm = {"IC": "集成电路", "XFMR": "变压器", "RELAY": "继电器"}[kind]
        raise ValueError(f"{nm}（{len(cp['pins'])} 脚，判断依据：{why}）没有 SPICE 模型，已跳过；可在 KiCad 里给它 Sim.Device=SUBCKT + Sim.Library")
    if kind in ("R", "C", "L"):
        n1, n2 = _two(cp)
        v = params.split("=", 1)[1] if params.lower().startswith(("r=", "c=", "l=")) else val
        x = parse_value(v.split()[0]) if v else None
        if x is None:
            raise ValueError(f"值 '{val}' 认不出")
        getattr(c, kind)(ref, n1, n2, x)
    elif kind == "POT":  # 3 脚电位器：按滑动端在中间（50 %）拆成两只电阻
        names = _pin_names(cp, libpins)
        w = [n for n, nm in names.items() if nm in ("W", "2", "WIPER")] or ["2"]
        ends = sorted([n for n in cp["pins"] if n not in w], key=lambda x: (len(x), x))
        x = parse_value((val or "").split()[0]) if val else None
        if x is None or w[0] not in cp["pins"] or not ends:
            raise ValueError(f"电位器 '{val}' 认不出阻值或引脚")
        if len(ends) == 1:  # 一端悬空 = 可变电阻
            c.R(ref, cp["pins"][ends[0]]["net"], cp["pins"][w[0]]["net"], x / 2)
            c.warnings.append(f"{ref}：电位器一端悬空，按可变电阻、滑动端 50 % 处理（{x / 2:g} Ω）")
            return
        c.R(f"{ref}_a", cp["pins"][ends[0]]["net"], cp["pins"][w[0]]["net"], x / 2)
        c.R(f"{ref}_b", cp["pins"][w[0]]["net"], cp["pins"][ends[1]]["net"], x / 2)
        c.warnings.append(f"{ref}：电位器按滑动端在 50 % 处理（{x / 2:g} Ω + {x / 2:g} Ω）")
    elif kind == "FUSE":
        n1, n2 = _two(cp)
        c.R(ref, n1, n2, 0.01)
        c.warnings.append(f"{ref}（{val}）：保险丝/PTC 按 10 mΩ 导线近似，没有动作（熔断/跳闸）模型")
    elif kind in ("V", "I"):
        n1, n2 = _pins_src(cp, libpins)
        spec = _src_from_fields(val, typ, params)
        c.raw(f"{ref if ref.upper().startswith(kind) else kind + ref} {n1} {n2} {spec}")
    elif kind == "D":
        names = _pin_names(cp, libpins)
        an = sorted([n for n, nm in names.items() if re.sub(r"\d+$", "", nm) == "A"], key=lambda x: (len(x), x))
        kn = sorted([n for n, nm in names.items() if re.sub(r"\d+$", "", nm) == "K"], key=lambda x: (len(x), x))
        if not an or not kn:
            if len(cp["pins"]) == 2:  # 没有引脚名：KiCad Device:D 约定 1=K 2=A
                kn, an = ["1"], ["2"]
                c.warnings.append(f"{ref}：二极管引脚没有名字，按 KiCad 约定 1=K 2=A")
            else:
                raise ValueError(f"二极管有 {len(cp['pins'])} 个脚但认不出 A/K（引脚名：{names}）")
        m = (model or (val.split()[0] if val else "") or "DGEN").upper()
        if m not in MODELS and m not in c.models:
            c.warnings.append(f"{ref}：没有模型 '{m}'，暂用通用二极管 DGEN")
            m = "DGEN"
        pairs = [(a, k) for a in an for k in kn] if (len(an) == 1 or len(kn) == 1) else list(zip(an, kn))
        if len(pairs) > 1:
            c.warnings.append(f"{ref}：多管封装，拆成 {len(pairs)} 只二极管")
        for i, (a, k) in enumerate(pairs):
            c.D(ref if len(pairs) == 1 else f"{ref}_{i + 1}", cp["pins"][a]["net"], cp["pins"][k]["net"], m)
    elif kind in ("NPN", "PNP", "BJT"):
        (cc, b, e), w, extra = _pins_by(cp, libpins, ["C", "B", "E"], ref)
        c.warnings += w
        if extra:
            c.warnings.append(f"{ref}：三极管多出的引脚 {extra} 未接入模型")
        pnp = kind == "PNP" or "PNP" in part or "PNP" in typ
        if kind == "BJT":
            c.warnings.append(f"{ref}：认出是三极管但看不出极性，按 NPN 处理（请加 Sim.Device=NPN/PNP）")
        m = (model or "").upper() or ("QPNP" if pnp else "QNPN")
        if m not in MODELS and m not in c.models:
            c.warnings.append(f"{ref}：没有模型 '{m}'，暂用通用 {'PNP' if pnp else 'NPN'}")
            m = "QPNP" if pnp else "QNPN"
        c.Q(ref, cc, b, e, m)
    elif kind in ("NMOS", "PMOS", "FET"):
        (d, g, s_), w, extra = _pins_by(cp, libpins, ["D", "G", "S"], ref)
        c.warnings += w
        names = _pin_names(cp, libpins)
        body = [n for n in extra if names.get(n) in ("B", "BULK")]
        if body and cp["pins"][body[0]]["net"] != s_:
            c.warnings.append(f"{ref}：衬底脚 B 没接源极，功率管模型(VDMOS)只有 3 端，衬底被忽略")
        if kind == "FET":
            c.warnings.append(f"{ref}：认出是 MOSFET 但看不出沟道，按 N 沟道处理（请加 Sim.Device=NMOS/PMOS）")
        if kind == "PMOS" or "PMOS" in part or "PMOS" in typ:
            c.models.setdefault("PPWR", ".model PPWR VDMOS(pchan Vto=-4 Kp=10 Rd=30m Rs=10m Rg=2 Cgdmax=1.5n Cgdmin=50p Cgs=2n Cjo=800p Is=1e-11)")
            m = (model or "PPWR").upper()
        else:
            m = (model or "NPWR").upper()
        if m not in MODELS and m not in c.models:
            c.warnings.append(f"{ref}：没有模型 '{m}'，暂用通用功率管")
            m = "PPWR" if m != "NPWR" and "PPWR" in c.models and kind == "PMOS" else "NPWR"
        c.M(ref, d, g, s_, m)
    elif kind == "X":  # 子电路：Sim.Library + Sim.Name，引脚顺序按 Sim.Pins（num=子电路脚名）或 .subckt 行
        lib = F.get("Sim.Library") or ""
        name = model or val
        base = os.path.dirname(os.path.abspath(c.source_file)) if getattr(c, "source_file", None) else "."
        lp = lib if (not lib or os.path.isabs(lib)) else os.path.join(base, lib)
        if not lib or not os.path.exists(lp):
            raise ValueError(f"子电路 {name}（{len(cp['pins'])} 脚）找不到模型库 '{lib}'，已跳过")
        txt = open(lp, encoding="utf-8", errors="replace").read()
        mm = re.search(r"(?im)^\s*\.subckt\s+" + re.escape(name) + r"\s+(.+)$", txt)
        if not mm:
            raise ValueError(f"模型库 {lib} 里没有 .subckt {name}")
        spins = [x for x in mm.group(1).split() if "=" not in x and x.lower() != "params:"]
        mp = dict(x.split("=", 1) for x in (F.get("Sim.Pins") or "").split() if "=" in x)
        nodes = []
        nums = sorted(cp["pins"], key=lambda x: (len(x), int(x) if x.isdigit() else 0, x))
        for k, sp in enumerate(spins):
            num = next((n for n, v in mp.items() if v.lower() == sp.lower()), None) if mp else (nums[k] if k < len(nums) else None)
            if num is None or num not in cp["pins"]:
                c.warnings.append(f"{ref}：子电路脚 {sp} 没有连接，接到悬空节点")
                nodes.append(f"nc_{ref}_{sp}")
            else:
                nodes.append(cp["pins"][num]["net"])
        if lp not in [x.strip('"') for x in c.includes]:
            c.includes.append(f'"{lp}"' if " " in lp else lp)
        c.X(ref, nodes, name)
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
