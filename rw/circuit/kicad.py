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
from . import devlib as DL

GROUND = {"0", "gnd", "/gnd", "gndref", "gnda", "gndd", "gndpwr", "earth", "/0"}


def _needs_ltpsa(subckt, path):
    """供应商模型是否需要 ngspice 的 LTspice 兼容模式 (set ngbehavior=ltpsa)。
    判据：环境变量 RW_LTPSA_PARTS（逗号分隔的型号前缀）命中，或模型文件里写明 LTspice。"""
    pref = [p.strip().upper() for p in os.environ.get("RW_LTPSA_PARTS", "").split(",") if p.strip()]
    if any(str(subckt).upper().startswith(p) for p in pref):
        return True
    try:
        return "LTSPICE" in open(path, encoding="utf-8", errors="ignore").read(65536).upper()
    except OSError:
        return False


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
            # 行为源（E/G 的 value={...}/table 形式）表达式不是节点名：含 {} 或 = 的字段原样保留，
            # 否则 _node 会把表达式里的 (){}= 换成下划线，网表被毁（接地卡 Gea value={200u*(V(ref)-V(fb))} 案例）
            f[1:1 + nn] = [_node(x) if not re.search(r"[{}=]", x) else x for x in f[1:1 + nn]]
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
    c.ratings, c.assumed, c.vendor_models, c.notes = {}, {}, [], {}  # v0.12：额定值（应力校核用）、假设值、引用的供应商模型
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
            nm, pv = _get(pr, "name"), _get(pr, "value", "")
            if nm is None and len(pr) >= 3 and isinstance(pr[1], str) and isinstance(pr[2], str):
                nm, pv = pr[1], pr[2]  # KiCad 位置式 (property "Value" "1k")（issue 1945：原写入 fields[None] 致下游崩溃）
            if nm:
                fields[nm] = pv
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
    c.lib_subckts = {"IGBT_GEN", "IR2181X_BEH", "MIC442X_BEH", "LDO_BEH", "FUSE_I2T"}
    _check_excitation(c)
    if c.excitation()[0] is False:
        c.warnings.append("提示：电源板可用 rw.circuit.devlib.add_grid(c, ('L1','L2','L3'), preset='nom'|'hi10'|'hi20', surge=…) 加三相电网/浪涌激励")
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
    (r"(^|:)(Q_NIGBT[\w\-.]*|Q_PIGBT[\w\-.]*|IGBT[\w\-.]*)$", "IGBT"),  # v0.12：IGBT 单独一类（C/G/E）
    (r"(^|:)(NETTIE[\w\-.]*|NET_TIE[\w\-.]*)$", "NETTIE"),  # v0.12：NetTie = 0 Ω 短接
    (r"(^|:)(Q_NMOS[\w\-.]*|NMOS[\w\-.]*)$", "NMOS"), (r"(^|:)(Q_PMOS[\w\-.]*|PMOS[\w\-.]*)$", "PMOS"),
    (r"(^|:)(D|D_[\w\-.]*|DIODE[\w\-.]*|LED[\w\-.]*|Z_[\w\-.]*|ZENER[\w\-.]*|TVS[\w\-.]*|SCHOTTKY[\w\-.]*)$", "D"),
    (r"^SIMULATION_SPICE:(V[\w\-.]*)$", "V"), (r"^SIMULATION_SPICE:(I[\w\-.]*)$", "I"),
    (r"(^|:)(VDC|VSIN|VPULSE|VPWL|VEXP|VAC|VSOURCE|BATTERY[\w\-.]*)$", "V"), (r"(^|:)(IDC|ISIN|IPULSE|IPWL|ISOURCE)$", "I"),
    (r"(^|:)(TRANSFORMER[\w\-.]*|TR_[\w\-.]*)$", "XFMR"),
    (r"^(CONNECTOR[\w\-.]*|MECHANICAL|TESTPOINT[\w\-.]*|MOUNTINGHOLE[\w\-.]*):", "PASSIVE"),
    (r"(^|:)(CONN[\w\-.]*|TERM[\w\-.]*|TP[\w\-.]*|TESTPOINT[\w\-.]*|MOUNTINGHOLE[\w\-.]*|JUMPER[\w\-.]*|SOLDERJUMPER[\w\-.]*|HEADER[\w\-.]*|PIN_HEADER[\w\-.]*)$", "PASSIVE"),
    (r"^RELAY[\w\-.]*:|(^|:)(RELAY[\w\-.]*|G5V[\w\-.]*|G6[\w\-.]*)$", "RELAY"),
    (r"^(ISOLATOR|INTERFACE[\w\-.]*|AMPLIFIER[\w\-.]*|MCU[\w\-.]*|REGULATOR[\w\-.]*|LOGIC[\w\-.]*|74XX|4XXX|ANALOG[\w\-.]*|DRIVER[\w\-.]*|POWER_MANAGEMENT|SENSOR[\w\-.]*|TIMER[\w\-.]*|OPTOCOUPLER[\w\-.]*|MEMORY[\w\-.]*):", "IC"),
    (r"(^|:)(ADM\d[\w\-.]*|LH1\d[\w\-.]*|SOIC\d*[\w\-.]*|SOT-?\d+[\w\-.]*|TSSOP[\w\-.]*|MSOP[\w\-.]*|QFN[\w\-.]*|DIP-?\d+[\w\-.]*)$", "IC"),  # v0.12：自建库的 IC 符号
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


def _comp_value(cp):
    """元件值统一提取（issue 1945）：comp 的 value 优先；为空时回退 fields/property 里的
    Value 字段（大小写不敏感）。冲突时以 comp value 为准。"""
    v = (cp["value"] or "").strip()
    if v:
        return v
    for k, fv in cp["fields"].items():
        if k and k.lower() == "value" and isinstance(fv, str) and fv.strip():
            return fv.strip()
    return ""


def classify(cp, libpins=None):
    """返回 (类型, 依据)。类型：R C L POT FUSE D NPN PNP BJT NMOS PMOS FET IGBT NETTIE V I X XFMR PASSIVE RELAY IC 或 None（判断不出）。"""
    libpins = libpins or {}
    F, ref, val = cp["fields"], cp["ref"], _comp_value(cp)
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
    ref, val, part, F = cp["ref"], _comp_value(cp), (cp["part"] or "").upper(), cp["fields"]
    if F.get("Sim.Enable", F.get("Spice_Netlist_Enabled", "1")).upper() in ("0", "N", "FALSE"):
        return
    if "dnp" in {k.lower() for k in F} or re.search(r"\bDNP\b", val or ""):  # v0.12：不贴装（DNP）元件不进仿真
        c.notes = getattr(c, "notes", {}); c.notes[ref] = "DNP（不贴装），未进仿真"
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
    rt = DL.ratings_from_text(val)
    if rt:
        c.ratings[ref] = dict(rt, src="值字段")
    if kind == "IC":
        if _map_ic(c, cp, libpins, why):
            return
        nm = "单片机" if re.search(r"(?i)ATMEGA|ATTINY|STM32|PIC\d|MCU", f"{part} {val}") else "集成电路"
        hint = "；单片机可用 rw.mcu（v0.7 AVR 协同仿真）在电路外联合仿真" if nm == "单片机" else "；可在 KiCad 里给它 Sim.Device=SUBCKT + Sim.Library"
        raise ValueError(f"{nm}（{len(cp['pins'])} 脚，判断依据：{why}）没有 SPICE 模型，已跳过{hint}")
    if kind == "XFMR":
        _map_xfmr(c, cp, libpins)
        return
    if kind == "RELAY":
        raise ValueError(f"继电器（{len(cp['pins'])} 脚，判断依据：{why}）没有 SPICE 模型，已跳过；可在 KiCad 里给它 Sim.Device=SUBCKT + Sim.Library")
    if kind == "NETTIE":  # v0.12：NetTie_2 = 0 Ω（ngspice 不接受 R=0，用 1 µΩ）
        nets = sorted({d["net"] for d in cp["pins"].values()})
        for i, n2 in enumerate(nets[1:]):
            c.R(ref if i == 0 else f"{ref}_{i + 1}", nets[0], n2, 1e-6)
        c.notes[ref] = "NetTie → 0 Ω 短接"
        return
    if kind == "IGBT":
        _map_igbt(c, cp, libpins, val, model)
        return
    if kind in ("R", "C", "L"):
        n1, n2 = _two(cp)
        v = params.split("=", 1)[1] if params.lower().startswith(("r=", "c=", "l=")) else val
        x = DL.value_for(kind, v) if v else None  # v0.12：值字段带说明文字（'MKP21 0.33uF 400V'）时抽第一个带单位的数
        if x is None:
            pol = "POLARIZED" in part or part.startswith("CP")
            x = {"R": None, "C": 100e-6 if pol else 100e-9, "L": 10e-6}[kind]
            if x is None:
                raise ValueError(f"值 '{val}' 认不出")
            c.assumed[ref] = x
            c.warnings.append(f"{ref}（{val}）：值没读出，按 {_eng(x)}{dict(C='F', L='H')[kind]} 假设（{'电解' if pol else '通用'}默认值，结果仅供参考）")
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
    elif kind == "FUSE":  # v0.12：I²t 熔断模型（PTC 仍按导线）
        n1, n2 = _two(cp)
        if re.search(r"PTC|POLY", f"{part} {val}".upper()):
            c.R(ref, n1, n2, 0.01)
            c.warnings.append(f"{ref}（{val}）：PTC/自恢复保险丝按 10 mΩ 导线近似，没有跳闸模型")
            return
        In, i2t, note = DL.fuse_params(val, F)
        c.subckts["FUSE_I2T"] = DL.FUSE_I2T
        c.X(ref, [n1, n2], "FUSE_I2T", i2t="%.6g" % i2t, r0="10m")
        c.ratings.setdefault(ref, {})["A"] = In
        c.notes[ref] = f"保险丝 I²t 模型：In={In:g} A，I²t={i2t:.3g} A²s" + (f"（{note}）" if note else "")
        if note:
            c.warnings.append(f"{ref}（{val}）：{note}")
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
        vend = None
        if m not in MODELS and m not in c.models:
            # v0.12：功率二极管库 + 供应商模型自动匹配（说明文字整段 + Sim.Name + 库符号名一起匹配）
            how, nm, det = DL.match_diode(" ".join([model or "", val or "", F.get("MPN", ""), F.get("Part", ""), cp["part"] or ""]))
            if how == "vendor":
                vend = (nm, det)
                f_, typ_, _p = det
                if f_ not in [x.strip('"') for x in c.includes]:
                    c.includes.append(f'"{f_}"' if " " in f_ else f_)
                c.vendor_models.append((ref, nm, f_))
                c.notes[ref] = f"供应商模型 {nm}（{os.path.basename(f_)}，仅内部评估，不随发布）"
                m = nm
            elif how == "lib":
                c.models[nm.upper()] = DL.diode_model_line(nm)
                rr = DL.DIODES[nm][1]
                for k_ in ("V", "A"):
                    c.ratings.setdefault(ref, {"src": "器件库"}).setdefault(k_, rr[k_])
                c.notes[ref] = f"器件库 {nm}：{DL.DIODES[nm][2]}"
                m = nm
            else:
                c.warnings.append(f"{ref}：没有模型 '{m}'，暂用通用二极管 DGEN")
                m = "DGEN"
        pairs = [(a, k) for a in an for k in kn] if (len(an) == 1 or len(kn) == 1) else list(zip(an, kn))
        if len(pairs) > 1:
            c.warnings.append(f"{ref}：多管封装，拆成 {len(pairs)} 只二极管")
        for i, (a, k) in enumerate(pairs):
            nm_i = ref if len(pairs) == 1 else f"{ref}_{i + 1}"
            na, nk = cp["pins"][a]["net"], cp["pins"][k]["net"]
            if vend and vend[1][1] == "subckt":
                order = _vendor_diode_order(vend[1][2], na, nk, tag=nm_i)
                if order is None:
                    order = [na, nk]
                    c.warnings.append(f"{ref}：供应商子电路 {vend[0]} 引脚 {vend[1][2]} 认不出 A/K，按 (A, K) 顺序接")
                c.X(nm_i, order, vend[0])
            else:
                c.D(nm_i, na, nk, m)
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
        if not model:  # v0.12：供应商 MOSFET/SiC 共源共栅子电路自动匹配（引脚 nd ng ns）
            for tok in re.findall(r"[A-Z0-9][A-Z0-9\-]{6,}", (val or "").upper()):
                hit = DL.find_vendor(tok)
                if hit and hit[1][1] == "subckt" and len(hit[1][2]) == 3:
                    mp = {"D": d, "G": g, "S": s_}
                    order = [mp.get(re.sub(r"^N(?=[DGS]$)", "", p.upper())[:1]) for p in hit[1][2]]
                    if None in order or len(set(p.upper()[-1] for p in hit[1][2])) != 3:
                        continue
                    f_ = hit[1][0]
                    if f_ not in [x.strip('"') for x in c.includes]:
                        c.includes.append(f'"{f_}"' if " " in f_ else f_)
                    c.vendor_models.append((ref, hit[0], f_))
                    c.notes[ref] = f"供应商模型 {hit[0]}（{os.path.basename(f_)}，仅内部评估，不随发布）"
                    c.X(ref, order, hit[0])
                    c.stress_nodes = getattr(c, "stress_nodes", {}); c.stress_nodes[ref] = (d, s_)
                    rt = DL.ratings_from_text(val)
                    if rt.get("V"):
                        c.ratings[ref] = dict(V=rt["V"], src="值字段")
                    if _needs_ltpsa(hit[0], f_):
                        c.extra_opts = getattr(c, "extra_opts", set()); c.extra_opts.add("ngbehavior=ltpsa")
                    return
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


def _eng(x):
    """工程计数法：1.5e-4 → '150µ'。"""
    if x == 0:
        return "0"
    for p, f in (("G", 1e9), ("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("µ", 1e-6), ("n", 1e-9), ("p", 1e-12)):
        if abs(x) >= f * 0.9999:
            return "%g%s" % (round(x / f, 4), p)
    return "%g" % x


def _vendor_diode_order(pins, na, nk, tag="d"):
    """供应商二极管子电路的引脚顺序 → [网络...]。认 A/ANODE/K/C/CATHODE/KATHODE；
    Cree 的 'Case'（内部阴极/管壳节点）、'NC'、'Tj' 等接各自独立的内部节点。认不出返回 None。"""
    out, seen = [], set()
    for p in pins:
        u = p.upper()
        if u in ("A", "AN", "ANODE", "A1", "P"):
            out.append(na); seen.add("A")
        elif u in ("K", "C", "CATHODE", "KATHODE", "K1", "N"):
            out.append(nk); seen.add("K")
        elif u in ("CASE", "TAB", "NC", "TJ", "TC", "TCASE", "TEMP"):
            out.append(f"nc_{tag}_{u.lower()}")
        else:
            return None
    return out if seen == {"A", "K"} else None


def _norm_pin(nm):
    return re.sub(r"[~{}\s]", "", nm or "").upper()


def _map_igbt(c, cp, libpins, val, model):
    """v0.12：IGBT（C/G/E）。没有 Sim.Library 时用通用近似 IGBT_GEN 子电路，额定值按型号表。"""
    ref = cp["ref"]
    names = _pin_names(cp, libpins)
    if "G" not in {re.sub(r"\d+$", "", n) for n in names.values()} and {"C", "E"} <= set(names.values()):
        # 网表里栅极没接任何网络（原理图漏线）：栅极接独立节点经 10 kΩ 到 E（= 关断），并警告
        (cc, e), w, extra = _pins_by(cp, libpins, ["C", "E"], ref)
        g = f"nc_{ref}_g"
        c.R(f"{ref}_gfloat", g, e, 1e4)
        c.warnings.append(f"{ref}（{val}）：栅极 G 在网表里没有连接（原理图可能漏线），按关断处理（G 经 10 kΩ 接 E）")
    else:
        (cc, g, e), w, extra = _pins_by(cp, libpins, ["C", "G", "E"], ref)
    c.warnings += w
    lib = cp["fields"].get("Sim.Library")
    if model and lib:
        c.X(ref, [cc, g, e], model)
        if lib not in c.includes:
            c.includes.append(lib)
    else:
        c.subckts["IGBT_GEN"] = DL.IGBT_GEN
        c.X(ref, [cc, g, e], "IGBT_GEN")
        c.notes[ref] = "通用 IGBT 近似（IGBT_GEN，非厂家模型）"
    up = (val or "").upper().replace(" ", "")
    for k, r in DL.IGBT_RATINGS.items():
        if k in up and r:
            c.ratings[ref] = dict(r, src="型号表")
            break
    c.stress_nodes = getattr(c, "stress_nodes", {})
    c.stress_nodes[ref] = (cc, e)


def _map_ic(c, cp, libpins, why):
    """v0.12：常用驱动/LDO 行为级子电路。认出返回 True；否则 False（调用者给"无模型跳过"警告）。"""
    ref, val = cp["ref"], _comp_value(cp)
    key_s = f"{cp['part'] or ''} {val}".upper()
    for key, d in DL.IC_BEH.items():
        if not re.search(d["match"], key_s):
            continue
        names = {num: _norm_pin(nm) for num, nm in _pin_names(cp, libpins).items()}
        nodes, missing = [], []
        for sp in d["pins"]:
            al = d["alias"].get(sp, [sp])
            hit = sorted([n for n, nm in names.items() if nm in al], key=lambda x: (len(x), x))
            if hit:
                nodes.append(cp["pins"][hit[0]]["net"])
            else:
                missing.append(sp); nodes.append(None)
        if key == "LDO" and nodes[2] is None:  # 没有使能脚：接 VIN
            nodes[2] = nodes[0]; missing.remove("EN")
        if any(n is None for n in nodes[:2]) or len(missing) > 2:
            c.warnings.append(f"{ref}（{val}）：{d['note']} 但引脚对不上（缺 {missing}，引脚名 {sorted(set(names.values()))}），已跳过")
            return True
        for i, sp in enumerate(d["pins"]):
            if nodes[i] is None:
                nodes[i] = f"nc_{ref}_{re.sub(r'[^A-Za-z0-9]', '', sp)}"
                c.warnings.append(f"{ref}：{sp} 脚没找到，接悬空节点")
        c.subckts[d["sub"]] = d["text"]
        c.X(ref, nodes, d["sub"], **DL.ic_params(key, cp["part"] or "", val))
        c.notes[ref] = d["note"] + "（行为级，MIT，典型值）"
        c.stress_nodes = getattr(c, "stress_nodes", {})
        if key == "IR2181X":
            c.ratings[ref] = dict(V=d["ratings"]["VS"], src="VS 对 COM 耐压")
            c.stress_nodes[ref] = (nodes[7], nodes[3])
        elif key == "MIC442X":
            c.ratings[ref] = dict(V=d["ratings"]["V+"], src="电源电压")
            c.stress_nodes[ref] = (nodes[0], nodes[1])
        elif key == "LDO":
            c.ratings[ref] = dict(V=d["ratings"]["VIN"], src="输入电压")
            c.stress_nodes[ref] = (nodes[0], nodes[1])
        return True
    return False


def _map_xfmr(c, cp, libpins):
    """v0.12：变压器 → 耦合电感。引脚按编号两两成绕组（1-2 原边，3-4、5-6 … 副边），每对第一个脚当同名端（假设）。
    参数来自 Sim.Params（lp= n= n2= lleak= k=）或 Lp/N/Ratio/Lleak/K 字段；缺省值记入 c.assumed 并警告。"""
    ref, F = cp["ref"], cp["fields"]
    nums = sorted(cp["pins"], key=lambda x: (len(x), int(x) if x.isdigit() else 0, x))
    if len(nums) < 4 or len(nums) % 2:
        raise ValueError(f"变压器有 {len(nums)} 个已连接引脚，无法两两配成绕组")
    p, note = DL.xfmr_params(_comp_value(cp), F)  # v0.30.1：漏改统一取值（Pally 审查发现，property Value 指定的变压器曾静默回退 1:1）
    nw = len(nums) // 2
    ratios = (p["ratios"] + [p["ratios"][-1]] * nw)[: nw - 1]
    if len(p["ratios"]) < nw - 1:
        note.append(f"只给了 {len(p['ratios'])} 个匝比，其余副边按同一匝比")
    net = lambda i: cp["pins"][nums[i]]["net"]
    mid = f"{ref}_lk"
    c.L(f"{ref}_lk", net(0), mid, p["lleak"])
    Ls = [c.L(f"{ref}_p", mid, net(1), p["lp"]).name]
    for i, n in enumerate(ratios):
        Ls.append(c.L(f"{ref}_s{i + 1}", net(2 + 2 * i), net(3 + 2 * i), p["lp"] / (n * n)).name)
        c.R(f"{ref}_s{i + 1}_dc", net(2 + 2 * i), "0", 1e9)  # 浮地副边的直流通路（1 GΩ，不影响结果）
    for i in range(len(Ls)):
        for j in range(i + 1, len(Ls)):
            c.K(f"{ref}_{i}{j}", Ls[i], Ls[j], min(p["k"], 0.999999))
    note.append("同名端按每个绕组的第一个引脚（编号小的）假设")
    c.assumed[ref] = p
    c.notes[ref] = "耦合电感变压器：Lp=%s H，匝比 %s，漏感 %s H，k=%g" % (_eng(p["lp"]), "/".join("%g" % r for r in ratios), _eng(p["lleak"]), p["k"])
    for n_ in note:
        c.warnings.append(f"{ref}：{n_}")


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
