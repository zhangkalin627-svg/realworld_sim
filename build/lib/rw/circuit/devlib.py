"""v0.12 通用器件库（MIT，公开）：值字段解析、功率二极管库 + 供应商模型自动匹配、IGBT、
驱动/LDO 行为级子电路、耦合变压器、保险丝 I²t、电网激励预设与 IEC 61000-4-5 浪涌。

如实说明：
- 这里所有 .model / .subckt 参数都是按公开数据手册的典型值**粗拟合的通用近似**（标注 "_GEN"/"BEH"），
  不是厂家模型，开关细节（反向恢复、结电容非线性、温度）只是量级正确。
- 供应商模型（如 Cree/Wolfspeed、onsemi）只在设置了环境变量 RW_SPICE_MODELS（模型目录）时自动引用，
  **不复制、不随仓库发布**（许可多为"仅内部评估"）；未设置则只用内置通用模型。
"""
import glob, math, os, re

# ======================================================================= 1) 值字段解析
_PFX = {"T": 1e12, "G": 1e9, "MEG": 1e6, "M": 1e-3, "K": 1e3, "k": 1e3, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "n": 1e-9,
        "p": 1e-12, "f": 1e-15, "m": 1e-3, "": 1.0}
_NUM = r"(?<![A-Za-z0-9_.\-])([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)"


def _pf(p):
    if p is None:
        return 1.0
    if p.upper() == "MEG":
        return 1e6
    if p == "M":   # 大写 M：电阻/电容里多为 mega（"2.2M"），法拉/亨利后缀前的 M 当 milli 很少见
        return 1e6
    return _PFX.get(p, _PFX.get(p.lower(), 1.0))


def parse_quantities(text):
    """从说明文字里抽出所有带单位的数值。返回 [(数值SI, 单位, 原文片段)]，单位 ∈ F H Ω V A W Hz ''。
    例：'Chemi-Con LXM 150uF 450V' → [(1.5e-4,'F',…), (450,'V',…)]；'MKP21 0.33uF 400V' 不会把 MKP21 当成 21。
    支持 4k7 / 2R2 / 0R47 / 0.47R / 39R / 1.05k 写法。"""
    if not text:
        return []
    s = str(text).replace("Ω", "R").replace("ohm", "R").replace("Ohm", "R")
    out = []
    # 4k7 2R2 0R47 4u7 这类“单位当小数点”写法
    for m in re.finditer(r"(?<![A-Za-z0-9_.\-])(\d+)([kKMRrunpµ])(\d+)(?![A-Za-z0-9.])", s):
        a, p, b = m.groups()
        f = 1.0 if p in "Rr" else _pf(p if p != "M" else "MEG")
        unit = "Ω" if p in "RrkKM" else ("F" if p in "unpµ" else "")
        out.append((float(f"{a}.{b}") * f, unit, m.group(0)))
    rx = _NUM + r"\s?(MEG|meg|[TGMkKmuµμnpf])?(F|H|R|V|A|W|Hz|VA|VDC|VAC)?(?![A-Za-z0-9])"
    for m in re.finditer(rx, s):
        num, p, u = m.group(1), m.group(2), m.group(3)
        x = float(num.replace(",", "."))
        unit = {"R": "Ω", "VDC": "V", "VAC": "V", "Hz": "Hz", "VA": "W"}.get(u, u or "")
        if p and not u and p in "kKM":   # 36k、2.2M：阻值
            unit = "Ω"
        if p == "M" and u in ("F", "H"):
            f = 1e-3  # 'mF' 大写写法 → milli
        else:
            f = _pf(p) if p else 1.0
        out.append((x * f, unit, m.group(0)))
    return out


def value_for(kind, text):
    """按器件类型挑出主值：C → 第一个 F 值，L → H，R → Ω（无单位的纯数也算）。认不出返回 None。"""
    q = parse_quantities(text)
    want = {"C": "F", "L": "H", "R": "Ω"}[kind]
    for x, u, _ in q:
        if u == want:
            return x
    # 第一个字段整体就是 SPICE 值（'100n'、'4.7u'、'10k'）
    from .spice import parse_value
    first = (text or "").split()
    if first:
        v = parse_value(first[0])
        if v is not None and not re.match(r"^[A-Za-z]", first[0]):
            return v
    if kind == "R":
        for x, u, _ in q:
            if u == "":
                return x
    return None


def ratings_from_text(text):
    """额定电压/电流（说明文字里的 'V'/'A' 值，取最大）：'S4M 1000V 4A' → {'V':1000,'A':4}。"""
    r = {}
    for x, u, _ in parse_quantities(text):
        if u in ("V", "A", "W") and x > 0:
            r[u] = max(r.get(u, 0), x)
    return r


# ======================================================================= 2) 功率二极管库
# 名称: (.model 参数, 额定 {V, A, IFSM}, 说明)。参数是按数据手册 VF/Cj/trr 典型值粗拟合的通用近似。
DIODES = {
    "S1M_GEN": ("Is=2n N=1.8 Rs=0.08 Cjo=12p M=0.4 Tt=1.8u BV=1100 Ibv=5u", dict(V=1000, A=1, IFSM=30), "1000 V 1 A 普通整流（S1M/1N4007 类）"),
    "S4M_GEN": ("Is=5n N=1.8 Rs=0.03 Cjo=40p M=0.4 Tt=2u BV=1100 Ibv=10u", dict(V=1000, A=4, IFSM=125), "1000 V 4 A 普通整流 SMC"),
    "GP10_GEN": ("Is=3n N=1.8 Rs=0.05 Cjo=15p M=0.4 Tt=2u BV=1100 Ibv=5u", dict(V=1000, A=1, IFSM=30), "ST/Vishay 'S10 GP' 玻璃钝化整流（型号未确认，按 1 kV 1 A 估）"),
    "SK54C_GEN": ("Is=5u N=1.2 Rs=0.015 Cjo=500p M=0.5 BV=45 Ibv=1m", dict(V=40, A=5, IFSM=150), "40 V 5 A 肖特基 SMC"),
    "SS34_GEN": ("Is=3u N=1.2 Rs=0.03 Cjo=250p M=0.5 BV=45 Ibv=1m", dict(V=40, A=3, IFSM=80), "40 V 3 A 肖特基"),
    "C3D06060_GEN": ("Is=5e-12 N=1.3 Rs=0.08 Cjo=300p M=0.45 Vj=1.2 BV=700 Ibv=50u", dict(V=600, A=6, IFSM=50), "600 V 6 A SiC 肖特基（通用近似，非 Cree 模型）"),
    "C4D05120_GEN": ("Is=3e-12 N=1.35 Rs=0.12 Cjo=250p M=0.45 Vj=1.2 BV=1350 Ibv=50u", dict(V=1200, A=5, IFSM=40), "1200 V 5 A SiC 肖特基（通用近似）"),
    "STTH1R06_GEN": ("Is=1n N=1.9 Rs=0.15 Cjo=10p M=0.4 Tt=20n BV=650 Ibv=5u", dict(V=600, A=1, IFSM=20), "600 V 1 A 超快恢复"),
    "STTH8R06_GEN": ("Is=5n N=1.8 Rs=0.03 Cjo=30p M=0.4 Tt=25n BV=650 Ibv=10u", dict(V=600, A=8, IFSM=100), "600 V 8 A 超快恢复"),
    "STTH512_GEN": ("Is=3n N=1.9 Rs=0.08 Cjo=20p M=0.4 Tt=40n BV=1250 Ibv=10u", dict(V=1200, A=5, IFSM=60), "1200 V 5 A 超快恢复"),
    "MUR160_GEN": ("Is=2n N=1.9 Rs=0.1 Cjo=15p M=0.4 Tt=50n BV=650 Ibv=5u", dict(V=600, A=1, IFSM=35), "600 V 1 A 超快恢复"),
    "ES1J_GEN": ("Is=2n N=1.9 Rs=0.1 Cjo=10p M=0.4 Tt=35n BV=650 Ibv=5u", dict(V=600, A=1, IFSM=30), "600 V 1 A 快恢复（自举二极管常用）"),
    "DLED_GEN": ("Is=1e-19 N=2 Rs=5 Cjo=20p BV=5 Ibv=10u", dict(V=5, A=0.03, IFSM=0.1), "通用指示 LED（Vf≈1.9 V@10 mA）"),
    "DZ_GEN": ("Is=1n N=1.5 Rs=1 Cjo=50p BV=15 Ibv=5m", dict(V=15, A=0.1, IFSM=1), "通用 15 V 稳压管（BV 按值字段里的 V 改写）"),
}
# 型号匹配（对说明文字整段，大写去空格）→ 库名
_DIODE_MATCH = [
    (r"C3D0606\d", "C3D06060"), (r"C4D05120", "C4D05120"), (r"C3D\d{5}", "C3D06060"), (r"C4D\d{5}", "C4D05120"),
    (r"(^|[^A-Z0-9])S4M|S4MB?($|[^A-Z0-9])", "S4M"), (r"(^|[^A-Z0-9])(S1M|1N4007|M7)($|[^A-Z0-9])", "S1M"),
    (r"SK5[45]C?|SK54", "SK54C"), (r"SS3[45]|SK3[45]", "SS34"), (r"STTH1R06", "STTH1R06"), (r"STTH8R06", "STTH8R06"),
    (r"STTH512|STTH5R12", "STTH512"), (r"MUR16\d|MURS16\d|UF400[57]", "MUR160"), (r"ES1[DGJ]|US1[DGJKM]", "ES1J"),
    (r"S10GP|GP\d{2,3}|S10", "GP10"),
    # 只有功能描述、没有型号时的通用默认（假设，GUI/报告里标注）
    (r"BOOTSTRAP|自举", "ES1J"), (r"ZENER|稳压", "DZ"), (r"^LED|(^|[^A-Z])LED($|[^A-Z])", "DLED"),
]
VENDOR_DIR = os.environ.get("RW_SPICE_MODELS", "")   # 无默认值：只有显式设置才引用供应商模型
_VINDEX = None


def vendor_index(root=None):
    """扫描供应商模型目录：{子电路/模型名大写: (文件, 'subckt'|'model', 引脚列表)}。小文件优先（摘录库优先于全系列库）。"""
    global _VINDEX
    root = root or VENDOR_DIR
    if not root or not os.path.isdir(root):   # 未设置 RW_SPICE_MODELS：不扫描任何目录
        return {}
    if _VINDEX is not None and _VINDEX[0] == root:
        return _VINDEX[1]
    idx = {}
    files = sorted(glob.glob(os.path.join(root, "**", "*.lib"), recursive=True) + glob.glob(os.path.join(root, "**", "*.mod"), recursive=True)
                   + glob.glob(os.path.join(root, "**", "*.sp*"), recursive=True), key=lambda p: os.path.getsize(p))
    for f in files:
        try:
            txt = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in re.finditer(r"(?im)^\s*\.subckt\s+(\S+)\s+([^\n]*)$", txt):
            pins = [x for x in m.group(2).split() if "=" not in x and x.lower() not in ("params:", "optional:")]
            idx.setdefault(m.group(1).upper(), (f, "subckt", pins))
        for m in re.finditer(r"(?im)^\s*\.model\s+(\S+)\s+(\w+)", txt):
            idx.setdefault(m.group(1).upper(), (f, "model", m.group(2).upper()))
    _VINDEX = (root, idx)
    return idx


def find_vendor(token, root=None):
    """token 如 'C3D06060' → 供应商库里 'C3D06060A'（完全相同或以 token 开头且只多 1–2 个封装后缀字母）。"""
    if not token:
        return None
    t = token.upper()
    idx = vendor_index(root)
    if t in idx:
        return t, idx[t]
    for k in sorted(idx):
        if k.startswith(t) and re.fullmatch(r"[A-Z]{1,2}(-?\w{0,3})?", k[len(t):]):
            return k, idx[k]
    return None


def match_diode(text, use_vendor=True):
    """说明文字 → (类型, 名称, 详情)。类型 'vendor'（供应商子电路/模型）| 'lib'（本库通用近似）| None。"""
    v = (text or "").upper().replace(" ", "")
    for rx, base in _DIODE_MATCH:
        m = re.search(rx, v)
        if not m:
            continue
        if use_vendor:
            tok = re.search(r"[A-Z0-9]{5,}", v[max(0, m.start() - 1):]) if base in ("C3D06060", "C4D05120") else None
            for cand in ([tok.group(0)] if tok else []) + [base]:
                hit = find_vendor(cand)
                if hit:
                    return "vendor", hit[0], hit[1]
        return "lib", base + "_GEN", DIODES[base + "_GEN"]
    return None, None, None


def diode_model_line(name, bv=None):
    p, _, _ = DIODES[name]
    if bv is not None:
        p = re.sub(r"BV=\S+", "BV=%g" % bv, p)
    return ".model %s D(%s)" % (name if bv is None else "%s_%g" % (name, bv), p)


# ======================================================================= 3) IGBT（通用近似）
IGBT_GEN = """.subckt IGBT_GEN c g e
* 通用 IGBT 近似（MIT）：串联结压降 + VDMOS 沟道 + 反并联续流二极管（带 copack 二极管的 xxDTP/xxH3 型）。
* 600 V 30 A 量级：Vce(sat)≈1.5 V@30 A、Vth≈5 V、Cies≈1.6 nF。不是英飞凌模型。
Dj c ci DIGBTJ
Mch ci g e MIGBTCH
Dfw e c DIGBTFW
.model DIGBTJ D(Is=1e-9 N=1 Rs=5m)
.model MIGBTCH VDMOS(Vto=5 Kp=12 Rd=10m Rs=5m Cgs=1.6n Cgdmax=600p Cgdmin=30p Cjo=150p Is=1e-14 BV=650)
.model DIGBTFW D(Is=5n N=1.6 Rs=20m Cjo=100p Tt=60n BV=650)
.ends IGBT_GEN"""
IGBT_RATINGS = {"IKW30N60": dict(V=600, A=30), "IKW40N65": dict(V=650, A=40), "IKW25N120": dict(V=1200, A=25), "IGBT": dict()}

# ======================================================================= 4) 驱动 / LDO 行为级子电路
# 平滑开关函数（避免 u() 的不连续导致不收敛）
_SW = "0.5*(1+tanh(({x})*{g}))"

IR2181X_BEH = """.subckt IR2181X_BEH vcc hin lin com lo vb ho vs params: td=180n uvcc=8.6 uvbs=8.0 rsrc=7.5
* 行为级 IR2181(4) 高低边驱动（MIT，按 IR2181/IR21814 数据手册典型值）：
*  输入阈值 ≈1.7 V（逻辑 0.8/2.5 V 中点）；传播延迟 td（RC 50 % 点）；VCC UVLO uvcc、VBS UVLO uvbs（无迟滞，简化）；
*  输出电阻 rsrc（1.9 A/2.3 A 量级）；静态电流 VCC ≈1 mA、VBS ≈60 µA。VS 耐压 600 V 由应力校核表判断。
Rih hin com 100k
Ril lin com 100k
Bhi hr com V=0.5*(1+tanh((V(hin,com)-1.7)*20))
Bli lr com V=0.5*(1+tanh((V(lin,com)-1.7)*20))
Rdh hr hd 1k
Cdh hd com {td/0.693/1k}
Rdl lr ld 1k
Cdl ld com {td/0.693/1k}
Bho hox vs V=V(vb,vs)*0.5*(1+tanh((V(hd,com)-0.5)*40))*0.5*(1+tanh((V(vb,vs)-uvbs)*20))*0.5*(1+tanh((V(vcc,com)-uvcc)*20))
Rho hox ho {rsrc}
Blo lox com V=V(vcc,com)*0.5*(1+tanh((V(ld,com)-0.5)*40))*0.5*(1+tanh((V(vcc,com)-uvcc)*20))
Rlo lox lo {rsrc}
Rqcc vcc com 15k
Rqbs vb vs 250k
.ends IR2181X_BEH"""

MIC442X_BEH = """.subckt MIC442X_BEH vplus gnd ina inb outa outb params: inva=1 invb=0 td=30n ro=7
* 行为级 MIC4426/4427/4428 双路 1.5 A 驱动（MIT）：MIC4428 = A 反相、B 同相（inva=1 invb=0）。
* 输入阈值 ≈1.6 V；延迟 td；输出电阻 ro；VS < 4.5 V 时输出拉低（数据手册工作范围 4.5–18 V，无内部 UVLO，简化）。
Ria ina gnd 1meg
Rib inb gnd 1meg
Ba ar gnd V=abs(inva-0.5*(1+tanh((V(ina,gnd)-1.6)*20)))
Bb br gnd V=abs(invb-0.5*(1+tanh((V(inb,gnd)-1.6)*20)))
Rda ar ad 1k
Cda ad gnd {td/0.693/1k}
Rdb br bd 1k
Cdb bd gnd {td/0.693/1k}
Boa oax gnd V=V(vplus,gnd)*0.5*(1+tanh((V(ad,gnd)-0.5)*40))*0.5*(1+tanh((V(vplus,gnd)-4.5)*20))
Roa oax outa {ro}
Bob obx gnd V=V(vplus,gnd)*0.5*(1+tanh((V(bd,gnd)-0.5)*40))*0.5*(1+tanh((V(vplus,gnd)-4.5)*20))
Rob obx outb {ro}
Rq vplus gnd 20k
.ends MIC442X_BEH"""

LDO_BEH = """.subckt LDO_BEH vin gnd en vout params: vset=5 vdo=0.28 ilim=0.35 iq=0.85m ven=1.4
* 行为级 LDO（MIT，LP2985 类：150 mA 额定，压差 280 mV@150 mA，限流 ≈350 mA，Iq ≈850 µA）。
* Vout = min(vset, Vin − vdo) × 使能；超过 ilim 后折返；输入电流 = 输出电流 + Iq。
Bset x gnd V=max(min(vset, V(vin,gnd)-vdo),0)*0.5*(1+tanh((V(en,gnd)-ven)*20))-max(I(Vsns)-ilim,0)*200
Vsns x y 0
Ro y vout 0.05
Bin vin gnd I=I(Vsns)+iq
.ends LDO_BEH"""

# KiCad 引脚名 → 子电路脚（大写，去 ~{} 等）
IC_BEH = {
    "IR2181X": dict(match=r"IR2181\d?|IR2181", sub="IR2181X_BEH", text=IR2181X_BEH,
                    pins=["VCC", "HIN", "LIN", "COM", "LO", "VB", "HO", "VS"], alias={"COM": ["COM", "VSS"]},
                    ratings={"VS": 600, "VB": 625, "VCC": 25}, note="行为级 IR2181x（UVLO/延迟/输出电阻；VS 600 V 由应力表报警）"),
    "MIC442X": dict(match=r"MIC442[678]", sub="MIC442X_BEH", text=MIC442X_BEH,
                    pins=["V+", "GND", "IN_A", "IN_B", "OUT_A", "OUT_B"], alias={"V+": ["V+", "VS", "VDD"], "OUT_A": ["OUT_A", "OUTA"], "OUT_B": ["OUT_B", "OUTB"], "IN_A": ["IN_A", "INA"], "IN_B": ["IN_B", "INB"]},
                    ratings={"V+": 18}, note="行为级 MIC442x 双路驱动"),
    "LDO": dict(match=r"LP2985|LP2981|LP2992|MIC5205|TLV7\d{3}|AP2112|XC6206|LDO", sub="LDO_BEH", text=LDO_BEH,
                pins=["VIN", "GND", "EN", "VOUT"], alias={"EN": ["ON/OFF", "EN", "SHDN", "ON"], "VOUT": ["VOUT", "OUT"], "VIN": ["VIN", "IN"]},
                ratings={"VIN": 16}, note="行为级 LDO"),
}


def ic_params(key, part, text):
    """按型号给子电路参数：MIC4426/4427/4428 的反相配置、LDO 的输出电压（型号后缀 -5.0 / -3.3）。"""
    up = (part + " " + (text or "")).upper()
    if key == "MIC442X":
        m = re.search(r"MIC442([678])", up)
        k = m.group(1) if m else "8"
        return {"inva": 1 if k in "68" else 0, "invb": 1 if k == "6" else 0}
    if key == "LDO":
        m = re.search(r"-(\d\.\d)", up)
        return {"vset": float(m.group(1)) if m else 5.0}
    return {}


# ======================================================================= 5) 保险丝 I²t
FUSE_I2T = """.subckt FUSE_I2T p n params: i2t=4 r0=10m roff=10meg
* 保险丝 I²t 熔断模型（MIT）：H 节点电压 = ∫i²dt（A²s），超过 i2t 后电阻在 ~2 % 范围内平滑升到 roff，并锁存（不恢复）。
* 只看熔化 I²t，不含电弧/分断时间；r0 是冷态电阻（假设）。
Vs p x 0
* 只在瞬态里积分（time>0）：否则直流工作点时 h = I²·Rh 会直接判熔断
Bi 0 h I=(time>0) ? I(Vs)*I(Vs) : 0
* 锁存：一旦越过 I²t，额外给 h 充电，几十微秒内推到完全断开（不会停在半熔断状态）
Bl 0 h I={i2t}*1e3*0.5*(1+tanh((V(h)/{i2t}-1)*200))*(V(h)<3*{i2t} ? 1 : 0)
Ch h 0 1
Rh h 0 1e15
* 电阻按对数插值 r0 → roff（过渡宽度约 ±1 % I²t），熔断后电流≈0、∫i²dt 不再下降 → 锁存
Bf x n I=V(x,n)/({r0}*exp(ln({roff}/{r0})*0.5*(1+tanh((V(h)/{i2t}-1)*200))))
.ends FUSE_I2T"""


def fuse_params(text, fields=None):
    """'T3.15A' / '2A 250V' → (In, I²t, 说明)。I²t 优先取字段 I2t；否则按"10 倍额定 10 ms 熔断"粗估 I²t = In²·1 s（假设）。"""
    fields = fields or {}
    i2t = None
    for k in ("I2t", "I2T", "i2t", "Sim.I2t"):
        if fields.get(k):
            from .spice import parse_value
            i2t = parse_value(fields[k].split()[0])
    r = ratings_from_text(text)
    m = re.search(r"(?i)(?<![A-Za-z])[FTM]?\s?(\d+(?:\.\d+)?)\s?(m?)A", text or "")
    In = r.get("A") or (float(m.group(1)) * (1e-3 if m.group(2) else 1) if m else None)
    note = []
    if In is None:
        In = 2.0; note.append("额定电流未知，假设 2 A")
    if i2t is None:
        i2t = In * In * 1.0; note.append("I²t 未给，按 10×In 约 10 ms 熔化估算 I²t=%.3g A²s（假设）" % i2t)
    return In, i2t, "；".join(note)


# ======================================================================= 6) 变压器 / 耦合电感
def xfmr_params(text, fields=None):
    """从 Sim.Params / 字段 / 值文字取 Lp、匝比、漏感、K。返回 (dict, 假设说明列表)。
    字段写法：Sim.Params='lp=1.2m n=12 lleak=15u k=0.9999'，或 Lp / N / Ratio('12:1:2') / Lleak / K 单独字段。"""
    from .spice import parse_value
    fields = fields or {}
    p = {}
    sp = fields.get("Sim.Params") or ""
    for kv in sp.split():
        if "=" in kv:
            k, v = kv.split("=", 1); p[k.lower()] = v
    for k in ("Lp", "N", "Ratio", "Lleak", "K", "N2", "N3"):
        if fields.get(k):
            p.setdefault(k.lower(), fields[k])
    note = []
    lp = parse_value(p["lp"]) if "lp" in p else None
    if lp is None:
        hv = [x for x, u, _ in parse_quantities(text) if u == "H"]
        lp = hv[0] if hv else None
    if lp is None:
        lp = 1e-3; note.append("Lp 未给，假设 1 mH")
    ratios = []
    if "ratio" in p:
        a = [float(x) for x in re.split(r"[:/]", p["ratio"]) if x]
        ratios = [a[0] / x for x in a[1:]] if len(a) > 1 else [a[0]]
    else:
        for k in ("n", "n2", "n3"):
            if k in p:
                ratios.append(float(parse_value(p[k])))
        m = re.search(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", text or "")
        if not ratios and m:
            ratios = [float(m.group(1)) / float(m.group(2))]
    if not ratios:
        ratios = [1.0]; note.append("匝比未给，假设 1:1")
    lleak = parse_value(p["lleak"]) if "lleak" in p else None
    if lleak is None:
        lleak = 0.01 * lp; note.append("漏感未给，假设 1 % Lp")
    k = float(p.get("k", 0.9999))
    return dict(lp=lp, ratios=ratios, lleak=lleak, k=k), note


# ======================================================================= 7) 电网激励与浪涌
GRID_PRESETS = {  # 名称: (线电压有效值倍数, 说明)
    "nom": (1.00, "三相 380 V/50 Hz 额定"),
    "hi10": (1.10, "+10 %（418 V）"),
    "hi20": (1.20, "+20 %（456 V）"),
    "lo10": (0.90, "−10 %（342 V）"),
    "lo20": (0.80, "−20 %（304 V）"),
}
# IEC 61000-4-5:2014 附录 A：1.2/50 µs 开路电压波形 v(t)=A·Vp·(1−e^(−t/τ1))·e^(−t/τ2)
SURGE_12_50 = dict(A=1.037, tau1=0.4074e-6, tau2=68.22e-6)
SURGE_LEVELS = {1: 500.0, 2: 1000.0, 3: 2000.0, 4: 4000.0}  # 试验等级 → 开路峰值 V


def surge_v(t, vp, t0=0.0):
    """1.2/50 µs 开路电压（V），t 秒（数组/标量）。"""
    import numpy as np
    x = np.maximum(np.asarray(t, float) - t0, 0.0)
    s = SURGE_12_50
    return s["A"] * vp * (1 - np.exp(-x / s["tau1"])) * np.exp(-x / s["tau2"])


def surge_pwl(vp, t0, n=60, tend=400e-6):
    """PWL 点表（对数加密前沿）。"""
    import numpy as np
    ts = np.unique(np.concatenate([[0.0], np.geomspace(5e-9, tend, n)]))
    vs = surge_v(ts, vp)
    pts = [(0.0, 0.0), (t0, 0.0)] + [(t0 + t, v) for t, v in zip(ts[1:], vs[1:])] + [(t0 + tend * 1.01, 0.0)]
    return pts


def add_grid(c, lines=("L1", "L2", "L3"), neutral=None, vll=380.0, f=50.0, preset="nom", zs=(72e-6, 0.3),
             surge=None, surge_t0=None, surge_mode="LL", surge_z=2.0):
    """在 Circuit c 上加三相电网激励（Y 接星点 neutral，默认内部节点 grid_n 经 1 MΩ 接地）。
    preset ∈ GRID_PRESETS；zs=(每相 L, R) 电网阻抗（假设值，默认 72 µH + 0.3 Ω）；
    surge=峰值 V（或试验等级 1–4）在 surge_t0 时刻（默认第 3 个电网周期电压峰附近）注入 IEC 61000-4-5 1.2/50 µs 组合波：
      'LL' 线-线：串联在 L1 相（2 Ω 内阻，即 IEC 规定的组合波有效内阻）；'LE' 线-地：经 10 Ω + 9 µF 耦合到 L1（IEC 61000-4-5 线-地耦合）。
    注意：用戴维南（1.2/50 开路电压 + 内阻）近似组合波发生器，短路电流波形不是严格的 8/20 µs。返回说明文字。"""
    k, desc = GRID_PRESETS[preset]
    vph = vll * k * math.sqrt(2) / math.sqrt(3)
    n = neutral or "grid_n"
    if neutral is None:
        c.R("grid_n", n, "0", 1e6)
    phases = (0, -120, 120)
    for i, (ln, ph) in enumerate(zip(lines, phases)):
        src = f"grid_{ln}_s"
        c.raw(f"Vgrid_{ln} {src} {n} SIN(0 {vph:.6g} {f:g} 0 0 {ph})")
        nxt = src
        if surge is not None and i == 0 and surge_mode.upper() == "LL":
            vp = SURGE_LEVELS.get(surge, surge)
            t0 = surge_t0 if surge_t0 is not None else 2.0 / f + 0.25 / f
            pts = surge_pwl(vp, t0)
            c.raw(f"Vsurge grid_{ln}_g {src} PWL(" + " ".join(repr(float(p[0])) + " " + repr(float(p[1])) for p in pts) + ")")
            c.R("surge", f"grid_{ln}_g", f"grid_{ln}_z", surge_z)
            nxt = f"grid_{ln}_z"
        c.L(f"grid_{ln}", nxt, f"grid_{ln}_r", zs[0])
        c.R(f"grid_{ln}", f"grid_{ln}_r", ln, zs[1])
    if surge is not None and surge_mode.upper() == "LE":
        vp = SURGE_LEVELS.get(surge, surge)
        t0 = surge_t0 if surge_t0 is not None else 2.0 / f + 0.25 / f
        pts = surge_pwl(vp, t0)
        c.raw("Vsurge surge_s 0 PWL(" + " ".join(repr(float(p[0])) + " " + repr(float(p[1])) for p in pts) + ")")
        c.R("surge", "surge_s", "surge_c", 10.0)
        c.C("surge", "surge_c", lines[0], 9e-6)
    out = f"电网 {desc}，相电压峰值 {vph:.1f} V，电网阻抗 {zs[0] * 1e6:g} µH + {zs[1]:g} Ω/相（假设）"
    if surge is not None:
        out += f"；IEC 61000-4-5 1.2/50 µs 浪涌 {SURGE_LEVELS.get(surge, surge):g} V（{surge_mode}，戴维南近似）"
    return out
