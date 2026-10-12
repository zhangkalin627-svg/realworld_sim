"""任意配比“配合金”：先判断能不能互溶，再按混合法则算密度/成本，强度只对已知牌号给实测值。

判断互溶的依据（按可信度从高到低）：
  1. 已知二元相图结论（KNOWN_BINARY，教科书/ASM 二元相图手册的常识结论）
  2. 模型估算：Miedema 半经验混合焓 + Hume-Rothery 固溶规则（原子尺寸差、晶体结构、电负性）
数值说明：
  - 密度：理想混合（比容按质量分数相加）1/ρ = Σ wᵢ/ρᵢ；实际合金一般在 ±3% 内（例：黄铜实测比算的高 3%）
  - 成本：Σ wᵢ·单价ᵢ（只是原料，不含冶炼加工）
  - 弹性模量：Voigt/Reuss 上下界（复合材料的严格界限）
  - 电阻率：两相混合在串/并联界限之间；固溶体比按比例算高很多（Nordheim 规律），只给范围
  - 强度：配比和已知牌号一致 → 牌号实测/标准值；否则只给“粗估”量级范围，不能用于设计
"""
import math
from .elements import EL, get, short_lived, is_transition, NOBLE_GAS, NONMETAL_REACTIVE, category
from .alloys import GRADES, match_grade, norm_comp

R_GAS = 8.314

# ---------------------------------------------------------------- 已知二元相图结论
# kind: ss=任意比例完全互溶(固溶体)  limited=部分固溶/共晶  imc=形成金属间化合物  liq_gap=液态就不互溶(分层)
_KB = [
    ("Cu", "Ni", "ss", "铜镍同为面心立方，任意比例互溶（白铜、康铜、蒙乃尔）"),
    ("Ag", "Au", "ss", "金银任意比例互溶"),
    ("Au", "Cu", "ss", "高温下任意比例互溶，低温有有序相 Cu3Au/CuAu"),
    ("Ag", "Pd", "ss", "任意比例互溶"), ("Au", "Pd", "ss", "任意比例互溶"), ("Cu", "Pd", "ss", "高温任意比例互溶"),
    ("Ni", "Pd", "ss", "任意比例互溶"), ("Ni", "Pt", "ss", "任意比例互溶"), ("Pd", "Pt", "ss", "任意比例互溶"),
    ("Mo", "W", "ss", "任意比例互溶"), ("Nb", "Ta", "ss", "任意比例互溶"), ("Mo", "Nb", "ss", "任意比例互溶"),
    ("Ta", "W", "ss", "任意比例互溶"), ("Nb", "W", "ss", "任意比例互溶"), ("Mo", "Ta", "ss", "任意比例互溶"),
    ("Cr", "V", "ss", "任意比例互溶"), ("Ti", "Zr", "ss", "任意比例互溶"), ("Hf", "Zr", "ss", "任意比例互溶"),
    ("Ge", "Si", "ss", "锗硅任意比例互溶（半导体 SiGe）"), ("Bi", "Sb", "ss", "任意比例互溶"),
    ("Co", "Ni", "ss", "高温任意比例互溶"),
    ("Fe", "Ni", "ss", "高温奥氏体区任意比例互溶（殷钢、坡莫合金）"),
    ("Cr", "Fe", "ss", "高温任意比例互溶；中间成分 500–800°C 会析出脆性 σ 相"),
    ("Ag", "Fe", "liq_gap", "铁和银连液态都不互溶，熔化后像油和水一样分层；固态互溶度几乎为零"),
    ("Fe", "Pb", "liq_gap", "液态不互溶，铅沉底"), ("Bi", "Fe", "liq_gap", "液态不互溶"),
    ("Fe", "Mg", "liq_gap", "液态不互溶（镁的沸点 1090°C 还低于铁的熔点）"),
    ("Cu", "Pb", "liq_gap", "液态有大的不混溶区（955°C 偏晶），铅以颗粒分布——铅青铜就是利用这一点"),
    ("Al", "Pb", "liq_gap", "液态不互溶"), ("Al", "Bi", "liq_gap", "液态不互溶"), ("Al", "In", "liq_gap", "液态不互溶区"),
    ("Pb", "Zn", "liq_gap", "液态不互溶区（炼锌除铅就靠这个）"), ("Ag", "Ni", "liq_gap", "液态、固态都几乎不互溶"),
    ("Ag", "Co", "liq_gap", "液态、固态都几乎不互溶"), ("Ag", "Cr", "liq_gap", "液态不互溶"),
    ("Cu", "W", "liq_gap", "几乎完全不互溶，只能粉末冶金（钨铜触头）"), ("Cu", "Mo", "liq_gap", "几乎不互溶，只能粉末冶金（钼铜）"),
    ("Ag", "W", "liq_gap", "不互溶，只能粉末冶金（银钨触头）"),
    ("Ga", "Pb", "liq_gap", "液态不互溶"),
    ("Ag", "Cu", "limited", "共晶 779°C；银里最多溶 8.8 wt% 铜（925 银就在这个范围内）"),
    ("Pb", "Sn", "limited", "共晶 183°C（Sn63Pb37 焊锡）"), ("Al", "Si", "limited", "共晶 577°C、12.6% Si（铸铝 ZL102）"),
    ("Al", "Cu", "limited", "铝里最多溶 5.65 wt% 铜，多了析出 θ(CuAl2)，硬铝靠这个时效强化"),
    ("Cu", "Zn", "limited", "≤约 39% Zn 为单相 α 黄铜，再多出现 β 相，变硬变脆"),
    ("Cu", "Sn", "limited", "≤约 15.8% Sn 为 α 青铜，多了出现脆性 δ 相"),
    ("Al", "Mg", "limited", "铝里最多溶约 15 wt% 镁"), ("Al", "Zn", "limited", "高温溶解度大，低温析出"),
    ("Al", "Mg_", "limited", ""), ("Bi", "Sn", "limited", "共晶 139°C（低温焊料）"),
    ("Ag", "Sn", "limited", "共晶 221°C，有 Ag3Sn"), ("Cu", "Sn_", "limited", ""),
    ("C", "Fe", "limited", "钢 ≤2.11% C，铸铁 2.11–6.69% C；碳多了以渗碳体/石墨存在"),
    ("Cu", "Fe", "limited", "液态互溶，固态几乎不溶（铜里 <0.1% 铁，铁里约 2% 铜）；冷却后铜铁各成一相"),
    ("Cu", "Mn", "ss", "高温任意比例互溶（锰铜电阻合金）"),
    ("Fe", "Mn", "limited", "大范围固溶（锰钢），高温 γ 相区宽"),
    ("Al", "Ni", "imc", "形成 NiAl、Ni3Al 等金属间化合物（高温合金强化相），硬而脆"),
    ("Al", "Ti", "imc", "形成 TiAl、Ti3Al 金属间化合物；少量 Al（≤6%）固溶在钛里（TC4）"),
    ("Al", "Fe", "imc", "富铁端可固溶约 10% Al；中间形成 FeAl、Fe3Al、FeAl3 等脆性相（铝铁焊接难点）"),
    ("Ni", "Ti", "imc", "NiTi 形状记忆合金（等原子比金属间化合物）"),
    ("Fe", "Ti", "imc", "形成 FeTi、Fe2Ti"), ("Nb", "Sn", "imc", "Nb3Sn 超导体"),
    ("Au", "Si", "limited", "共晶 363°C，几乎不固溶"), ("Al", "Li", "imc", "形成 AlLi"),
]
KNOWN_BINARY = {}
for a, b, kind, txt in _KB:
    if a.endswith("_") or b.endswith("_") or not txt:
        continue
    KNOWN_BINARY[frozenset((a, b))] = (kind, txt)

KIND_ZH = {"ss": "能任意比例互溶（单相固溶体）", "limited": "部分互溶（有溶解度极限，超出析出第二相/共晶）",
           "imc": "会形成金属间化合物（通常硬而脆）", "liq_gap": "不互溶（液态就分层，熔炼做不成合金）",
           "compound": "不是合金：会反应成化合物（氧化物/氮化物/盐等）", "inert": "惰性气体，不参与成合金",
           "na": "做不成材料（超重元素寿命太短）", "unknown": "缺少数据，判断不了"}

# Miedema 杂化项 R/P（过渡金属与 p 区多价元素混合时才有），常用值（近似）
_RP = {"Al": 1.9, "Ga": 1.9, "In": 1.9, "Tl": 1.9, "Si": 2.1, "Ge": 1.9, "Sn": 1.9, "Pb": 1.9, "Sb": 1.9, "Bi": 1.9,
       "B": 1.9, "C": 2.1, "N": 2.3, "P": 2.1, "As": 2.1, "Zn": 1.4, "Cd": 1.4, "Hg": 1.4, "Mg": 0.4, "Be": 0.4,
       "Ca": 0.4, "Sr": 0.4, "Ba": 0.4}


def wt_to_at(wt):
    w = norm_comp(wt)
    n = {e: v / EL[e]["mass"] for e, v in w.items()}
    s = sum(n.values())
    return {e: 100 * v / s for e, v in n.items()}


def at_to_wt(at):
    a = norm_comp(at)
    m = {e: v * EL[e]["mass"] for e, v in a.items()}
    s = sum(m.values())
    return {e: 100 * v / s for e, v in m.items()}


def _mied_tm(s):
    """Miedema 模型里的“过渡金属”：3–11 族 d 区（含 Cu/Ag/Au）和镧系锕系；Zn/Cd/Hg 算非过渡。"""
    e = EL[s]
    return e["block"] == "f" or (e["block"] == "d" and (e["group"] or 3) <= 11)


def miedema_dH(A, B, xA=0.5):
    """液态/无序固溶体混合焓 kJ/mol（Miedema 模型，不含弹性与结构项）。缺参数返回 None。"""
    a, b = EL[A], EL[B]
    if None in (a["mied_phi"], a["mied_V"], a["mied_n"], b["mied_phi"], b["mied_V"], b["mied_n"]):
        return None
    tA, tB = _mied_tm(A), _mied_tm(B)
    P = 14.1 if (tA and tB) else (10.7 if not (tA or tB) else 12.35)
    QP = 9.4
    RP = 0.0
    if tA != tB:
        nt, tm = (B, A) if tA else (A, B)
        if (EL[tm]["group"] or 0) <= 10:     # 铜/银/金（11 族）d 壳层已满，杂化项很小，不计
            RP = _RP.get(nt, 0.0)
    dphi = a["mied_phi"] - b["mied_phi"]
    n13A, n13B = a["mied_n"] ** (1 / 3), b["mied_n"] ** (1 / 3)
    dn = n13A - n13B
    VA, VB = a["mied_V"] ** (2 / 3), b["mied_V"] ** (2 / 3)
    core = (-(dphi ** 2) + QP * dn ** 2 - RP)
    denom = (1 / n13A + 1 / n13B)
    H_A_in_B = 2 * P * VA * core / denom
    xB = 1 - xA
    csA = xA * VA / (xA * VA + xB * VB); csB = 1 - csA
    return xA * csB * H_A_in_B   # ΔH_mix = x_A · c_B^s · ΔH(A 溶于 B)


def hume_rothery(A, B):
    """返回 (尺寸差%, 结构相同?, 电负性差, 文字)。"""
    a, b = EL[A], EL[B]
    out = []
    size = None
    if a["r_met"] and b["r_met"]:
        size = abs(a["r_met"] - b["r_met"]) / min(a["r_met"], b["r_met"]) * 100
        out.append(f"原子尺寸差 {size:.1f}%（≤15% 才容易固溶）")
    same = (a["struct"] is not None and a["struct"] == b["struct"])
    out.append(f"晶体结构 {a['struct']}/{b['struct']}" + ("（相同）" if same else "（不同）"))
    dchi = None
    if a["chi"] and b["chi"]:
        dchi = abs(a["chi"] - b["chi"])
        out.append(f"电负性差 {dchi:.2f}" + ("（大，倾向形成化合物）" if dchi > 0.4 else ""))
    return size, same, dchi, "；".join(out)


def model_binary(A, B):
    """模型判断一对元素（不查已知表）。返回 (kind, 说明, dH50)"""
    H = miedema_dH(A, B, 0.5)
    size, same, dchi, hr = hume_rothery(A, B)
    if H is None:
        return "unknown", "缺 Miedema 参数；" + hr, None
    Om = 4 * H * 1000  # J/mol 正规溶液参数
    Tc = Om / (2 * R_GAS)  # 不混溶临界温度 K
    Tl = max(EL[A]["melt_C"] or 0, EL[B]["melt_C"] or 0) + 273.15
    txt = f"Miedema 混合焓(50/50) {H:+.1f} kJ/mol；{hr}"
    if H > 0 and Tc > 1.15 * Tl:
        return "liq_gap", txt + f"；不混溶临界温度≈{Tc - 273:.0f}°C 高于熔点 → 液态分层", H
    if H < -15:
        return "imc", txt + " → 强烈吸引，倾向形成化合物", H
    if size is not None and size <= 12 and same and H <= 4:
        return "ss", txt + " → 满足固溶规则", H
    return "limited", txt, H


def binary(A, B):
    """一对元素能不能互溶：(kind, 说明, 依据)。依据='相图' 或 '模型估算'。"""
    for s in (A, B):
        if s in NOBLE_GAS:
            return "inert", f"{EL[s]['zh']}是惰性气体，不和任何金属成合金", "化学常识"
        if short_lived(s):
            return "na", f"{EL[s]['zh']}(Z={EL[s]['Z']}) 只合成过几个原子，寿命极短，做不成材料", "核物理事实"
    k = KNOWN_BINARY.get(frozenset((A, B)))
    if k:
        return k[0], k[1], "二元相图"
    for s, o in ((A, B), (B, A)):
        if s in NONMETAL_REACTIVE and category(o) not in ("非金属", "惰性气体"):
            if s in ("H", "N") :
                return "limited", f"{EL[s]['zh']}只能少量间隙溶入{EL[o]['zh']}，多了生成{'氢' if s == 'H' else '氮'}化物", "化学常识"
            return "compound", f"{EL[s]['zh']}和{EL[o]['zh']}会反应生成化合物，不是合金", "化学常识"
    kind, txt, H = model_binary(A, B)
    return kind, txt, "模型估算"



SEVERITY = {"ss": 0, "limited": 1, "unknown": 1, "imc": 2, "compound": 3, "liq_gap": 3, "inert": 3, "na": 4}


def _range(vals):
    return min(vals), max(vals)


def mix(comp_wt, minor=2.0):
    """comp_wt: {'Fe':70,'Ag':30} 质量百分比（自动归一化）。返回结果字典。"""
    w = norm_comp({get(k)["sym"]: v for k, v in comp_wt.items()})
    x = wt_to_at(w)
    els = list(w)
    res = dict(wt=w, at=x, pairs=[], warnings=[])
    # ---- 互溶判断：所有 ≥minor wt% 的组元两两判断
    major = [e for e in els if w[e] >= minor] or els
    worst = "ss"
    for i in range(len(major)):
        for j in range(i + 1, len(major)):
            k, t, src = binary(major[i], major[j])
            res["pairs"].append((major[i], major[j], k, t, src))
            if SEVERITY[k] > SEVERITY[worst]:
                worst = k
    if len(major) == 1:
        worst = "ss"
    res["verdict"] = worst
    res["verdict_zh"] = KIND_ZH[worst]
    if len(major) == 1 and len(els) > 1:
        res["verdict_zh"] = f"以{EL[major[0]]['zh']}为基体，少量元素溶入（稀合金，可以熔炼）"
    if worst in ("compound", "inert", "na"):
        res["strength"] = dict(src="无", Rm_range=None, note="不是合金，不估算力学性能")
        res["grade"] = None
        return res
    # ---- 密度：理想混合 + ±3%
    if all(EL[e]["density"] for e in els):
        # 气体元素（N、O、H…）少量时以间隙原子溶在金属里：有质量、几乎不占体积
        gas = [e for e in els if EL[e]["density_src"] == "gas_25C"]
        if sum(w[e] for e in gas) > 2:
            gas = []
        rho = 1 / sum(w[e] / 100 / EL[e]["density"] for e in els if e not in gas)
        res["density"] = rho; res["density_range"] = (0.97 * rho, 1.035 * rho)
        if any(EL[e]["density_src"] == "gas_25C" and w[e] > 0.5 for e in els):
            res["warnings"].append("含较多气体元素：按气体密度混合没有意义（气体会以化合物或间隙原子形式存在）")
    # ---- 成本
    if all(EL[e]["price"] is not None for e in els):
        res["cost"] = sum(w[e] / 100 * EL[e]["price"] for e in els)
    # ---- 熔化范围（粗估）
    mj = [e for e in major if EL[e]["melt_C"] is not None and EL[e]["phase"] == "固"]
    g0 = match_grade(w)
    if g0:
        res["melt_est"] = GRADES[g0]["melt"]; res["melt_note"] = "牌号手册值（固相线–液相线）"
    elif mj:
        mps = [EL[e]["melt_C"] for e in mj]
        lin = sum(w[e] * EL[e]["melt_C"] for e in mj) / sum(w[e] for e in mj)
        if worst == "ss" or len(mj) == 1:
            res["melt_est"] = (min(mps) + 0.6 * (lin - min(mps)), lin)
            res["melt_note"] = "粗估：固溶体在两组元熔点之间熔化（按质量比例插值）"
        else:
            res["melt_est"] = (min(mps), max(mps))
            res["melt_note"] = "粗估：有共晶/第二相时，开始熔化的温度可能比低熔点组元还低"
    # ---- 弹性模量 Voigt/Reuss（体积分数）
    vf = {}
    if "density" in res:
        vv = {e: (0 if EL[e]["density_src"] == "gas_25C" else w[e] / EL[e]["density"]) for e in els}
        vf = {e: vv[e] / sum(vv.values()) for e in els}
        Es = {e: EL[e]["E"] for e in major}
        if all(Es.values()):
            ev = sum(vf[e] * Es[e] for e in major) / sum(vf[e] for e in major)
            er = 1 / (sum(vf[e] / Es[e] for e in major) / sum(vf[e] for e in major))
            res["E_range"] = (er, ev)
    # ---- 电阻率范围
    # 只按金属组元算电阻率（Si、C 等半导体/非金属按溶质原子处理）
    rs = {e: EL[e]["resist"] for e in major if EL[e]["resist"] and EL[e]["resist"] < 1e-5}
    if len(rs) > 1 and len(rs) == len([e for e in major if EL[e]["resist"] and EL[e]["resist"] < 1e-5]):
        vs = {e: vf.get(e, x[e] / 100) for e in rs}; tv = sum(vs.values())
        par = 1 / sum(vs[e] / tv / rs[e] for e in rs)
        ser = sum(vs[e] / tv * rs[e] for e in rs)
        if worst in ("ss", "limited", "imc"):
            xs = 1 - max(x.values()) / 100  # 溶质原子分数
            # Nordheim: ρ = ρ_混合 + C·x(1-x)，C 实测约 0.1–3.5 μΩ·m（黄铜偏小，康铜、铁镍铬偏大）
            res["resist_range"] = (ser + 0.1e-6 * xs * (1 - xs), ser + 3.5e-6 * xs * (1 - xs))
            res["resist_note"] = "固溶体：杂质原子散射电子，电阻率比按比例算的高得多（例：康铜是纯铜的 28 倍）"
        else:
            res["resist_range"] = (par, ser)
            res["resist_note"] = "两相混合：在并联(连通好)和串联(连通差)之间"
    # ---- 强度
    g = match_grade(w)
    res["grade"] = g
    if g:
        G = GRADES[g]
        res["strength"] = dict(src="牌号实测/标准值", grade=g, Rp=G["Rp"], Rm=G["Rm"], A=G["A"], basis=G["basis"], state=G["state"])
    else:
        res["strength"] = rough_strength(w, x, worst)
    if worst in ("liq_gap",):
        res["warnings"].append("熔炼会分层：只能用粉末冶金（压制+烧结/熔渗）做成“假合金”或复合材料，性能按复合材料看")
    if worst == "imc":
        res["warnings"].append("会形成金属间化合物：硬、脆、难加工，强度粗估不可靠")
    for e in els:
        if EL[e]["radioactive"] and w[e] > 0.01:
            res["warnings"].append(f"{EL[e]['zh']}有放射性")
    return res


# 退火态工业纯金属的典型抗拉强度 MPa（手册量级；纯度、晶粒、加工态影响很大）
PURE_RM = {"Fe": 250, "Cu": 220, "Al": 70, "Ni": 400, "Ti": 240, "Mg": 160, "Zn": 120, "Sn": 15, "Pb": 17,
           "Ag": 140, "Au": 120, "Pt": 140, "Pd": 170, "Mo": 650, "W": 900, "Ta": 300, "Nb": 275, "Co": 250,
           "Zr": 330, "Cr": 280, "V": 250, "Be": 370, "Hf": 450}


def pure_Rm(s):
    if s in PURE_RM:
        return PURE_RM[s]
    hb = EL[s]["HB"]
    return None if hb is None else 0.35 * hb   # 没有手册值：按布氏硬度 Rm≈HB/3 推算（数据质量一般）


def rough_strength(w, x, verdict):
    """粗估抗拉强度范围 MPa（只有量级意义）。"""
    base = max(w, key=w.get)
    e = EL[base]
    if base == "Fe" and 0.02 <= w.get("C", 0) <= 2.1:
        c = w["C"]; lo = 330 + 450 * c; hi = 360 + 650 * c
        return dict(src="粗估", Rm_range=(lo, hi), note=f"碳钢 {c:.2f}%C 正火态经验范围；淬火+回火可到 2 倍以上，强度主要看热处理")
    pure = pure_Rm(base)
    if pure is None:
        return dict(src="粗估", Rm_range=None, note="基体元素没有强度数据，无法粗估")
    xs = 1 - x[base] / 100
    if verdict == "liq_gap":
        vals = [pure] + [pure_Rm(k) for k in w if k != base and pure_Rm(k)]
        return dict(src="粗估", Rm_range=(0.5 * min(vals), max(vals)),
                    note="两种不互溶的金属：界面结合差，强度在两者之间甚至更低（铸造会分层，只能粉末冶金）")
    hi = pure * min(3.0, 1 + 8 * xs)
    return dict(src="粗估", Rm_range=(0.7 * pure, hi),
                note=f"以 {e['zh']} 退火态 ≈{pure:.0f} MPa 为下限，固溶/析出强化最多到约 3 倍；冷加工、热处理影响更大")


def report(res):
    """mix() 结果 → [(级别, 文字)]，级别 OK/WARN/FAIL。"""
    v = []
    lv = {"ss": "OK", "limited": "WARN", "unknown": "WARN", "imc": "WARN"}.get(res["verdict"], "FAIL")
    v.append((lv, "结论：" + res["verdict_zh"]))
    for a, b, k, t, src in res["pairs"]:
        l2 = {"ss": "OK", "limited": "WARN", "unknown": "WARN", "imc": "WARN"}.get(k, "FAIL")
        v.append((l2, f"{EL[a]['zh']}-{EL[b]['zh']}：{KIND_ZH[k]}  [{src}] {t}"))
    if res.get("grade"):
        G = GRADES[res["grade"]]
        v.append(("OK", f"成分与牌号 {G['name']}（{G['std']}）一致 → 强度用该牌号{'标准最小值' if G['basis'] == 'min' else '手册典型值'}"))
        v.append(("OK", f"牌号实测：密度 {G['rho']} kg/m³，导热 {G['k']} W/(m·K)，电阻率 {G['resist'] * 1e8:.3g} μΩ·cm，"
                        f"线膨胀 {G['cte'] * 1e6:.3g} μm/(m·K)" + (f"，E {G['E']} GPa" if G['E'] else "")))
        if G.get("note"):
            v.append(("WARN", G["note"]))
    if "density" in res:
        a, b = res["density_range"]
        v.append(("OK", f"密度 ≈ {res['density']:.0f} kg/m³（混合法则；实际一般在 {a:.0f}–{b:.0f}）"))
    if "cost" in res:
        v.append(("OK", f"原料成本 ≈ {res['cost']:.2f} USD/kg（数据库参考价，只看量级，不含冶炼加工）"))
    if "melt_est" in res:
        a, b = res["melt_est"]
        v.append(("WARN", f"熔化范围 粗估 {a:.0f}–{b:.0f}°C（{res['melt_note']}）"))
    if "E_range" in res:
        a, b = res["E_range"]
        v.append(("OK", f"弹性模量 {a:.0f}–{b:.0f} GPa（Voigt/Reuss 上下界）"))
    if "resist_range" in res:
        a, b = res["resist_range"]
        v.append(("WARN", f"电阻率 粗估 {a * 1e8:.2f}–{b * 1e8:.1f} μΩ·cm（{res['resist_note']}）"))
    s = res["strength"]
    if s["src"] == "无":
        v.append(("FAIL", s["note"]))
    elif s["src"] != "粗估" and s["Rm"] is None:
        v.append(("WARN", "该牌号的强度数据没有收录（不瞎编）"))
    elif s["src"] != "粗估":
        rm = s["Rm"]; rm = f"{rm[0]}–{rm[1]}" if isinstance(rm, tuple) else rm
        v.append(("OK", f"强度（{s['state']}）：屈服 {s['Rp'] or '—'} MPa，抗拉 {rm or '—'} MPa，伸长率 {s['A'] or '—'}%"))
    elif s.get("Rm_range"):
        a, b = s["Rm_range"]
        v.append(("WARN", f"抗拉强度 粗估 {a:.0f}–{b:.0f} MPa —— 不是实测，不能用于设计。{s['note']}"))
    else:
        v.append(("WARN", "强度：" + s["note"]))
    for t in res["warnings"]:
        v.append(("FAIL" if "分层" in t or "放射" in t else "WARN", t))
    return v


def check_against_grades():
    """自检：混合法则密度 vs 牌号实测密度；模型互溶判断 vs 已知相图。"""
    rows = []
    for k, g in GRADES.items():
        r = mix(g["comp"])
        if "density" in r:
            rows.append((k, g["rho"], r["density"], (r["density"] - g["rho"]) / g["rho"] * 100))
    agree = []
    for pair, (kind, _) in KNOWN_BINARY.items():
        A, B = sorted(pair)
        mk, _, H = model_binary(A, B)
        if mk == "unknown":
            continue
        agree.append((A, B, kind, mk, kind == mk or (kind == "limited" and mk == "ss" and False)))
    return rows, agree
