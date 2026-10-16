"""118 种元素基础物性（开源数据：pymatgen + mendeleev，见 build_elements.py）。

    from rw.matdb.elements import EL, get
    get('Fe')['density']   # kg/m3
    get('铁')              # 中文名也可以
"""
import json, os

_D = json.load(open(os.path.join(os.path.dirname(__file__), "elements.json"), encoding="utf-8"))
META = _D["meta"]
ELEMENTS = _D["elements"]
EL = {e["sym"]: e for e in ELEMENTS}
_ZH = {e["zh"]: e for e in ELEMENTS}
_Z = {e["Z"]: e for e in ELEMENTS}

STRUCT_ZH = {"BCC": "体心立方 bcc", "FCC": "面心立方 fcc", "HEX": "密排六方 hcp", "DIA": "金刚石型", "CUB": "立方(复杂)",
             "SC": "简单立方", "TET": "四方", "ORC": "正交", "MCL": "单斜", "RHL": "菱方"}
DENS_SRC = {"solid": "实测(固态, 20°C)", "gas_25C": "气体, 25°C 1atm", "liquid": "液态, 20°C",
            "predicted": "理论预测(从没称出过)", "mendeleev": "实测"}

# 元素类别（用来判断能不能当“金属合金”的组元）
NOBLE_GAS = {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"}
NONMETAL_REACTIVE = {"H", "N", "O", "F", "S", "Cl", "Se", "Br", "I", "P"}   # 跟金属形成化合物，而不是合金
METALLOID = {"B", "C", "Si", "Ge", "As", "Sb", "Te"}
ALKALI = {"Li", "Na", "K", "Rb", "Cs", "Fr"}


def get(x):
    """按元素符号 / 中文名 / 原子序数 取元素。"""
    if isinstance(x, int):
        return _Z[x]
    x = str(x).strip()
    if x in EL:
        return EL[x]
    if x in _ZH:
        return _ZH[x]
    for k in EL:
        if k.lower() == x.lower():
            return EL[k]
    raise KeyError(f"没有这个元素: {x}")


def is_transition(sym):
    e = EL[sym]
    return e["block"] in ("d", "f")


def short_lived(sym):
    """Z≥100 的元素一次只造出几个原子，半衰期秒级以下，做不成任何材料。"""
    return EL[sym]["Z"] >= 100


def category(sym):
    if sym in NOBLE_GAS: return "惰性气体"
    if sym in NONMETAL_REACTIVE: return "非金属"
    if sym in METALLOID: return "类金属"
    if sym in ALKALI: return "碱金属"
    e = EL[sym]
    if e["block"] == "f": return "镧系/锕系"
    if e["block"] == "d": return "过渡金属"
    if e["block"] == "s": return "碱土金属" if sym != "H" else "非金属"
    return "主族金属"


def describe(sym):
    """一个元素的中文说明（给界面用）。"""
    e = get(sym); u = []
    f = lambda v, fmt: "—" if v is None else fmt % v
    u.append(f"{e['Z']}  {e['sym']}  {e['zh']}（{e['name']}）  原子量 {e['mass']:.3f}   类别：{category(e['sym'])}")
    u.append(f"常温状态：{e['phase']}" + ("    ☢ 放射性" if e["radioactive"] else ""))
    u.append(f"密度：{f(e['density'], '%.5g')} kg/m³  [{DENS_SRC.get(e['density_src'], e['density_src'])}]")
    u.append(f"熔点：{f(e['melt_C'], '%.1f')} °C    沸点：{f(e['boil_C'], '%.0f')} °C")
    u.append(f"电阻率：{f(None if e['resist'] is None else e['resist'] * 1e8, '%.4g')} μΩ·cm    导热：{f(e['k'], '%.4g')} W/(m·K)")
    u.append(f"比热：{f(e['cp'], '%.4g')} J/(kg·K)    线膨胀：{f(None if e['cte'] is None else e['cte'] * 1e6, '%.3g')} μm/(m·K)")
    u.append(f"弹性模量 E：{f(e['E'], '%.4g')} GPa   剪切 G：{f(e['G'], '%.3g')} GPa   泊松比：{f(e['nu'], '%.2f')}")
    u.append(f"布氏硬度：{f(e['HB'], '%.4g')} MPa（开源库数值，个别元素明显偏离手册，仅参考）")
    st = STRUCT_ZH.get(e["struct"], e["struct"] or "—")
    u.append(f"晶体结构：{st}   晶格常数 {f(e['a'], '%.3f')} Å   金属半径 {f(e['r_met'], '%.0f')} pm   电负性 {f(e['chi'], '%.2f')}")
    u.append(f"参考价：{f(e['price'], '%.4g')} USD/kg（数据库参考价，年份不一，只看量级）")
    if short_lived(e["sym"]):
        u.append("⚠ 这种元素全世界只合成过很少的原子，几秒甚至毫秒就衰变，物性多为理论预测，不能做材料。")
    return "\n".join(u)
