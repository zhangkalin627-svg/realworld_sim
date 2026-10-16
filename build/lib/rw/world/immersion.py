"""沉在液体里的仪表/零件：静压、沸点、浮力、额定值校核。
p(h) = p_液面 + ρ(T)·g·h  （ρ 随温度变；液体按不可压缩，深 1000 m 内误差 <0.5%）"""
import math
from dataclasses import dataclass, field
from .fluids import get_fluid, G, P_ATM

OK, WARN, FAIL = "OK", "WARN", "FAIL"


@dataclass
class PressureGauge:
    name: str = "耐高温压力表"
    range_MPa: float = 1.6          # 满量程（表压）
    T_medium_max: float = 200.0     # 介质温度上限 °C
    rated_depth_m: float = 100.0    # 外壳防水/耐压额定（按清水深度标的，IP68）
    accuracy: float = 1.6           # 精度等级（%FS）
    wetted: str = "SUS316"          # 接液材料
    mass_kg: float = 0.9
    volume_L: float = 0.35          # 整体排开体积
    T_min: float = -20.0


@dataclass
class Env:
    fluid: object = "water"         # 键名或密度（0.7 / 700 都行）
    T: float = 20.0                 # 液体温度 °C
    depth_m: float = 10.0
    p_surface: float = P_ATM        # 液面上方绝对压力 Pa（开口=大气；密闭罐可加压）
    open_tank: bool = True


COMPAT = {  # (接液材料, 液体) → (等级, 说明)
    ("Brass", "seawater"): (WARN, "黄铜在海水里会脱锌腐蚀，长期用选 316"),
    ("SUS304", "seawater"): (WARN, "304 在海水里有点蚀风险，长期浸泡选 316L 或钛"),
    ("Brass", "pag10"): (OK, ""),
}


def pressure_at(env, depth=None):
    f = get_fluid(env.fluid)
    h = env.depth_m if depth is None else depth
    rho = float(f["rho"](env.T))
    p_abs = env.p_surface + rho * G * h
    return p_abs, p_abs - P_ATM, rho


def check_gauge(g: PressureGauge, env: Env):
    f = get_fluid(env.fluid)
    p_abs, p_gauge, rho = pressure_at(env)
    out = []
    add = lambda lv, t: out.append((lv, t))
    out_vals = dict(rho=rho, p_abs=p_abs, p_gauge=p_gauge, fluid=f["name"])

    # 1 液体本身在这个温度/压力下是不是液体
    Tb_surf = f["tsat"](env.p_surface)
    Tb_here = f["tsat"](p_abs)
    out_vals.update(Tb_surface=Tb_surf, Tb_here=Tb_here)
    if env.T >= Tb_here:
        add(FAIL, f"{f['name']}在 {env.depth_m:g} m 处（绝压 {p_abs/1e6:.3f} MPa）沸点只有 {Tb_here:.0f}°C，"
                  f"{env.T:g}°C 的液体在这里会沸腾/汽化，这个工况在现实里不存在")
    elif env.T >= Tb_surf:
        need = None
        if f["name"] == "清水" or "水" in f["name"]:
            from .fluids import water_psat
            need = water_psat(env.T)
        add(FAIL if env.open_tank else WARN,
            f"液面压力 {env.p_surface/1e3:.0f} kPa 时沸点 {Tb_surf:.0f}°C，{env.T:g}°C 的液体在液面会剧烈沸腾"
            + (f"；要让它保持液态，液面至少要加压到 {need/1e6:.2f} MPa（绝压），必须用密闭压力容器" if need else ""))
    if env.T <= f["freeze"]:
        add(FAIL, f"{env.T:g}°C 低于凝固点 {f['freeze']:g}°C，已经结冰/凝固")

    # 2 测量范围
    frac = p_gauge / (g.range_MPa * 1e6)
    out_vals["frac"] = frac
    if frac > 1.0:
        add(FAIL, f"表压 {p_gauge/1e6:.3f} MPa 超过量程 {g.range_MPa:g} MPa（{frac*100:.0f}%），指针打到头，会损坏弹簧管")
    elif frac > 0.75:
        add(WARN, f"读数在量程的 {frac*100:.0f}%，超过 3/4，寿命和精度变差，建议换大一档量程")
    elif frac < 1 / 3:
        add(WARN, f"读数只占量程的 {frac*100:.0f}%（<1/3），相对误差大；精度 {g.accuracy}% 级 → ±{g.accuracy/100*g.range_MPa*1e3:.0f} kPa，"
                  f"约等于 ±{g.accuracy/100*g.range_MPa*1e6/(rho*G):.1f} m 液柱")
    else:
        add(OK, f"读数 {p_gauge/1e6:.3f} MPa，在量程的 {frac*100:.0f}%（1/3~3/4 推荐区）")

    # 3 外壳耐压（额定按清水深度标）
    rated_p = 1000 * G * g.rated_depth_m
    ext = p_gauge  # 外壳内外压差（内部为大气）
    max_depth_here = g.rated_depth_m * 1000 / rho
    out_vals.update(rated_p=rated_p, max_depth_here=max_depth_here, eq_water_depth=ext / (1000 * G))
    if ext > rated_p:
        add(FAIL, f"外壳额定 {g.rated_depth_m:g} m 水深 = {rated_p/1e6:.3f} MPa，这里外压 {ext/1e6:.3f} MPa，会进水/压坏")
    else:
        add(OK, f"外壳承受 {ext/1e6:.3f} MPa（相当于 {ext/(1000*G):.1f} m 清水），额定 {g.rated_depth_m:g} m 水深；"
                f"在{f['name']}里最深可放 {max_depth_here:.0f} m")

    # 4 温度
    if env.T > g.T_medium_max:
        add(FAIL, f"液体 {env.T:g}°C 超过压力表耐温 {g.T_medium_max:g}°C")
    elif env.T > 0.9 * g.T_medium_max:
        add(WARN, f"液体 {env.T:g}°C 接近耐温上限 {g.T_medium_max:g}°C，弹簧管热漂移大（约 0.04%/°C）")
    else:
        add(OK, f"温度 {env.T:g}°C ≤ 耐温 {g.T_medium_max:g}°C")
    # 热漂移：典型 ±0.04%FS/°C （相对 20°C）
    drift = 0.0004 * abs(env.T - 20) * g.range_MPa * 1e6
    out_vals["thermal_drift"] = drift

    # 5 浮力
    Fb = rho * g.volume_L / 1000 * G
    W = g.mass_kg * G
    out_vals.update(Fb=Fb, W=W)
    if Fb > W:
        add(WARN, f"表重 {W:.1f} N < 浮力 {Fb:.1f} N，放下去会漂起来，需要配重或固定（多 {Fb-W:.1f} N）")
    else:
        add(OK, f"会下沉（重 {W:.1f} N，浮力 {Fb:.1f} N），吊绳受力 {W-Fb:.1f} N")

    # 6 材料
    key = env.fluid if isinstance(env.fluid, str) else ""
    c = COMPAT.get((g.wetted, key))
    if c and c[1]:
        add(c[0], c[1])
    return out, out_vals
