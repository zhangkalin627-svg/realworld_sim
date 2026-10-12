"""通用示例电路（参数都是教科书/通用值，不对应任何具体产品）。

每个示例返回 Example：电路 + 推荐的分析 + 示波器预设 + 理论值说明。
"""
import math
from dataclasses import dataclass, field
from .spice import Circuit, opamp_subckt


@dataclass
class Example:
    key: str
    name: str
    circuit: Circuit
    tran: dict                      # Circuit.tran(**tran)
    scope: dict                     # 通道/时基/触发预设
    theory: str = ""
    ac: dict = None                 # 可选：交流分析参数
    values: dict = field(default_factory=dict)  # 理论值（validate.py 用）


def rc_square(R=1e3, C=1e-6, f=50.0, V=1.0):
    tau = R * C
    c = Circuit("RC 方波响应")
    c.V("in", "in", "0", pulse=(0, V, 0, 1e-9, 1e-9, 0.5 / f - 1e-9, 1 / f))
    c.R("1", "in", "out", R)
    c.C("1", "out", "0", C)
    return Example("rc", "RC 方波响应", c,
                   tran=dict(step=tau / 200, stop=3 / f, tmax=tau / 200),
                   scope=dict(ch=[("in", 0.2, -0.4), ("out", 0.2, -0.4)], tdiv=2e-3, hpos=4e-3,
                              trig=(1, "rise", V / 2)),
                   theory=f"τ = RC = {tau*1e3:.3g} ms；63.2% 点在 t=τ；上升时间(10–90%) = τ·ln9 ≈ 2.197τ = {2.197*tau*1e3:.3g} ms",
                   values=dict(tau=tau, tr=tau * math.log(9), V=V, f=f))


def rlc_series(R=10.0, L=1e-3, C=1e-6):
    f0 = 1 / (2 * math.pi * math.sqrt(L * C))
    zeta = R / 2 * math.sqrt(C / L)
    fd = f0 * math.sqrt(1 - zeta ** 2) if zeta < 1 else None
    c = Circuit("RLC 串联谐振")
    c.V("in", "in", "0", ac=1, pulse=(0, 1, 0, 1e-9, 1e-9, 2.5e-3, 5e-3))
    c.L("1", "in", "a", L)
    c.C("1", "a", "b", C)
    c.R("1", "b", "0", R)
    return Example("rlc", "RLC 串联谐振", c,
                   tran=dict(step=2e-7, stop=10e-3, tmax=2e-7),
                   ac=dict(fstart=f0 / 20, fstop=f0 * 20, points=2000),
                   scope=dict(ch=[("in", 0.5, -1.5), ("a-b", 0.5, 0), ("b", 0.2, 1.0)], tdiv=0.2e-3, hpos=0.8e-3,
                              trig=(1, "rise", 0.5)),
                   theory=f"f0 = 1/(2π√LC) = {f0:.1f} Hz；ζ = (R/2)√(C/L) = {zeta:.3f}；"
                          f"方波激励下电容电压按阻尼振荡频率 fd = f0√(1-ζ²) = {fd:.1f} Hz 振铃（CH2 = a-b 为电容两端差分）",
                   values=dict(f0=f0, fd=fd, zeta=zeta, Q=1 / R * math.sqrt(L / C)))


def bridge_rectifier(Vpk=17.0, f=50.0, C=2200e-6, Rl=100.0, model="D1N4007"):
    c = Circuit("桥式整流 + 电容滤波")
    c.V("ac", "ac1", "ac2", sin=(0, Vpk, f))
    c.R("leak", "ac2", "0", 1e7)          # 给浮地的变压器副边一个直流参考
    c.D("1", "ac1", "out", model); c.D("2", "ac2", "out", model)
    c.D("3", "0", "ac1", model); c.D("4", "0", "ac2", model)
    c.C("f", "out", "0", C)
    c.Vs = c.V("sense", "out", "load", dc=0)  # 0 V 电流探头
    c.R("l", "load", "0", Rl)
    Vdc = Vpk - 2 * 0.85
    I = Vdc / Rl
    return Example("rect", "桥式整流 + 电容滤波", c,
                   tran=dict(step=10e-6, stop=0.4, tmax=10e-6),
                   scope=dict(ch=[("ac1-ac2", 5.0, 0), ("out", 0.2, -14.6), ("i(vsense)", 0.05, -0.3)],
                              tdiv=5e-3, hpos=0.0, trig=(1, "rise", 0.0)),
                   theory=f"纹波近似 ΔV ≈ I/(2fC)（假设电容整个半周都在放电，偏大）≈ {I/(2*f*C):.3f} V（I≈{I:.3f} A）",
                   values=dict(f=f, C=C, Rl=Rl))


def opamp_astable(R=10e3, C=0.1e-6, R1=10e3, R2=10e3, vsat=13.0):
    beta = R1 / (R1 + R2)
    T = 2 * R * C * math.log((1 + beta) / (1 - beta))
    c = Circuit("运放方波振荡器（施密特 + RC）")
    c.subckt(opamp_subckt("OPAMP", gain=2e5, gbw=10e6, vsat=vsat))
    c.X("1", ["inp", "inn", "out"], "OPAMP")
    c.R("f", "out", "inn", R)
    c.C("t", "inn", "0", C, ic=-1.0)
    c.R("1", "inp", "0", R1)
    c.R("2", "out", "inp", R2)
    return Example("astable", "运放方波振荡器", c,
                   tran=dict(step=T / 2000, stop=12 * T, tmax=T / 2000, uic=True),
                   scope=dict(ch=[("out", 10.0, 20.0), ("inn", 5.0, -10.0), ("inp", 5.0, -10.0)], tdiv=T / 2.5, hpos=0,
                              trig=(1, "rise", 0.0)),
                   theory=f"T = 2RC·ln((1+β)/(1-β))，β=R1/(R1+R2)={beta:.2f} → f = {1/T:.1f} Hz（理想运放、对称饱和 ±{vsat:g} V）",
                   values=dict(f=1 / T, vsat=vsat))


def mosfet_halfbridge(Vbus=48.0, f=40e3, dead=0.5e-6, Lr=20e-6, Cr=1e-6, Rl=0.5, Rg=4.7, Vg=12.0):
    """感应加热常见拓扑的通用版：半桥 + 串联谐振负载（线圈 Lr、谐振电容 Cr、等效负载电阻 Rl）。
    参数是通用演示值，不是任何具体电源或用户设备的参数。"""
    T = 1 / f
    f0 = 1 / (2 * math.pi * math.sqrt(Lr * Cr))
    c = Circuit("MOSFET 半桥 + 串联谐振负载")
    c.V("bus", "bus", "0", dc=Vbus)
    c.C("bus", "bus", "0", 10e-6)
    on = T / 2 - dead
    c.V("gh", "gh_drv", "sw", pulse=(0, Vg, dead, 20e-9, 20e-9, on - 20e-9, T))       # 浮地驱动（自举/隔离驱动的理想化）
    c.V("gl", "gl_drv", "0", pulse=(0, Vg, T / 2 + dead, 20e-9, 20e-9, on - 20e-9, T))
    c.R("gh", "gh_drv", "gh", Rg); c.R("gl", "gl_drv", "gl", Rg)
    c.M("1", "bus", "gh", "sw", "NPWR")
    c.M("2", "sw", "gl", "0", "NPWR")
    c.V("coil", "sw", "a", dc=0)  # 0 V 电流探头：线圈电流
    c.L("r", "a", "b", Lr)
    c.R("l", "b", "c", Rl)
    c.C("r", "c", "0", Cr)
    return Example("halfbridge", "MOSFET 半桥 + 串联谐振（感应加热拓扑）", c,
                   tran=dict(step=10e-9, stop=1.0e-3, tmax=10e-9),
                   scope=dict(ch=[("gl", 5.0, -15.0), ("gh-sw", 5.0, -5.0), ("sw", 20.0, 20.0), ("i(vcoil)", 20.0, 0.0)],
                              tdiv=5e-6, hpos=0.0, trig=(3, "rise", Vbus / 2)),
                   theory=f"谐振频率 f0 = {f0/1e3:.1f} kHz，驱动 {f/1e3:.0f} kHz（高于谐振 → 感性负载，便于零电压开通）；"
                          f"死区 {dead*1e6:.2f} µs。功率管为通用 VDMOS 模型。",
                   values=dict(f0=f0, f=f, Vbus=Vbus))


EXAMPLES = {
    "rc": rc_square, "rlc": rlc_series, "rect": bridge_rectifier, "astable": opamp_astable, "halfbridge": mosfet_halfbridge,
}
NAMES = {"rc": "RC 方波响应", "rlc": "RLC 串联谐振", "rect": "桥式整流 + 电容滤波", "astable": "运放方波振荡器",
         "halfbridge": "MOSFET 半桥 + 串联谐振（感应加热拓扑）"}


def get(key, **kw):
    return EXAMPLES[key](**kw)
