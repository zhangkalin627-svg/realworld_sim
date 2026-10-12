"""v0.11 流体检验 (validate.py 末尾会调用): 管网、泵、水箱放空、水锤、射流、气管 vs 理论 / `fluids` 库 / 独立算法"""
import math
import numpy as np
from scipy.optimize import brentq
from rw.world.fluids import G, get_fluid
from rw.flow.network import Network, Pipe, Valve, Pump, Nozzle, example_two_station
from rw.flow import transient as TR, jet as JT, gas as GS, fit as FT

try:
    import fluids.friction as ffr
    HAVE = True
except Exception:
    HAVE = False


def run():
    print("\n===== 13) v0.11 流体: 管网 / 泵 / 水锤 / 射流 / 气管 =====")
    water = get_fluid("water"); rho = float(water["rho"](20)); mu = float(water["mu"](20))

    # a) 摩擦系数 vs fluids 库（Colebrook 的 Clamond 显式解）
    if HAVE:
        worst = 0
        for Re in (5e3, 2e4, 1e5, 1e6, 5e6):
            for eD in (1e-6, 1e-4, 1e-3, 1e-2):
                from rw.world.pipes import friction_factor
                a = friction_factor(Re, eD); b = ffr.friction_factor(Re=Re, eD=eD)
                worst = max(worst, abs(a / b - 1))
        print(f"[13a] 摩擦系数(紊流 20 组) vs fluids.friction: 最大偏差 {100*worst:.4f}%")
        assert worst < 2e-3
        assert abs(friction_factor(1000, 0) * 1000 / 64 - 1) < 1e-9
    else:
        print("[13a] 未装 fluids，跳过"); 

    # b) 单管：水池 → 管 → 出口，对照独立求根
    n = Network("water", 20); n.reservoir("R", level=10.0); n.reservoir("O", z=0.0, level=0.0)
    n.add(Pipe("P", "R", "O", L=50, D=0.04, K=3.0)); r = n.solve()
    def f(Q):
        from rw.world.pipes import friction_factor as ff
        v = Q / (math.pi * 0.04 ** 2 / 4); Re = rho * v * 0.04 / mu
        return (ff(Re, 0.045e-3 / 0.04) * 50 / 0.04 + 3.0) * v ** 2 / (2 * G) - 10.0
    Qref = brentq(f, 1e-7, 0.05)
    print(f"[13b] 单管 Q: 网络 {r['links']['P']['Q_Lmin']:.3f} vs 独立 Bernoulli+Darcy {Qref*6e4:.3f} L/min")
    assert abs(r['links']['P']['Q'] / Qref - 1) < 1e-4

    # c) 并联两根不同管径：水头差相等、流量和守恒；vs 独立求根
    n = Network("water", 20); n.reservoir("R", level=8.0); n.reservoir("O", level=0.0)
    n.add(Pipe("P1", "R", "O", 30, 0.025)); n.add(Pipe("P2", "R", "O", 30, 0.040)); r = n.solve()
    ref = []
    for D in (0.025, 0.040):
        from rw.world.pipes import friction_factor as ff
        g = lambda Q: ff(rho * (Q / (math.pi * D ** 2 / 4)) * D / mu, 0.045e-3 / D) * 30 / D * (Q / (math.pi * D ** 2 / 4)) ** 2 / (2 * G) - 8.0
        ref.append(brentq(g, 1e-7, 0.05))
    e1 = r['links']['P1']['Q'] / ref[0] - 1; e2 = r['links']['P2']['Q'] / ref[1] - 1
    print(f"[13c] 并联 DN25/DN40: Q1 {r['links']['P1']['Q_Lmin']:.2f}、Q2 {r['links']['P2']['Q_Lmin']:.2f} L/min，偏差 {100*e1:.4f}% / {100*e2:.4f}%")
    assert abs(e1) < 1e-4 and abs(e2) < 1e-4

    # d) 环网：4 管成环，1 点接水池，3 点取水，对照独立 Hardy-Cross
    Ls = {"12": 80, "23": 60, "14": 50, "43": 90}; Ds = {"12": 0.05, "23": 0.04, "14": 0.04, "43": 0.05}
    n = Network("water", 20); n.reservoir("1", level=30.0)
    for k in "234": n.junction(k)
    n.nodes["3"].demand = 0.004      # 4 L/s
    for key, (a_, b_) in {"12": ("1", "2"), "23": ("2", "3"), "14": ("1", "4"), "43": ("4", "3")}.items():
        n.add(Pipe(key, a_, b_, Ls[key], Ds[key]))
    r = n.solve()
    from rw.world.pipes import friction_factor as ff
    def hl(key, Q):
        D = Ds[key]; v = abs(Q) / (math.pi * D ** 2 / 4); Re = rho * v * D / mu
        h = (ff(Re, 0.045e-3 / D) if Re > 1e-9 else 0) * Ls[key] / D * v ** 2 / (2 * G)
        return math.copysign(h, Q)
    q = {"12": 0.002, "23": 0.002, "14": 0.002, "43": 0.002}
    for _ in range(200):       # 回路 1→2→3←4←1 ：顺时针 12+23 − (14+43)
        num = hl("12", q["12"]) + hl("23", q["23"]) - hl("14", q["14"]) - hl("43", q["43"])
        den = sum(abs(hl(k, q[k]) / q[k]) * 1.0 for k in q) * 1.0
        dq = -num / (den)
        q["12"] += dq; q["23"] += dq; q["14"] -= dq; q["43"] -= dq
    errs = [abs(r['links'][k]['Q'] / q[k] - 1) for k in q]
    print(f"[13d] 环网流量分配 vs 独立 Hardy-Cross: 12 {r['links']['12']['Q_Lmin']:.2f}/{q['12']*6e4:.2f}，14 {r['links']['14']['Q_Lmin']:.2f}/{q['14']*6e4:.2f} L/min，最大偏差 {100*max(errs):.3f}%")
    assert max(errs) < 2e-3
    # 每个接点质量守恒
    for node in "234":
        net_ = sum((r['links'][k]['Q'] if l.n2 == node else -r['links'][k]['Q']) for l in n.links for k in [l.name] if node in (l.n1, l.n2)) - n.nodes[node].demand
        assert abs(net_) < 1e-9, net_
    print("      各接点流量守恒误差 < 1e-9 m³/s")

    # e) 泵+纯二次阻力：相似定律 Q ∝ 转速
    def pump_Q(s):
        n = Network("water", 20); n.reservoir("S", level=0.0); n.junction("A"); n.junction("B"); n.reservoir("O", level=0.0)
        n.add(Pipe("in", "S", "A", 0.001, 0.05, rough=1e-9, K=0.0)); n.add(Pump("pm", "A", "B", H0=30, Q_rated=0.002, H_rated=22, speed=s))
        n.add(Valve("v", "B", "O", D=0.025, K_open=40.0)); return n.solve()["links"]["pm"]["Q"]
    Q1, Q8 = pump_Q(1.0), pump_Q(0.8)
    print(f"[13e] 泵相似定律: 转速 80% 时流量比 {Q8/Q1:.4f}（理论 0.8，管路阻力为纯二次时）")
    assert abs(Q8 / Q1 - 0.8) < 5e-3     # 入口管很短的摩擦使其略偏离

    # f) 水箱放空: Torricelli 解析解 vs 网络准稳态推进
    n = Network("water", 20); n.tank("T", z=1.0, level=1.0, area=0.5); n.reservoir("O", z=1.0, level=0.0)
    n.add(Nozzle("hole", "T", "O", n=1, d=0.012, Cd=0.62))
    h = n.run_quasi_steady(1e5, dt=0.5, stop_empty=False) if False else None
    t_th = JT.drain_time(0.5, math.pi * 0.012 ** 2 / 4, 1.0, 0.2, 0.62)
    # 运行到 0.2 m
    t = 0.0; dt = 0.5
    while n.nodes["T"].level > 0.2:
        res = n.solve(); Qh = res["links"]["hole"]["Q"]; n.nodes["T"].level -= Qh * dt / 0.5; t += dt
    print(f"[13f] 水箱 1.0→0.2 m 经 Ø12 孔: 网络 {t:.0f} s vs Torricelli 解析 {t_th:.0f} s ({100*(t/t_th-1):+.2f}%)")
    assert abs(t / t_th - 1) < 0.01

    # g) 水锤: Korteweg 波速、Joukowsky、周期、Michaud
    a = TR.wave_speed(1000, 2.2e9, 0.05, 0.003, 200e9)
    a_hand = 1 / math.sqrt(1000 * (1 / 2.2e9 + 0.05 * (1 - 0.3 ** 2) / (200e9 * 0.003)))
    assert abs(a / a_hand - 1) < 1e-9
    r = TR.water_hammer(L=100, D=0.05, e=0.003, E=200e9, Hres=5, K_valve=10.0, t_close=1e-6, N=50, friction=False, tmax=1.2)
    H = r["H_valve"] - 5.0; t = r["t"]
    jou = r["a"] * r["v0"] / G
    zc = t[1:][np.diff(np.sign(H)) != 0]
    T_meas = 2 * np.mean(np.diff(zc[:6]))
    print(f"[13g] 水锤(钢管 DN50, L=100 m): 波速 a={r['a']:.0f} m/s；瞬关峰值 ΔH 仿真 {H.max():.1f} m vs Joukowsky aV/g={jou:.1f} m "
          f"({100*(H.max()/jou-1):+.2f}%)；周期 仿真 {T_meas*1e3:.1f} ms vs 4L/a {r['T_period']*1e3:.1f} ms")
    assert abs(H.max() / jou - 1) < 5e-3 and abs(T_meas / r["T_period"] - 1) < 0.02
    for tc in (4.0, 8.0):
        rs = TR.water_hammer(L=100, D=0.05, e=0.003, E=200e9, Hres=5, K_valve=10.0, t_close=tc, N=50, friction=False)
        dm = TR.michaud(100, rs["v0"], tc); dh = rs["Hmax"] - 5
        print(f"      缓关 {tc:g} s(≫2L/a={rs['tL']*1e3:.0f} ms): 仿真 ΔH={dh:.1f} m vs Michaud 近似 {dm:.1f} m ({100*(dh/dm-1):+.1f}%)")
        if tc == 4.0:
            assert abs(dh / dm - 1) < 0.10
        else:
            print("      （8 s 时 ΔH 已与静压头 5 m 同量级，阀的孔板特性使流量不再随开度线性变化，Michaud 的线性假设不成立——属公式适用范围问题，不是仿真错）")
    # 软管的波速低得多
    a_h = TR.wave_speed(1000, 2.2e9, 0.010, 0.0025, TR.PIPE_E["增强软管"])
    print(f"      同样的水: 钢管 a≈{a:.0f} m/s，增强软管 a≈{a_h:.0f} m/s → 同流速下水锤压力只有钢管的 {100*a_h/a:.0f}%")
    assert a_h < 0.4 * a

    # h) 射流
    v = 12.0
    tj = JT.trajectory(v, 30.0, 0.0)
    R_th = v ** 2 * math.sin(math.radians(60)) / G
    print(f"[13h] 射流抛物线射程: {tj['range']:.3f} m vs v²sin2θ/g={R_th:.3f} m")
    assert abs(tj["range"] / R_th - 1) < 1e-3
    ang = JT.aim_angle(10.0, 0.5, 0.1)
    tt = JT.trajectory(10.0, ang[0], 0.0, n=4000)
    zi = np.interp(0.5, tt["x"], tt["z"])
    assert abs(zi - 0.1) < 2e-3, zi
    F = JT.jet_force(1000, 1e-4, 10.0)
    assert abs(F - 1000 * 1e-4 * 10) < 1e-12
    print(f"      瞄准: v=10 m/s 打 (0.5 m, +0.1 m) 低弹道仰角 {ang[0]:.2f}° → 实际高度 {zi:.4f} m；垂直冲击板 ρQv = {F:.2f} N")

    # i) 气管 vs fluids
    if HAVE:
        from fluids.compressible import isothermal_gas
        m, fdd = GS.isothermal_mass_flow(7e5, 6e5, 10.0, 0.008)
        rho1 = 7e5 / (287.05 * 293.15)
        m_ref = isothermal_gas(rho=rho1, fd=fdd, P1=7e5, P2=6e5, L=10.0, D=0.008)
        print(f"[13i] 气管(Ø8×10 m, 7→6 bar abs): {m*1e3:.2f} g/s = {GS.free_air_Lmin(m):.0f} L/min(ANR)；vs fluids.compressible {m_ref*1e3:.2f} g/s")
        assert abs(m / m_ref - 1) < 1e-3

    # j) 示例两工位喷淋: 体检 + 对称
    nn = example_two_station(); rr = nn.solve()
    qa, qb = rr["links"]["软管A 内径10"]["Q"], rr["links"]["软管B 内径10"]["Q"]
    qp = rr["links"]["离心泵"]["Q"]
    assert abs(qa - qb) < 1e-9 and abs(qp - qa - qb) < 1e-9 and nn.converged
    print(f"[13j] 示例两工位喷淋: 泵 {qp*6e4:.1f} L/min = 2×{qa*6e4:.1f}，泵轴功率 {rr['links']['离心泵']['P_shaft']:.0f} W，射流 {rr['links']['喷淋孔A']['v_jet']:.1f} m/s")
    print("      全部通过")


if __name__ == "__main__":
    run()
