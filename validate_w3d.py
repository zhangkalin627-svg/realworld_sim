"""v0.10 三维装配世界检验 (validate.py 末尾会调用): MuJoCo 刚体 + 梁体检 vs 力学理论"""
import numpy as np
from rw.world3d.scene import Scene
from rw.world3d.presets import beam_bridge, harmonic_stack, ramp, table_cart
from rw.world3d.physics import simulate, check, float_check

G = 9.81


def run():
    print("\n===== 12) v0.10 三维装配世界 (MuJoCo 刚体 + 梁体检) =====")
    # a) 自由落体
    s = Scene(); s.add("ball", "sphere", [100], "Q235", pos=[0, 0, 1050])
    r = simulate(s, tmax=1.0)
    t, z = r.times, r.frames[:, 0, 2]
    m = (t > 0.05) & (t < 0.4)
    a = np.polyfit(t[m], z[m], 2)[0] * 2
    print(f"[12a] 自由落体加速度: 仿真 {-a:.3f} m/s²（理论 9.81），偏差 {100*(-a/G-1):.2f}%")
    assert abs(-a / G - 1) < 0.01

    # b) 钢板桥：弯矩、挠度、支反力 = 材料力学
    s = beam_bridge(); r = simulate(s, tmax=3); c = check(r)
    b = [x for x in c["beams"] if x["name"] == "钢板"][0]
    pl, ld = s.get("钢板"), s.get("重物")
    w = pl.mass * G / 1200.0; P = ld.mass * G; L = b["span"]
    R_ = (w * 1200 + P) / 2
    M_th = R_ * 500 - w * 600 ** 2 / 2 - P / 2 * 100         # 中点弯矩：重物接触面 ±100 mm 两条线载荷
    EI = 206e3 * 200 * 30 ** 3 / 12
    d_th = 5 * w * L ** 4 / (384 * EI) - (w * 100 ** 2 / 2) * L ** 2 / (8 * EI) + 2 * (P / 2) * 400 * (3 * L ** 2 - 4 * 400 ** 2) / (48 * EI)
    print(f"[12b] 钢板桥: 跨距 {L:.0f} mm；Mmax 仿真 {b['Mmax']/1e3:.2f} vs 理论 {M_th/1e3:.2f} N·m ({100*(b['Mmax']/M_th-1):+.2f}%)；"
          f"挠度 {b['defl']:.4f} vs {d_th:.4f} mm ({100*(b['defl']/d_th-1):+.2f}%)")
    assert abs(b["Mmax"] / M_th - 1) < 0.01 and abs(b["defl"] / d_th - 1) < 0.02
    rs = sorted(x["F"] for x in c["bearing"] if "钢板" in x["pair"] and "墩" in x["pair"])
    assert abs(rs[0] - R_) / R_ < 0.01 and abs(rs[1] - R_) / R_ < 0.01   # 两个墩的支反力都 = (自重+重物)/2
    # c) 质量守恒：地面支反力 = 总重
    s = Scene(); s.add("a", "box", [500, 500, 100], "Q235", pos=[0, 0, 50]); s.add("b", "cyl", [200, 300], "Cu", pos=[0, 0, 250])
    r = simulate(s, tmax=2); c = check(r)
    print(f"[12c] 地面支反力 {c['floor']['total_N']:.2f} N vs 总重 {s.total_mass()[0]*G:.2f} N")
    assert abs(c["floor"]["total_N"] / (s.total_mass()[0] * G) - 1) < 2e-3

    # d) 斜面：库仑摩擦临界角 tanθ=μ（钢-混凝土取 μ=0.5 → 26.57°）与滑动加速度
    def status(th, mat="Q235", w=100, h=100):
        r = simulate(ramp(th, mat, w, h), tmax=3.0); return check(r)["stability"][0], r
    st25, _ = status(26); st28, r28 = status(27)
    print(f"[12d] 钢块/混凝土斜面 μ=0.5（临界 26.57°）: 26° {st25['status']}, 27° {st28['status']}")
    assert st25["status"] == "稳定" and st28["status"] != "稳定"
    th = 32.0; _, r = status(th)
    tt = r.times; pos = r.frames[:, 0, :3] * 1000
    sl = (pos[:, 0] - pos[0, 0]) * np.cos(np.radians(th)) - (pos[:, 2] - pos[0, 2]) * np.sin(np.radians(th))
    mk = (tt > 0.3) & (tt < 1.0)
    a_sim = np.polyfit(tt[mk], sl[mk] / 1000, 2)[0] * 2
    a_th = G * (np.sin(np.radians(th)) - 0.5 * np.cos(np.radians(th)))
    print(f"[12d] 32° 滑动加速度: 仿真 {a_sim:.3f} vs 理论 g(sinθ-μcosθ)={a_th:.3f} m/s² ({100*(a_sim/a_th-1):+.1f}%)")
    assert abs(a_sim / a_th - 1) < 0.08
    # e) 倾倒：w/h=1/3 的橡胶块，临界 18.43°
    s17, _ = status(17.5, "rubber", 100, 300); s19, _ = status(19.5, "rubber", 100, 300)
    print(f"[12e] 100×300 橡胶块（临界 arctan(1/3)=18.43°）: 17.5° {s17['status']}, 19.5° {s19['status']}")
    assert s17["status"] == "稳定" and s19["status"] == "倾倒/滑落"
    # f) 叠积木：调和级数外伸，临界系数 1.0
    r1 = check(simulate(harmonic_stack(0.95), tmax=4)); r2 = check(simulate(harmonic_stack(1.05), tmax=4))
    print(f"[12f] 4 块叠积木外伸（理论临界 1.0）: ×0.95 {'稳' if not r1['unstable'] else '倒'}, ×1.05 {'稳' if not r2['unstable'] else '倒'}")
    assert not r1["unstable"] and r2["unstable"]
    # g) 台架偏载：前后腿反力 = 力矩平衡
    s = table_cart(); r = simulate(s, tmax=3); c = check(r)
    front = sum(x["F"] for x in c["bearing"] if x["pair"].startswith("地面↔腿1"))
    Wt = (s.total_mass()[0] - s.get("偏载").mass) * G; Pl = s.get("偏载").mass * G
    front_th = Wt / 2 + Pl * (250 + 275) / 550
    print(f"[12g] 台架前排两腿反力 仿真 {front:.1f} vs 力矩平衡 {front_th:.1f} N ({100*(front/front_th-1):+.2f}%)")
    assert abs(front / front_th - 1) < 0.01
    # h) 浮沉
    sf = Scene(); sf.fluid = ("water", 400); sf.add("w", "box", [100, 100, 100], "wood"); sf.add("s", "box", [100, 100, 100], "Q235")
    f = float_check(sf)
    print(f"[12h] 水中: 松木浮(吃水 {f[0]['draft']*100:.0f}% ≈ 密度比 {500/998.2*100:.0f}%), 钢沉={not f[1]['floats']}")
    assert f[0]["floats"] and not f[1]["floats"] and abs(f[0]["draft"] - 500 / 998.2) < 0.01
    print("[12] 三维装配世界 全部通过")


if __name__ == "__main__":
    run()
