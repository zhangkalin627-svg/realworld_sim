"""v0.10 三维装配世界检验 (validate.py 末尾会调用): MuJoCo 刚体 + 梁体检 vs 力学理论"""
import numpy as np
from rw.world3d.scene import Scene
from rw.world3d.presets import beam_bridge, harmonic_stack, ramp, table_cart, ibeam_bridge, wall_bracket, bearing_rotor, glass_drop, weld_bracket
from rw.world3d.fracture import break_and_rerun
from rw.world3d import profile as _prof
from rw.world3d import joints as _jnt
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
    # ---- v0.12 ----
    # i) 型材：理论重量 vs GB、截面特性 vs 解析、工字钢梁挠度 vs 材料力学
    worst = 0.0
    for table, kind in [(_prof.H_BEAMS, "ibeam"), (_prof.CHANNELS, "channel"), (_prof.ANGLES, "angle"), (_prof.SQ_TUBES, "sqtube")]:
        for label, spec in table.items():
            dims = list(spec[:-1]) + [1000]
            worst = max(worst, abs(_prof.kg_per_m(kind, dims) / spec[-1] - 1))
    print(f"[12i] 26 种国标型材理论重量: 尖角近似 vs GB 最大偏差 {100*worst:.1f}%（忽略圆角偏轻）")
    assert worst < 0.06
    I_sq = _prof.section_moments("sqtube", (50, 50, 3, 1000))["Iy"]; I_sq_th = (50 ** 4 - 44 ** 4) / 12
    I_ang = _prof.section_moments("angle", (50, 50, 5, 1000))["Iy"]
    print(f"[12i] 截面特性: 方管50×3 I={I_sq:.0f} vs 解析 {I_sq_th:.0f} mm⁴ ({100*(I_sq/I_sq_th-1):+.3f}%)；角钢50×5 I={I_ang/1e4:.2f} cm⁴（手册 11.2，尖角近似）")
    assert abs(I_sq / I_sq_th - 1) < 1e-6 and abs(I_ang / 11.21e4 - 1) < 0.02
    s = ibeam_bridge(); r = simulate(s, tmax=3); c = check(r)
    b = [x for x in c["beams"] if x["name"] == "工字钢"][0]
    pl, ld = s.get("工字钢"), s.get("钢块")
    I_ind = (200 * 200 ** 3 - (200 - 8) * (200 - 24) ** 3) / 12      # 独立手算强轴 I
    w_pl = pl.mass * G / 2400.0; P_ld = ld.mass * G; Lb = b["span"]
    d_th = P_ld * Lb ** 3 / (48 * 206e3 * I_ind) + 5 * w_pl * Lb ** 4 / (384 * 206e3 * I_ind)
    print(f"[12i] 工字钢梁: I 采用 {b['section']}={b['I'] if 'I' in b else I_ind:.3g} mm⁴(手算 {I_ind:.3g})；跨中挠度 仿真 {b['defl']:.3f} vs 材料力学 {d_th:.3f} mm ({100*(b['defl']/d_th-1):+.1f}%)")
    assert abs(b["defl"] / d_th - 1) < 0.06
    # j) 螺栓组：壁挂支架静力 → 手算螺栓拉力/剪力
    s = wall_bracket(); r = simulate(s, tmax=2.5); c = check(r)
    j = c["bolts"][0]
    W = (s.get("背板").mass + s.get("悬臂").mass + s.get("钢块").mass) * G
    M_y = G * (s.get("背板").mass * 10 + s.get("悬臂").mass * 170 + s.get("钢块").mass * 270)   # N·mm 绕 y 轴
    T_top = M_y * 70 / (4 * 70 ** 2)                                # 顶排螺栓拉力(弹性分配)
    V_each = W / 4
    tops = [x for x in j["bolts"] if x["v"] > 0]
    print(f"[12j] 螺栓组: 接头力 |F| 仿真 {np.linalg.norm(j['F']):.1f} vs 总重 {W:.1f} N；M_y {abs(j['M'][1])/1e3:.2f} vs 手算 {M_y/1e3:.2f} N·m；"
          f"顶排螺栓 T 仿真 {tops[0]['T']:.0f} vs {T_top:.0f} N，V {j['bolts'][0]['V']:.0f} vs {V_each:.0f} N；利用率 {j['worst']:.3f} ok={j['ok']}")
    assert abs(np.linalg.norm(j["F"]) / W - 1) < 0.02 and abs(abs(j["M"][1]) / M_y - 1) < 0.03
    assert abs(tops[0]["T"] / T_top - 1) < 0.03 and abs(j["bolts"][0]["V"] / V_each - 1) < 0.02
    # k) 轴承：静态悬挂反力 = 重量；3000 rpm 不平衡力 = m·e·ω²；L10 量级
    s0 = bearing_rotor(0); r0 = simulate(s0, tmax=3); c0 = check(r0)
    h0 = c0["hinges"][0]
    W_rot = (s0.get("轴").mass + s0.get("偏心块").mass) * G
    print(f"[12k] 轴承静态: 反力径向 {h0['F_radial']:.2f} N vs 转子总重 {W_rot:.2f} N ({100*(h0['F_radial']/W_rot-1):+.2f}%)")
    assert abs(h0["F_radial"] / W_rot - 1) < 0.01
    s = bearing_rotor(3000); r = simulate(s, tmax=1.5); c = check(r)
    h = c["hinges"][0]
    om = 3000 * 2 * np.pi / 60
    F_ub = s.get("偏心块").mass * 0.150 * om ** 2
    print(f"[12k] 轴承 3000rpm: 不平衡力 仿真 {h['F_unbalance']:.0f} vs m·e·ω²={F_ub:.0f} N ({100*(h['F_unbalance']/F_ub-1):+.2f}%)；"
          f"6206 L10 = {h['life']['L10h']:.0f} h (P={h['P']:.0f} N)")
    assert abs(h["F_unbalance"] / F_ub - 1) < 0.05
    L10_th = (19500 / h["P"]) ** 3 * 1e6 / (60 * 3000)
    assert abs(h["life"]["L10h"] / L10_th - 1) < 1e-9
    # l) 赫兹接触：钢球在钢面, 公式 vs 手算值; check() 报告与手算一致
    hh = _jnt.hertz_point(100, 10, 206e9, 0.3, 206e9, 0.3)
    a_th = (3 * 100 * 0.01 / (4 * 206e9 / 2 / (1 - 0.3 ** 2))) ** (1 / 3) * 1000      # 独立手算 a
    p0_th = 3 * 100 / (2 * np.pi * (a_th / 1000) ** 2) / 1e6
    print(f"[12l] 赫兹点接触: R10 钢球 100 N → a={hh['a']:.3f} mm (手算 {a_th:.3f})，p0={hh['p0']:.0f} MPa (手算 {p0_th:.0f})")
    assert abs(hh["a"] / a_th - 1) < 1e-9 and abs(hh["p0"] / p0_th - 1) < 0.01
    s = Scene(); s.add("球", "sphere", [20], "Q235", pos=[0, 0, 10]); s.add("块", "box", [100, 100, 100], "Cu", pos=[0, 0, 80])
    r = simulate(s, tmax=2); c = check(r)
    hz = [x for x in c["hertz"] if x["part"] == "球" and x["with_"] == "块"][0]
    F_blk = s.get("块").mass * G                       # 球-块接触只传铜块重量
    hh2 = _jnt.hertz_point(F_blk, 10, 206e9, 0.3, 117e9, 0.34)      # 钢球-紫铜 手算
    print(f"[12l] 铜块压钢球: F={hz['F']:.1f} N vs {F_blk:.1f} N；p0 报告 {hz['p0']:.0f} vs 手算 {hh2['p0']:.0f} MPa")
    assert abs(hz["F"] / F_blk - 1) < 0.01 and abs(hz["p0"] / hh2["p0"] - 1) < 0.01
    # m) 断裂判据：玻璃板 0.74 倍断裂载荷 → 正常；1.2 倍 → 断裂
    Wsec = 100 * 6 ** 2 / 6; F_break = 4 * 40.0 * Wsec / 360          # 跨360简支、中点集中力: M=FL/4=fu·W
    r1 = check(simulate(glass_drop(20), tmax=2.5)); r2 = check(simulate(glass_drop(80), tmax=2.5))
    b1 = [x for x in r1["beams"] if x["name"] == "玻璃板"][0]; b2 = [x for x in r2["beams"] if x["name"] == "玻璃板"][0]
    print(f"[12m] 断裂判据: 理论断裂载荷 {F_break/G:.1f} kg（点载荷近似，实际分布载荷略高）；20 kg → σ={b1['sigma']:.0f} MPa {b1['status']}，80 kg → σ={b2['sigma']:.0f} MPa {b2['status']}")
    assert b1["status"] == "正常" and b2["status"] == "断裂" and r2["broken"]
    # n) 焊缝群：悬臂焊墙，σ/τ vs 手算
    s = weld_bracket(); r = simulate(s, tmax=2.5); c = check(r)
    wd = c["bolts"][0]
    Wa = s.get("悬臂").mass * G; Wb = s.get("钢块").mass * G
    M1 = Wa * 150 + Wb * 250                      # N·mm 绕 y
    a_eff = 0.7 * 6; SL = 160.0; Iu = 2 * 80 * 10 ** 2
    sig_th = M1 * 10 / (a_eff * Iu); tau_th = (Wa + Wb) / (a_eff * SL)
    print(f"[12n] 焊缝群: σ_b 仿真 {wd['sig_b']:.2f} vs 手算 {sig_th:.2f} MPa，τ {wd['tau']:.2f} vs {tau_th:.2f} MPa，利用率 {wd['util']:.4f} ok={wd['ok']}")
    assert abs(wd["sig_b"] / sig_th - 1) < 0.02 and abs(wd["tau"] / tau_th - 1) < 0.02 and wd["ok"]
    # o) 断裂后果：玻璃板断裂 → 分裂再仿真 → 两段失去支撑下落
    s = glass_drop(80); r = simulate(s, tmax=2.5); c = check(r)
    out = break_and_rerun(r, c, tmax=3)
    assert out is not None
    s2, r2, c2 = out
    seg = [x for x in c2["stability"] if "玻璃板" in x["body"]]
    blk = [x for x in c2["stability"] if "钢块" in x["body"]][0]
    print(f"[12o] 断裂后果: 断成 {len([p for p in s2.parts if '玻璃板' in p.name])} 段；左/右段状态: "
          f"{[x['status'] for x in seg]}；钢块下落 { -blk['dz_mm']:.0f} mm（断后不再被托住）")
    assert any(x["status"] != "稳定" for x in seg) and blk["dz_mm"] < -50
    print("[12] 三维装配世界 全部通过")
    matdb_edge()


def matdb_edge():
    """[7z] matdb mix() 边界回归（issue 20261012-1928）。"""
    from rw.matdb import mixing as MX
    for el in ("O", "H", "N", "Rn"):                       # 单一气态元素不崩
        r = MX.mix({el: 50}); MX.report(r); assert r["density"] > 0
    for bad in ({}, {"Fe": 0}, {"Fe": -10, "Cu": 50}):     # 空/全零/负含量 → ValueError
        try:
            MX.mix(bad); assert False, bad
        except ValueError:
            pass
    w = MX.mix({"Fe": 50, "铁": 50, "Cu": 100})["wt"]      # 符号+中文名合并
    assert abs(w["Fe"] - 50) < 1e-9 and abs(w["Cu"] - 50) < 1e-9, w
    for ag in (1.99, 2.0):                                 # Fe-Ag 少量也不能说可熔炼
        r = MX.mix({"Fe": 100 - ag, "Ag": ag})
        assert "不互溶" in r["verdict_zh"] and r["warnings"], (ag, r["verdict_zh"])
    assert "可以熔炼" in MX.mix({"Fe": 98, "Cu": 1.5})["verdict_zh"]
    print("[7z] matdb mix 边界（单气体元素/空配比/负值/中英名合并/Fe-Ag 少量）通过")


if __name__ == "__main__":
    run()
