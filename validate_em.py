"""v0.9 电磁升级检验 (validate.py 末尾会调用): B-H 描述函数、一维非线性时域真值、二维非线性有限元、导磁体、三维涡流"""
import json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from rw.em.bh import STEEL45_BH, MU0
from rw.em import slab1d
from rw import materials as M


def run():
    print("\n===== 11) v0.9 非线性 B-H / 导磁体 / 三维涡流 =====")
    c = STEEL45_BH
    # a) 描述函数 = 时间积分的基波
    ph = np.linspace(0, 2 * np.pi, 20001)
    worst = 0
    for T in (20, 500, 740):
        for H0 in (1e3, 3e4, 1e5):
            B1 = np.trapezoid(c.B(H0 * np.sin(ph), T) * np.sin(ph), ph) / np.pi
            worst = max(worst, abs(B1 / c.B1(H0, T) - 1))
    print(f"[11a] 描述函数 B1(H0) 与逐点时间积分: 最大偏差 {worst*100:.3f}%")
    assert worst < 2e-3
    # b) 居里点
    assert abs(float(c.mu_eff(3e4, 800.0)) - 1.0) < 1e-6 and c.mu_eff(3e4, 700.0) < c.mu_eff(3e4, 20.0)
    print(f"[11b] μ_eff(30 kA/m): 20°C {float(c.mu_eff(3e4,20.)):.1f}, 700°C {float(c.mu_eff(3e4,700.)):.1f}, 800°C(>居里点) {float(c.mu_eff(3e4,800.)):.2f}")

    # c) 一维: 线性极限 = 解析解; 非线性: 谐波平衡 vs 时域真值
    class Lin:
        def __init__(s, m): s.mur = m
        def B(s, H, T): return MU0 * s.mur * np.asarray(H, float)
        def mu_eff(s, H, T): return np.full_like(np.asarray(H, float), s.mur)
    rho, f = 2.0e-7, 30e3
    for mur in (1, 50):
        Pa = slab1d.linear_power(2e4, f, mur, rho)
        P1 = slab1d.time_domain(Lin(mur), 20, rho, f, 2e4); P2 = slab1d.harmonic(Lin(mur), 20, rho, f, 2e4)[0]
        print(f"[11c] 线性 μr={mur}: 时域/解析 {P1/Pa:.4f}, 谐波/解析 {P2/Pa:.4f}")
        assert abs(P1 / Pa - 1) < 0.01 and abs(P2 / Pa - 1) < 0.01
    worst = 0
    for T, H0 in ((20, 5e3), (20, 8e4), (500, 2e4), (700, 2e4)):
        Pt = slab1d.time_domain(c, T, rho, f, H0); Ph = slab1d.harmonic(c, T, rho, f, H0)[0]
        worst = max(worst, abs(Ph / Pt - 1))
        print(f"[11c] 非线性 T={T}°C H0={H0/1e3:g} kA/m: 谐波平衡/时域真值 = {Ph/Pt:.4f}")
    assert worst < 0.03

    # d) 二维非线性有限元 vs 一维时域真值 (同一表面磁场), 并和线性模式的偏差比较 (几何效应相同)
    from rw.induction import InductionQuenchSim
    from rw.em import check
    r_nl = check.local_power_check(InductionQuenchSim(freq=30e3, power_kw=6, nonlinear=True))
    r_li = check.local_power_check(InductionQuenchSim(freq=30e3, power_kw=6, nonlinear=False))
    e_nl = r_nl["P_fem_MWm2"] / r_nl["P_1d_time_MWm2"] - 1
    e_li = r_li["P_fem_MWm2"] / r_li["P_1d_lin_MWm2"] - 1
    print(f"[11d] 二维有限元(淬火区中部,表面H0={r_nl['H0_kAm']:.0f} kA/m): 非线性 {r_nl['P_fem_MWm2']:.3f} vs 一维时域 {r_nl['P_1d_time_MWm2']:.3f} MW/m² ({e_nl*100:+.1f}%)；"
          f"线性模式对一维解析 {e_li*100:+.1f}% (两者共同的偏差是圆柱曲率/有限线圈的几何效应)")
    assert abs(e_nl) < 0.10 and abs(e_nl - e_li) < 0.04

    # e) 导磁体: 同功率下电流下降, 磁密不超过饱和
    from rw.em.conc import Concentrator
    I = {}
    for k in (None, "smc"):
        s = InductionQuenchSim(freq=30e3, power_kw=6, nonlinear=True, conc=Concentrator(k) if k else None)
        q, Ic = s.solve_em(np.full(s.tb.N, 25.0)); I[k] = Ic / np.sqrt(2)
        if k:
            Bm = float(s._B1_el[s.conc_el].max()); loss = s.conc_loss_W
    print(f"[11e] 同样 6 kW: 无导磁体 {I[None]:.0f} A → SMC 导磁体 {I['smc']:.0f} A ({(I['smc']/I[None]-1)*100:+.0f}%)，SMC 磁密峰值 {Bm:.2f} T (饱和 1.4 T)，自身损耗 {loss:.0f} W")
    assert I["smc"] < I[None] and Bm < 1.45 and loss > 0

    # f) 三维 (缓存结果: python3 -c "from rw.em.eddy3d import study_3d ..." 约 3–5 分钟)
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rw", "em", "eddy3d_results.json")
    if os.path.exists(fn):
        R = json.load(open(fn)); e0 = R["ecc0"]
        d = e0["P_3d"] / e0["P_2d"] - 1
        sp = np.array(e0["sectors"]); asym = sp.max() / sp.min() - 1
        print(f"[11f] 三维 vs 二维轴对称(同心, {e0['tets']} 四面体): P3D={e0['P_3d']:.2f} W, P2D={e0['P_2d']:.2f} W, 差 {d*100:+.2f}%；各方位功率不均匀度 {asym*100:.3f}% (应≈0)；规范假损耗 {e0['P_air']/e0['P_3d']*100:.2f}%")
        assert abs(d) < 0.05 and asym < 0.01
        if "sigma_sweep" in R:
            sw = R["sigma_sweep"]
            dev = max(abs(v / e0["P_3d"] - 1) for v in sw.values())
            print(f"[11f] 空气人为电导 σ=0.1/1/10 S/m: 功率变化 ≤ {dev*100:.2f}%")
            assert dev < 0.02
        e1 = R["ecc"]; s1 = np.array(e1["sectors"])
        print(f"[11f] 偏心 {e1['ecc_mm']:g} mm: 管壁方位功率最大/最小 {s1.max()/s1.min():.3f}，总功率 {100*(e1['P_3d']/e0['P_3d']-1):+.1f}%")
    else:
        print("[11f] 三维缓存结果不存在，跳过 (运行 study_3d 生成)")

    # g) 非圆截面涡流 (v0.14, gmsh 可选依赖): ①同几何空心管对照结构网格缓存 ②方坯物理检查。现场跑粗网格(≈1.5 万四面体/例, 共约 30 s)
    try:
        import gmsh as _gmsh_chk  # noqa: F401
        from rw.em.eddy3d_gmsh import BilletInCoil3D
    except ImportError:
        print("[11g] gmsh 未安装，跳过非圆截面涡流自检 (pip install gmsh 可解锁，不影响其它功能)")
    else:
        import gc
        pt = BilletInCoil3D(a=0.0, ri=8.0, ro=11.0, ecc_mm=0.0).build().solve()
        msg = f"[11g] gmsh 非结构网格·同几何空心管 ({pt.mesh.t.shape[1]} 四面体): P={pt.P_tube:.2f} W"
        if os.path.exists(fn):
            pr = json.load(open(fn))["ecc0"]
            d3, d2 = pt.P_tube / pr["P_3d"] - 1, pt.P_tube / pr["P_2d"] - 1
            print(f"{msg} | vs 结构网格 {pr['P_3d']:.2f} W {d3*100:+.2f}%，vs 轴对称 {pr['P_2d']:.2f} W {d2*100:+.2f}%")
            assert abs(d3) < 0.06 and abs(d2) < 0.06
        else:
            print(msg + " (无结构网格缓存，仅自检运行)")
        assert pt.P_air_reg / pt.P_tube < 0.01
        del pt; gc.collect()
        pb = BilletInCoil3D(a=20.0, ecc_mm=0.0).build().solve()
        t4, pp = pb.mesh.t, pb.mesh.p                    # 网格为 SI 单位(m)，读数时换回 mm
        cr = np.cross((pp[:, t4[1]] - pp[:, t4[0]]).T, (pp[:, t4[2]] - pp[:, t4[0]]).T)
        vol = np.abs(np.sum(cr * (pp[:, t4[3]] - pp[:, t4[0]]).T, axis=1)) / 6.0
        msk = pb.tube
        assert abs(vol[msk].sum() * 1e9 / 28000 - 1) < 0.01          # 几何自检: 20×20×70 方坯体积
        x, y, z = (pb.cen[i][msk] * 1000 for i in range(3))
        pq, vv = pb.pq[msk], vol[msk]
        zw = (z >= 27) & (z <= 52)                        # 线圈窗口
        depth = 10.0 - np.maximum(np.abs(x), np.abs(y))   # 到方坯表面的距离
        sf, co = zw & (depth < 1.5), zw & (depth > 4.5)
        skin = (pq[sf].sum() / vv[sf].sum()) / (pq[co].sum() / vv[co].sum())
        cor = sf & (np.minimum(np.abs(x), np.abs(y)) > 6.0)
        fac = sf & (np.minimum(np.abs(x), np.abs(y)) < 3.0)
        corner = (pq[cor].sum() / vv[cor].sum()) / (pq[fac].sum() / vv[fac].sum())
        sym = pq[zw & (y > 0)].sum() / pq[zw & (y < 0)].sum()
        air = pb.P_air_reg / pb.P_tube
        print(f"[11g] 方坯 20×20 (δ≈3 mm): P={pb.P_tube:.2f} W，表层/心部功率密度 {skin:.1f}× (≫1 集肤)，"
              f"角部/面心 {corner:.2f} (<1: δ 与角部过渡区重叠、两侧感应电流相消的「冷角」regime)，"
              f"左右对称 {sym:.3f}，空气假损耗 {air*100:.2f}%")
        assert skin > 3 and 0.9 < sym < 1.1 and 0.3 < corner < 3 and air < 0.01
    print("§11 全部通过")


if __name__ == "__main__":
    run()
