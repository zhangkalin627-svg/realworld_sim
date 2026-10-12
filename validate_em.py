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
    # g) 非法输入在入口抛 ValueError (问题单 20261012-1925-em-input-validation)
    bad = [dict(freq=-30e3), dict(freq=0.0), dict(power_kw=-5), dict(mur_cold=0.0), dict(dt=0.0),
           dict(t_heat=-1), dict(freq=float("nan")), dict(medium="no_such"), dict(inner_quench=2)]
    nok = 0
    for kw in bad:
        try:
            InductionQuenchSim(**kw)
        except ValueError:
            nok += 1
    print(f"[11g] 非法输入(负/零频率、负功率、μr<1、dt=0、负时间、NaN、未知介质…) 入口拒绝: {nok}/{len(bad)}")
    assert nok == len(bad)
    # h) 100 kW 满功率专项 (rw.em.fullpower): 空气磁场 / 杂散场加热 / 线圈铜损 / 导磁体集总热
    from rw.em import fullpower as FP
    from rw.em.aux import AuxBody
    from rw.induction import BottleNeckGeometry
    g = BottleNeckGeometry()
    s0 = InductionQuenchSim(geom=g, freq=30e3, power_kw=100.0, nonlinear=True, t_heat=1, t_quench=0, t_air=0)
    fc = FP.field_check(s0)
    print(f"[11h-1] 线圈空气场 有限元 vs Biot–Savart 圆环求和({fc['n']} 单元): 中位 {fc['p50']*100:.2f}%、95% 分位 {fc['p95']*100:.2f}% (相对峰值 {fc['Bmax_mT_per_kA']:.1f} mT/kA)")
    assert fc["p50"] < 0.02 and fc["p95"] < 0.04
    from scipy.integrate import solve_ivp
    P_, V_, kind = 440.0, 22e-6, "smc"
    C_ = FP.CONC_RHO_CP[kind] * V_
    for hA in (0.0, 2.0):
        r = FP.conc_thermal(P_, V_, kind, T0=30.0, T_water=30.0, hA_W_K=hA, t_end=15.0)
        sol = solve_ivp(lambda t, T: (P_ - hA * (T - 30.0)) / C_, (0, 15.0), [30.0], rtol=1e-10, atol=1e-10)
        e = abs(sol.y[0, -1] - r["T_end"]) / (sol.y[0, -1] - 30.0)
        print(f"[11h-2] 导磁体集总热 hA={hA}: 解析 T(15s)={r['T_end']:.2f} vs 数值积分 {sol.y[0,-1]:.2f} °C (偏差 {e*100:.4f}%)；到 {r['T_limit']:.0f}°C 用时 {r['t_limit_s']:.1f} s")
        assert e < 1e-5
    res_aux = {}
    for mat, lay in (("al", (0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4)), ("ss304", (0.25, 0.5, 1.0, 2.0, 3.5))):
        a = AuxBody(27, 32, 15, 65, mat, "套筒", layers=lay)
        sa = InductionQuenchSim(geom=g, freq=30e3, power_kw=100.0, nonlinear=True, t_heat=1, t_quench=0, t_air=0, aux=[a])
        r = FP.stray_check(sa)[0]
        res_aux[mat] = r
        print(f"[11h-3] 杂散场加热 {mat} 套筒(r 27–32 mm, δ={r['skin_mm']:.2f} mm): 有限元 {r['P_fem_W']/1e3:.2f} kW vs 表面阻抗公式 {r['P_est_W']/1e3:.2f} kW (比值 {r['ratio']:.3f})，绝热升温 {r['dTdt_K_s']:.0f} K/s")
        assert abs(r["ratio"] - 1) < 0.08
    for bad_a in (AuxBody(16, 18, 30, 40, "al"), AuxBody(100, 105, 0, 10, "al")):
        try:
            InductionQuenchSim(geom=g, aux=[bad_a])
            raise AssertionError("辅助体非法位置未拒绝")
        except ValueError:
            pass
    rr = {}
    for nm, n in (("粗", (0.5, 1.0)), ("细", (0.1, 0.25, 0.5, 1.0))):
        sc = InductionQuenchSim(geom=g, freq=30e3, power_kw=100.0, nonlinear=True, t_heat=1, t_quench=0, t_air=0,
                                mesh_extra=FP.coil_refine_lines(g, 30e3, n=n))
        rr[nm] = FP.coil_loss_fem(sc)
    f_ = rr["细"]
    d_ = abs(rr["粗"]["coil_loss_kW"] / f_["coil_loss_kW"] - 1)
    print(f"[11h-4] 线圈铜损(真导体 FEM, 100 kW/30 kHz): {f_['coil_loss_kW']:.1f} kW，效率 {f_['eff']*100:.1f}%，电流 {f_['I_rms_A']/1e3:.2f} kA；复功率恒等式偏差 {f_['identity_err']*100:.1f}%；"
          f"粗/细网格差 {d_*100:.1f}%；旧表面薄层估算 {f_['coil_loss_est_kW']:.1f} kW (FEM/估算 {f_['ratio_vs_est']:.2f})；水冷热流 {f_['heat_flux_MWm2']:.1f} MW/m²")
    assert f_["identity_err"] < 0.04 and d_ < 0.05 and 0.5 < f_["ratio_vs_est"] < 1.0
    # [11h-5] 铜电阻率随温度: 100 °C 铜损 vs 薄趋肤层标度 sqrt(ρ) (独立解析标度, 允许 4% 偏差), 并做水冷自洽迭代
    import numpy as _np
    r20 = FP.coil_loss_hot(sc, 20.0)
    r100 = FP.coil_loss_hot(sc, 100.0)
    sq = _np.sqrt(FP.rho_cu_at(100.0) / FP.RHO_CU)
    rat = r100["coil_loss_kW"] / r20["coil_loss_kW"] / sq
    cc = FP.coil_thermal_coupled(sc, T_water=30.0, h_w=1e5, t_wall_mm=1.5, cool_frac=0.6)
    Rth = 1.0 / 1e5 + 1.5e-3 / 390.0
    T_chk = 30.0 + cc["q_in_MWm2"] * 1e6 * Rth
    print(f"[11h-5] 铜损随温度: 20→100°C FEM 比值 {r100['coil_loss_kW']/r20['coil_loss_kW']:.3f}，薄趋肤标度 √ρ 比值 {sq:.3f}，FEM/标度 {rat:.3f}；"
          f"水冷自洽(h_w=1e5): 铜温 {cc['T_cu']:.1f}°C、铜损 {cc['P_c_kW']:.1f} kW、效率 {cc['eff']*100:.1f}%、热流 {cc['q_in_MWm2']:.2f} MW/m²，迭代 {cc['iters']} 次，"
          f"收敛={cc['converged']}，热阻回代残差 {abs(T_chk-cc['T_cu']):.2f} K")
    assert abs(rat - 1) < 0.04 and cc["converged"] and abs(T_chk - cc["T_cu"]) < 1.0
    # [11h-6] 导磁体居里点/温度反馈
    from rw.em.conc import Concentrator
    P0_, k_, hA_, C_V = 20.0, 0.1, 0.05, 22e-6
    Tg_ = _np.array([20.0, 100.0, 220.0])
    Pg_ = P0_ - k_ * (Tg_ - 20.0)
    rh = FP.conc_heatup(Tg_, Pg_, C_V, "ferrite", T0=20.0, T_water=20.0, hA_W_K=hA_, t_end=300.0, T_stop=1e9)
    Cc = FP.CONC_RHO_CP["ferrite"] * C_V
    lam = (k_ + hA_) / Cc
    T_an = 20.0 + P0_ / (k_ + hA_) * (1 - _np.exp(-lam * rh["t"][-1]))
    print(f"[11h-6a] 导磁体升温 ODE(损耗随 T 线性下降) vs 解析指数解: T(300s)={rh['T_end']:.4f} vs {T_an:.4f} °C")
    assert abs(rh["T_end"] - T_an) < 1e-4
    sfe = InductionQuenchSim(geom=g, freq=30e3, power_kw=100.0, nonlinear=True, t_heat=1, t_quench=0, t_air=0, conc=Concentrator("ferrite"))
    sn = InductionQuenchSim(geom=g, freq=30e3, power_kw=100.0, nonlinear=True, t_heat=1, t_quench=0, t_air=0)
    _, In = sn.solve_em(_np.full(sn.tb.N, 25.0)); In /= _np.sqrt(2)
    Tgrid = [20.0, 120.0, 190.0, 200.0, 230.0]
    rf = FP.conc_loss_vs_T(sfe, Tgrid)
    old = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rw", "em", "fullpower_results.json")))["conc"]["ferrite"]["loss_kW"] * 1e3
    d20 = rf["loss_W"][0] / old - 1
    dI = rf["I_rms_A"][-1] / In - 1
    print(f"[11h-6b] 铁氧体(Tc=200°C) 20/120/190/200/230°C: 损耗 " + "/".join(f"{x:.1f}" for x in rf["loss_W"]) + " W，线圈电流 "
          + "/".join(f"{x:.0f}" for x in rf["I_rms_A"]) + f" A；20°C 损耗与旧结果差 {d20*100:+.2f}%；230°C 电流与无导磁体差 {dI*100:+.2f}% (无导磁体 {In:.0f} A)")
    assert abs(d20) < 0.01 and abs(dI) < 0.015 and rf["loss_W"][-1] < 1e-6 * rf["loss_W"][0] and _np.all(_np.diff(rf["I_rms_A"]) >= 0) and rf["I_rms_A"][2] > rf["I_rms_A"][0]
    V_ = FP.conc_volume_m3(sfe)
    hu = FP.conc_heatup(rf["T"], rf["loss_W"], V_, "ferrite", T0=30.0, hA_W_K=0.0, t_end=3000.0, T_stop=1e9)
    Cf = FP.CONC_RHO_CP["ferrite"] * V_
    E_int = _np.trapezoid(hu["P_W"], hu["t"])
    t190 = float(_np.interp(190.0, hu["T"], hu["t"]))
    print(f"[11h-6c] 铁氧体绝热升温(30°C 起, 100 kW): {t190:.0f} s 到 190°C，3000 s 末 {hu['T_end']:.1f}°C；损耗积分 {E_int:.0f} J vs C·ΔT {Cf*(hu['T_end']-30):.0f} J")
    assert abs(E_int / (Cf * (hu["T_end"] - 30.0)) - 1) < 0.02 and 198.0 < hu["T_end"] < 200.5
    # [11h-7] 外部热输入 (辐射)
    from rw.em import rad as RAD
    from scipy.optimize import brentq
    vf = [(0.05, 0.05, 0.02), (0.03, 0.08, 0.05), (0.1, 0.02, 0.01)]
    dvf = max(abs(RAD.view_factor_numeric(*a) / RAD.view_factor_coaxial_discs(*a) - 1) for a in vf)
    print(f"[11h-7a] 同轴圆盘视角系数 数值积分 vs Hottel 解析式 最大相对差 {dvf:.2e}")
    assert dvf < 1e-6
    Cb, eA, Tsrc = 2000.0, 0.8 * 0.5 * 0.01, 900.0
    t_an = RAD.radiation_only_time(30.0, 150.0, Tsrc, Cb, eA)
    qr = lambda T: RAD.SIGMA * eA * ((Tsrc + 273.15) ** 4 - (T + 273.15) ** 4)
    ho = FP.conc_heatup([0.0, 1000.0], [0.0, 0.0], Cb / FP.CONC_RHO_CP["ferrite"], "ferrite", T0=30.0, q_ext_W=qr, t_end=1e5, T_stop=150.0)
    print(f"[11h-7b] 纯辐射升温 30→150°C: 解析 {t_an:.2f} s vs ODE {ho['t_stop_s']:.2f} s")
    assert abs(ho["t_stop_s"] / t_an - 1) < 1e-4
    F12_, A_f = 0.5, 0.01
    qf = lambda T: RAD.rad_exchange_W(Tsrc, T, A_f, 0.8, 0.9, F12=F12_)
    hAw = 4.0
    hs = FP.conc_heatup(rf["T"], rf["loss_W"], V_, "ferrite", T0=30.0, T_water=30.0, hA_W_K=hAw, q_ext_W=qf, t_end=1e6, T_stop=1e9)
    from scipy.interpolate import PchipInterpolator as _Pc
    Pf = _Pc(rf["T"], rf["loss_W"])
    bal = lambda T: float(Pf(T)) + qf(T) - hAw * (T - 30.0)
    T_ss = brentq(bal, 30.0, rf["T"][-1])
    print(f"[11h-7c] 铁氧体 损耗+辐射+水冷 稳态 ODE {hs['T_end']:.3f} vs 根求解 {T_ss:.3f} °C")
    assert abs(hs["T_end"] - T_ss) < 1e-3
    for bad in [dict(eps1=0.0), dict(eps2=1.5), dict(F12=0.0), dict(A1=-1.0)]:
        kw = dict(Ts_C=900.0, Tb_C=30.0, A1=0.01); kw.update(bad)
        try:
            RAD.rad_exchange_W(**kw); raise AssertionError(f"应拒绝 {bad}")
        except ValueError:
            pass
    print("[11h-7d] 辐射输入检查: 发射率/视角系数/面积越界 全部拒绝")

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
