import numpy as np, warnings; warnings.filterwarnings('ignore')
from rw.induction import InductionQuenchSim
from rw import materials as M
# 1) 透入深度检验：冷态 100 kHz，焦耳热应按 exp(-2x/δ) 衰减
for f in (100e3, 30e3):
    s=InductionQuenchSim(freq=f,power_kw=9)
    T=np.full(s.tb.N,25.0); q,I=s.solve_em(T)
    p=s.mesh_mm.p; t=s.mesh_mm.t[:,s.steel_idx]; c=p[:,t].mean(axis=1)
    sel=(np.abs(c[1]-40)<0.6)
    x=s.g.ro-c[0,sel]; qq=q[sel]; o=np.argsort(x); x,qq=x[o],qq[o]
    m=(x<1.0*s.skin_depth_mm(25)) 
    slope=np.polyfit(x[m],np.log(qq[m]),1)[0]
    print(f"f={f/1e3:.0f}kHz 理论δ={s.skin_depth_mm(25):.3f}mm  FEM拟合δ={-2/slope:.3f}mm")
# 2) 能量守恒：加热阶段吸收能量 vs 输入功率×时间
s=InductionQuenchSim(freq=30e3,power_kw=5,t_heat=2,t_quench=0.01,t_air=0.0)
s.run()
Tq=s.tb.interpolate(s.T_end_heat).value
from skfem import Functional
@Functional
def E(w): return M.STEEL_45['density']*w['e']*w.x[0]*2*np.pi
# 焓 = ∫cp dT
Tg=np.linspace(25,1500,3000); H=np.concatenate([[0],np.cumsum(np.diff(Tg)*M.steel_cp(Tg[:-1]))])
e=np.interp(Tq,Tg,H)
print("吸收能量 %.0f J, 输入 %.0f J (差额=空冷散热+时间离散)"%(E.assemble(s.tb,e=e),5000*2))

# 3) 梁单元：悬臂梁端部受力，挠度/应力 对比材料力学公式
from rw.beam import Frame, Section
from rw.matlib import MAT
f = Frame(); sec = Section.rhs(80, 80, 4)
ids = f.beam((0, 0, 0), (1000, 0, 0), sec, MAT['Q235'], nseg=10)
f.fix[ids[0]] = [True] * 6; f.add_load(ids[-1], F=(0, 0, -1000)); f.solve(self_weight=False)
print("悬臂梁挠度 FEM %.4f mm / 理论 %.4f mm" % (-f.u[ids[-1]][2], 1000 * 1000**3 / (3 * 206e3 * sec.Iy)))
# 4) 板单元：内边固定、外边受线载荷的环形板，对比解析解
from rw.plate import annulus_mesh, Plate
a, b, h, nu = 350., 55., 20., 0.33
m = annulus_mesh(b, a, 24, 120); pl = Plate(m, h, 69e9, nu, 2700)
outer = np.nonzero(np.isclose(np.hypot(*m.p), a))[0]
for i in outer: pl.f[pl.basis.nodal_dofs[0, i]] -= 1000 / len(outer)
pl.clamp_inner(b); pl.solve(); D = pl.D
def rows(r):
    return (np.array([1, r*r, np.log(r), r*r*np.log(r)]), np.array([0, 2*r, 1/r, 2*r*np.log(r)+r]),
            np.array([0, 2, -1/r**2, 2*np.log(r)+3]), np.array([0, 0, 2/r**3, 2/r]))
wb, w1b, _, _ = rows(b); wa, w1a, w2a, w3a = rows(a)
A_ = np.vstack([wb, w1b, -D*(w2a+nu*w1a/a), -D*(w3a+w2a/a-w1a/a**2)])
C_ = np.linalg.solve(A_, [0, 0, 0, -1000/(2*np.pi*a)])
print("环形板外缘挠度 FEM %.4f mm / 解析 %.4f mm" % (-pl.deflection_at((a, 0)), -(wa @ C_)))
# 5) PLC 演示程序冒烟测试：FX 指令表（STL 步进）+ 单工位设备模型，跑 60 s
from rw.plc.demo import DemoSim
s = DemoSim()
for _ in range(50): s.scan()
s.press('X000')
for _ in range(6000): s.scan()
print("PLC 演示：60 s 完成 %d 件，测温不合格 %d 件，每件加热 %.2f s / 淬火 %.2f s（程序设定 3.0 / 3.0）" % (
    s.plc.words.get('D0', 0), s.plc.words.get('D1', 0), s.plant.done[0].heat, s.plant.done[0].quench))
# 6) 物理沙盒：静压、沸点、层流摩擦系数、电机额定点
from rw.world.immersion import pressure_at, Env
from rw.world.fluids import water_tsat, water_psat
from rw.world.pipes import friction_factor
from rw.world.motor import InductionMotor
pa, pg, rho = pressure_at(Env('water', 4, 10))
print("10 m 清水(4°C) 表压 %.2f kPa（理论 98.07）；沸点@1atm %.2f°C；200°C 饱和压力 %.3f MPa（蒸汽表 1.555）" % (pg/1e3, water_tsat(101325), water_psat(200)/1e6))
print("层流 Re=1000 摩擦系数 %.4f（理论 0.064）" % friction_factor(1000, 0))
m = InductionMotor(); import math
print("电机额定点转矩 %.3f N·m（P/ω=%.3f）" % (m.torque(2*math.pi*m.n_rated/60), m.Tn))
# 7) 材料库：元素数据对照手册值（CRC Handbook），混合法则对照牌号实测，互溶判断对照二元相图
from rw.matdb.elements import ELEMENTS, get
from rw.matdb.mixing import mix, check_against_grades, binary
from rw.matdb.alloys import GRADES
print("元素数 %d（应为 118）；Fe 密度 %.0f kg/m³（CRC 7874）；Al 熔点 %.2f°C（660.32）；W 熔点 %.0f°C（3422）；"
      "Cu 电阻率 %.3f μΩ·cm（1.678）；Fe 弹性模量 %.0f GPa（211）" % (
      len(ELEMENTS), get('Fe')['density'], get('Al')['melt_C'], get('W')['melt_C'], get('Cu')['resist'] * 1e8, get('Fe')['E']))
print("铁+银：%s；铜+镍：%s" % (binary('Fe', 'Ag')[0], binary('Cu', 'Ni')[0]), "（应为 liq_gap / ss）")
rows, agree = check_against_grades()
errs = [abs(r[3]) for r in rows]
worst = max(rows, key=lambda r: abs(r[3]))
print("混合法则密度 vs %d 个牌号实测：平均偏差 %.1f%%，最大 %s %+.1f%%" % (len(rows), sum(errs) / len(errs), worst[0], worst[3]))
print("模型(Miedema+Hume-Rothery) vs 二元相图结论：%d/%d 一致 (%.0f%%)" % (sum(a[4] for a in agree), len(agree), 100 * sum(a[4] for a in agree) / len(agree)))
inr = []
for k in ("Constantan", "H62", "H68", "Monel400", "Cr20Ni80", "QSn6.5-0.1", "Invar36", "304"):
    r = mix(GRADES[k]["comp"]); lo, hi = r["resist_range"]; inr.append(lo <= GRADES[k]["resist"] <= hi)
print("固溶体电阻率：实测落在估算范围内 %d/%d 个牌号" % (sum(inr), len(inr)))
# 8) 电路仿真（ngspice）+ 虚拟示波器：和解析解对比
from rw.circuit.spice import find_ngspice, ngspice_version
if not find_ngspice():
    print("电路仿真：没找到 ngspice，跳过（Ubuntu: sudo apt install ngspice）")
else:
    from rw.circuit import examples as CX, kicad as KI
    from rw.circuit.scope import Scope, measure, fft_dbv, peak_near, crossings
    pct = lambda a, b: 100 * (a - b) / b
    print("ngspice 版本 %s" % ngspice_version())
    # 8a RC：63.2% 点 = τ；10–90% 上升时间 = τ·ln9 ≈ 2.197τ
    e = CX.get("rc"); w = e.circuit.tran(**e.tran); tau = e.values["tau"]
    t, y = w.x, w["out"]; t63 = crossings(t, y, 0.632120559 * e.values["V"], "rise")[0]
    tr = measure(t[t < 0.5 / e.values["f"]], y[t < 0.5 / e.values["f"]])["上升时间"]
    print("RC：τ 理论 %.4f ms / 仿真 63.2%%点 %.4f ms（误差 %+.2f%%）；上升时间 理论 %.4f ms / 仿真 %.4f ms（%+.2f%%）" % (
        tau * 1e3, t63 * 1e3, pct(t63, tau), e.values["tr"] * 1e3, tr * 1e3, pct(tr, e.values["tr"])))
    # 示波器：触发 + 屏幕测量（RC 输出，第 2 个周期）
    sc = Scope(); sc.load(w); sc.ch[0].source = "out"; sc.ch[0].on = True; sc.tdiv = 4e-3; sc.hpos = 4 * 4e-3
    sc.trig.source, sc.trig.level = 1, 0.05; fr = sc.acquire(); m = sc.measure(1, fr)
    print("  示波器(RC 输出)：频率 %.3f Hz（理论 %.0f）、上升时间 %.4f ms（理论 %.4f）、Vmax %.4f V（理论 1-e^-10=%.4f）" % (
        m["频率"] or float('nan'), e.values["f"], (m["上升时间"] or float('nan')) * 1e3, e.values["tr"] * 1e3, m["Vmax"], 1 - 2.718281828 ** -10))
    # 8b RLC 串联：交流扫频找电流峰值（R 两端电压最大）→ f0
    e = CX.get("rlc"); w = e.circuit.ac(**e.ac); import numpy as _np
    mag = _np.abs(w["b"]); i = int(_np.argmax(mag)); f = w.x
    a_, b_, c_ = _np.log(mag[i - 1:i + 2]); fx = _np.log(f[i - 1:i + 2])  # 对数坐标抛物线插值
    fpk = _np.exp(fx[1] + 0.5 * (fx[2] - fx[1]) * (a_ - c_) / (a_ - 2 * b_ + c_))
    print("RLC：谐振频率 理论 %.2f Hz / 仿真 %.2f Hz（误差 %+.3f%%）；峰值 |V_R/V_in| = %.4f（理论 1）" % (e.values["f0"], fpk, pct(fpk, e.values["f0"]), mag[i]))
    wt = e.circuit.tran(**e.tran); vc = wt["a-b"]; tt = wt.x; m = (tt > 1e-6) & (tt < 2.4e-3)
    pk = crossings(tt[m], vc[m] - 0.5 * 1.0 * 0 - 1.0, 0.0, "rise")  # 绕终值 1 V 的振荡
    fd = 1 / _np.mean(_np.diff(pk[:6])) if len(pk) > 2 else float('nan')
    print("  方波激励振铃频率 理论 fd=%.2f Hz / 仿真 %.2f Hz（误差 %+.2f%%）" % (e.values["fd"], fd, pct(fd, e.values["fd"])))
    # 8c 桥式整流 + 电容滤波：纹波 vs I/(2fC)
    e = CX.get("rect"); w = e.circuit.tran(**e.tran); t = w.x; m = t > t[-1] - 5 / e.values["f"]
    v = w["out"][m]; I = measure(t[m], w["i(vsense)"][m])["Vavg"]; rip = v.max() - v.min()
    th = I / (2 * e.values["f"] * e.values["C"])
    print("整流：负载电流 %.4f A，纹波 仿真 %.4f V / 近似公式 I/(2fC)=%.4f V（公式偏大 %+.1f%%，因为公式假设整个半周都在放电）" % (I, rip, th, pct(th, rip)))
    # 8d FFT：理想方波（0~1 V，1 kHz）基波 2/π/√2 Vrms，3 次/5 次谐波 = 1/3、1/5
    from rw.circuit.spice import Circuit
    q = Circuit("square"); q.V("1", "sq", "0", pulse=(0, 1, 0, 1e-9, 1e-9, 0.5e-3 - 1e-9, 1e-3)); q.R("1", "sq", "0", 1e3)
    w = q.tran(1e-6, 20e-3, tmax=1e-6); f, db, vr = fft_dbv(w.x, w["sq"], "flattop")
    f1, a1 = peak_near(f, vr, 1e3); f3, a3 = peak_near(f, vr, 3e3); f5, a5 = peak_near(f, vr, 5e3)
    import math as _m
    th1 = 2 / _m.pi / _m.sqrt(2)
    print("FFT（平顶窗，理想方波、整周期采样 → 主要验证 FFT 幅值归一化）：基波 %.1f Hz（理论 1000）幅值 %.4f Vrms（理论 %.4f，%+.2f%%，%.2f dBV）；3 次/基波 %.4f（理论 0.3333，%+.2f%%）；5 次/基波 %.4f（理论 0.2000，%+.2f%%）" % (
        f1, a1, th1, pct(a1, th1), 20 * _m.log10(a1), a3 / a1, pct(a3 / a1, 1 / 3), a5 / a1, pct(a5 / a1, 0.2)))
    sc = Scope(); sc.load(w); sc.ch[0].source = "sq"; sc.ch[0].on = True; sc.tdiv = 0.5e-3; sc.trig.level = 0.5
    fr = sc.acquire(); mm = sc.measure(1, fr)
    print("  示波器(方波)：Vpp %.4f（1）、Vavg %.4f（0.5）、Vrms %.4f（0.7071）、频率 %.3f Hz（1000）、占空比 %.4f（0.5）" % (
        mm["Vpp"], mm["Vavg"], mm["Vrms"], mm["频率"], mm["占空比"]))
    # 8e 正弦：示波器 Vrms = A/√2，频率；运放振荡器频率 vs 2RC·ln3
    s = Circuit("sin"); s.V("1", "x", "0", sin=(0.5, 2.0, 50e3)); s.R("1", "x", "0", 1e3)
    w = s.tran(10e-9, 200e-6, tmax=10e-9); sc = Scope(); sc.load(w); sc.ch[0].source = "x"; sc.ch[0].on = True
    sc.tdiv = 10e-6; sc.trig.level = 0.5; fr = sc.acquire(); mm = sc.measure(1, fr)
    sc.ch[0].coupling = "AC"; sc.trig.level = 0.0; fr2 = sc.acquire(); ma = sc.measure(1, fr2)
    print("  示波器(正弦 0.5+2sin)：Vpp %.4f（4）、Vavg %.4f（0.5）、Vrms DC耦合 %.4f（√(0.25+2)=1.5000）、AC耦合 %.4f（1.4142）、频率 %.2f Hz（50000）" % (
        mm["Vpp"], mm["Vavg"], mm["Vrms"], ma["Vrms"], mm["频率"]))
    e = CX.get("astable"); w = e.circuit.tran(**e.tran); m = w.x > w.x[-1] * 0.4
    fo = measure(w.x[m], w["out"][m])["频率"]
    print("运放振荡器：频率 理论 %.2f Hz / 仿真 %.2f Hz（误差 %+.2f%%；行为级运放压摆率 10 V/µs、输出 ±13 V）" % (e.values["f"], fo, pct(fo, e.values["f"])))
    # 8f KiCad 网表导入：两种格式结果一致；集电极上升时间 ≈ 2.2·R2·C1
    c1 = KI.load("models/netlists/npn_switch.cir"); c2 = KI.load("models/netlists/npn_switch_kicad.net")
    w1 = c1.run(KI.default_analysis(c1)); w2 = c2.run(".tran 1u 5m")
    r1, r2 = measure(w1.x, w1["/col"]), measure(w2.x, w2["/col"])
    print("KiCad 导入：.cir 与 .net 集电极上升时间 %.4f / %.4f ms（理论 2.197·1k·100n = 0.2197 ms，%+.2f%%）；.net 未映射元件 %d 个（J1 连接器，预期）" % (
        r1["上升时间"] * 1e3, r2["上升时间"] * 1e3, pct(r1["上升时间"], 2.197e-4), len(c2.warnings)))
    e = CX.get("halfbridge"); w = e.circuit.tran(**e.tran); m = w.x > 0.6e-3
    mi = measure(w.x[m], w["i(vcoil)"][m]); ms = measure(w.x[m], w["sw"][m])
    print("半桥演示（只做冒烟检查，无解析对比）：开关频率 %.0f Hz（设定 40000），开关节点 %.1f~%.1f V，线圈电流 %.1f A 峰值 / %.1f A 有效值" % (
        ms["频率"], ms["Vmin"], ms["Vmax"], mi["Vmax"], mi["Vrms"]))


# 9) v0.7 单片机仿真接 PLC：指令计时、定时器、PWM、ADC、UART、PLC↔MCU 握手、过流保护（理论 vs 仿真）
print("\n=== 9) 单片机（AVR 子集）+ FX PLC + 电路 联合仿真 ===")
from rw.mcu import selftest as _mcu_st
_mcu_st.run()


# 10) v0.8 整机联动：上电 → PLC → 变频器 → 主回路 → 机械 → 加热 → 喷淋（理论/ngspice 对比 + 能量守恒 + 故障传播）
print("\n=== 10) 整机联动 ===")
from rw.machine import selftest as _mach_st
_mach_st.run()


# 10) v0.8 整机联动：上电 → PLC → 变频器 → 主回路 → 机械 → 加热 → 喷淋（理论/ngspice 对比 + 能量守恒 + 故障传播）
print("\n=== 10) 整机联动 ===")
from rw.machine import selftest as _mach_st
_mach_st.run()


# 11) v0.9 电磁升级（非线性B-H/导磁体/三维涡流，见 validate_em.py）
import validate_em; validate_em.run()

