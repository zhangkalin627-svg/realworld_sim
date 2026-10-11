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
