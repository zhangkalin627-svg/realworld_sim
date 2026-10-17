"""铁磁粉尘迁移/堆积/拉弧风险的简化粒子动画模型（用户 2026-10-17 工况）。

⚠ 简化动力学，演示趋势，不是时间预测：
- 粒子过阻尼假设（微米级铁粉惯性可忽略）：v = v_空气 + F_磁/(6πηr)（限速）+ 重力沉降
- 磁力用饱和偶极量级 F = V·Ms·∇B，B 场来自轴对称涡流求解（同一材料求解一次，稳态）
- 粘附：粒子落到导磁体表面或线圈内壁即粘（水平过环工况，现场视频校准）；吹扫：局部空气拖曳力 > 磁保持力时释放
- 拉弧风险 = (最大堆积厚度/临界厚度) × (单匝电压/参考电压) 的定性分级，不可解击穿过程
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

MU0 = 4e-7 * np.pi
M_S_FE = 1.7e6          # 纯铁饱和磁化强度 A/m（上限）
M_S_DUST = 5e5          # 现场粉尘=热轧氧化铁皮(Fe3O4 为主)，磁化强度远低于纯铁
RHO_FE = 7800.0         # 铁密度 kg/m3
ETA_AIR = 1.8e-5        # 空气动力黏度 Pa·s
G = 9.81


def _gradB_vec(mesh, B1):
    """单元 ∇B 两个分量 (T/m)：单元值→节点平均→三角形线性梯度。"""
    p, t = mesh.p, mesh.t
    bn = np.zeros(p.shape[1]); cnt = np.zeros(p.shape[1])
    np.add.at(bn, t, np.broadcast_to(B1, t.shape))
    np.add.at(cnt, t, 1)
    bn /= np.maximum(cnt, 1)
    x = p[0, t] * 1e-3; y = p[1, t] * 1e-3
    v = bn[t]
    den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    dvdx = ((y[1] - y[2]) * (v[0] - v[2]) + (y[2] - y[0]) * (v[1] - v[2])) / den
    dvdy = ((x[2] - x[1]) * (v[0] - v[2]) + (x[0] - x[2]) * (v[1] - v[2])) / den
    return dvdx, dvdy


class DustSim:
    """对一个已求解的感应仿真（solve_em 之后）建粉尘动画场。单位：位置 mm，时间 s。"""

    def __init__(self, sim, V_turn=4.0, d_um=50.0, seed=7):
        self.s = sim
        self.g = sim.g
        self.V_turn = V_turn
        self.r_p = d_um * 0.5e-6                    # 颗粒半径 m
        self.V_p = 4 / 3 * np.pi * self.r_p ** 3
        self.m_p = self.V_p * RHO_FE
        self.rng = np.random.default_rng(seed)
        g = self.g
        r_out = (sim.conc.r_edges(g) + 8.0) if sim.conc is not None else g.coil_r1 + 10.0
        z0 = g.coil_z0 - (sim.conc.leg_t if sim.conc else 0) - 12.0
        z1 = g.coil_z1 + (sim.conc.leg_t if sim.conc else 0) + 12.0
        self.bounds = (0.0, r_out, z0, z1)
        # ---- 场插值：单元心 KDTree → 规则网格 → 线性插值器
        c = sim.mesh_mm.p[:, sim.mesh_mm.t].mean(axis=1)   # (2, nel) mm
        B1 = sim._B1_el
        gx, gy = _gradB_vec(sim.mesh_mm, B1)
        gr = np.linspace(self.bounds[0], self.bounds[1], 90)
        gz = np.linspace(self.bounds[2], self.bounds[3], 160)
        GG = np.meshgrid(gr, gz, indexing="ij")
        pts = np.stack([GG[0].ravel(), GG[1].ravel()], axis=1)
        _, idx = cKDTree(c.T).query(pts)
        Bg = B1[idx].reshape(GG[0].shape)
        gx_g = gx[idx].reshape(GG[0].shape)
        gy_g = gy[idx].reshape(GG[0].shape)
        self._iB = RegularGridInterpolator((gr, gz), Bg, bounds_error=False, fill_value=0.0)
        self._igx = RegularGridInterpolator((gr, gz), gx_g, bounds_error=False, fill_value=0.0)
        self._igz = RegularGridInterpolator((gr, gz), gy_g, bounds_error=False, fill_value=0.0)
        # ---- 粒子状态
        self.pos = np.zeros((0, 2))                  # 空气中 (r, z) mm
        self.stuck = np.zeros((0, 2))                # 已粘附
        self.t = 0.0
        self.n_spawned = 0
        # 统计：积灰区分段（腿尖内侧），用于层厚与风险
        self.seg_count = {}

    # ------------------------------------------------------------ 场查询
    def B_at(self, pts):
        return self._iB(pts)

    def gradB_at(self, pts):
        gx = self._igx(pts); gz = self._igz(pts)
        return np.stack([gx, gz], axis=1)

    def F_mag(self, pts, cap=300.0):
        """饱和偶极磁力 (N, 方向 +∇B)。|∇B| 截断到 cap (T/m)：
        颗粒只响应自身尺度上的外场梯度；材料界面 μ 突变处的插值尖峰（可达 1e5 T/m）非物理。"""
        gb = self.gradB_at(pts)
        mag = np.minimum(np.hypot(gb[:, 0], gb[:, 1]), cap)
        F = self.V_p * M_S_DUST * mag                # N（粉尘按氧化铁皮量级，非纯铁）
        d = np.zeros_like(gb)
        nz = mag > 1e-12
        d[nz] = gb[nz] / mag[nz, None]
        return F, d

    # ------------------------------------------------------------ 区域判定
    def in_conc(self, pts):
        if self.s.conc is None:
            return np.zeros(len(pts), bool)
        return self.s.conc.mask(self.g, pts[:, 0], pts[:, 1])

    def in_coil(self, pts):
        g = self.g
        return (pts[:, 0] >= g.coil_r0) & (pts[:, 0] <= g.coil_r1) & (pts[:, 1] >= g.coil_z0) & (pts[:, 1] <= g.coil_z1)

    def in_steel(self, pts, bar_shift=0.0):
        g = self.g
        return (pts[:, 0] <= g.ro) & (pts[:, 1] >= -g.Ls - 5) & (pts[:, 1] <= g.L + 5)

    def air_velocity(self, pts, blow):
        """环境气流 m/s（简化）：热工件浮力羽流把粉尘沿 +r 推离表面（近场 ~0.6 m/s 指数衰减），
        工件连续过环拖动空气沿线圈端面外流（~0.25 m/s 指向近端）；气枪=工业风枪 ~60 m/s 径向射流。"""
        g = self.g
        v = np.zeros((len(pts), 2))
        near = np.exp(-np.maximum(pts[:, 0] - g.ro, 0) / 6.0)
        v[:, 0] += 0.6 * near                         # 热羽流：径向推离工件
        zmid = 0.5 * (g.coil_z0 + g.coil_z1)
        v[:, 1] += 0.25 * np.sign(pts[:, 1] - zmid) * near   # 棒料连续进给拖动空气沿轴向端面外
        if blow:
            band = (pts[:, 1] > g.coil_z0 - 8) & (pts[:, 1] < g.coil_z1 + 8) & (pts[:, 0] > g.ro)
            v[band, 0] += 60.0                        # 工业风枪射流 ~60 m/s（0.4-0.6 MPa 喷嘴近场）
            v[band, 1] += 8.0
        return v

    # ------------------------------------------------------------ 步进
    def spawn(self, n):
        g = self.g
        if n <= 0:
            return
        r = g.ro + self.rng.uniform(0.3, 1.5, n)
        z = self.rng.uniform(g.coil_z0, g.coil_z1, n)
        self.pos = np.vstack([self.pos, np.stack([r, z], axis=1)])
        self.n_spawned += n

    def step(self, dt, blow=False, spawn_rate=8.0):
        """推进 dt 秒。过阻尼：v = v_空气 + F/(6πηr)（磁漂移限速 1.5 m/s）+ 重力沉降。
        内部子步：单子步位移 <=0.4 mm（防穿透粘附面），大步长自动拆。"""
        g = self.g
        # 新粉尘从热工件表面持续产生
        ns = self.rng.poisson(spawn_rate * dt)
        self.spawn(ns)
        n_sub = int(np.clip(np.ceil(dt / 0.05), 1, 200))
        h = dt / n_sub
        for _ in range(n_sub):
            if not len(self.pos):
                break
            F, d = self.F_mag(self.pos)
            v_term = np.minimum(F / (6 * np.pi * ETA_AIR * self.r_p), 1.5)
            v = self.air_velocity(self.pos, blow)
            v += d * v_term[:, None]
            # 重力：现场视频校准(2026-10-17 立式转 horizontal 通过式)：工件水平连续过环，
            # 重力 ⊥ 工件轴。取底侧半平面建模：氧化皮从棒料底面剥落，重力沿 +r（落向线圈/导磁体底侧）。
            v[:, 0] += self.m_p * G / (6 * np.pi * ETA_AIR * self.r_p)   # 重力沉降终端速度（径向）
            disp = v * h * 1e3                                # → mm
            L = np.hypot(disp[:, 0], disp[:, 1])
            cap = L > 0.4
            disp[cap] *= (0.4 / L[cap])[:, None]               # 子步位移上限 0.4 mm
            self.pos = self.pos + disp
            # 碰撞：导磁体/线圈内壁→粘附（现场视频：水平过环，氧化皮下落直接积在线圈内壁
            # 与导磁体底侧，二者都是积灰面；吹扫时粘不上，直接被射流带走）；工件→移除；出界→移除
            mc = self.in_conc(self.pos) | self.in_coil(self.pos)
            if blow and mc.any():
                self.pos = self.pos[~mc]              # 吹扫时打到导磁体的粒子视为被带走
                mc = np.zeros(len(self.pos), bool)
            if mc.any():
                self.stuck = np.vstack([self.stuck, self.pos[mc]])
                for p in self.pos[mc]:
                    key = (round(float(p[0]), 1), round(float(p[1]), 1))
                    self.seg_count[key] = self.seg_count.get(key, 0) + 1
            keep = ~mc
            bad = self.in_steel(self.pos)
            r0, r1, z0, z1 = self.bounds
            out = (self.pos[:, 0] < r0) | (self.pos[:, 0] > r1) | (self.pos[:, 1] < z0) | (self.pos[:, 1] > z1)
            self.pos = self.pos[keep & ~bad & ~out]
        # 吹扫释放：射流动压（½ρv²A）+ Stokes 拖曳 > 磁保持力 → 重新悬浮（量级判据）
        if blow and len(self.stuck):
            F, _ = self.F_mag(self.stuck)
            va = self.air_velocity(self.stuck, True)
            spd = np.hypot(va[:, 0], va[:, 1])
            drag = (6 * np.pi * ETA_AIR * self.r_p * spd
                    + 0.5 * 1.2 * spd ** 2 * np.pi * self.r_p ** 2)
            rel = drag > F
            if rel.any():
                self.pos = np.vstack([self.pos, self.stuck[rel]])
                self.stuck = self.stuck[~rel]
        self.t += dt

    # ------------------------------------------------------------ 指标
    def layer_max_um(self):
        """当前最大堆积厚度 µm（把现存粘附粒子按 0.1mm 网格分桶，最多桶的颗数折算，粗略量级）。"""
        if not len(self.stuck):
            return 0.0
        keys = np.round(self.stuck, 1)
        _, cnt = np.unique(keys, axis=0, return_counts=True)
        n = int(cnt.max())
        # 一颗占据表面积 ~ (2r_p)^2；轴对称环带面积极大，取保守"点堆积"上限：n 颗竖直叠放
        return n * (2 * self.r_p * 1e6) * 0.15     # 0.15=堆积孔隙修正（量级）

    def risk(self):
        """拉弧定性风险：层厚比 × 电压比 → 低/中/高。非预测，只作相对比较。"""
        x = (self.layer_max_um() / 200.0) * (self.V_turn / 50.0)
        if x >= 1.0:
            return "高", x
        if x >= 0.3:
            return "中", x
        return "低", x

    def stats(self):
        lv, x = self.risk()
        return dict(t=self.t, airborne=len(self.pos), stuck=len(self.stuck),
                    layer_um=self.layer_max_um(), risk=lv, risk_x=x, spawned=self.n_spawned)


def build_dust_sim(kind=None, freq=30e3, power_kw=6.0):
    """按材料建场（求解一次冷态电磁），返回 (DustSim, case 记录)。kind=None=无导磁体(木头基线)。"""
    from ..induction import InductionQuenchSim, BottleNeckGeometry
    from .conc import Concentrator
    from .check import _nonmag_mesh_extra
    conc = Concentrator(kind) if kind else None
    g = BottleNeckGeometry()
    extra = _nonmag_mesh_extra(conc, g, freq) if (conc and not conc.is_magnetic) else None
    s = InductionQuenchSim(geom=g, freq=freq, power_kw=power_kw, nonlinear=True,
                           conc=conc, t_heat=1.0, t_quench=0.0, t_air=0.0, mesh_extra=extra)
    q, I = s.solve_em(np.full(s.tb.N, 25.0))
    I_rms = I / np.sqrt(2)
    Pc = s.coil_loss_estimate(I)
    V_turn = (power_kw * 1e3 + Pc + s.conc_loss_W) / max(I_rms, 1e-9)
    rec = dict(name=(conc.name if conc else "无（木头基线）"), I_rms=I_rms, V_turn=V_turn,
               conc_loss_kW=s.conc_loss_W / 1e3)
    return DustSim(s, V_turn=V_turn), rec


# ------------------------------------------------------------ 场景绘制（GUI 与 GIF 共用）
def draw_static(ax, ds):
    """画静态背景：工件/线圈/导磁体 + 坐标轴。返回 (空气粒子 scatter, 粘附粒子 scatter)。"""
    from matplotlib.patches import Rectangle
    g = ds.g
    r0, r1, z0, z1 = ds.bounds
    ax.add_patch(Rectangle((0, z0), g.ro, z1 - z0, fc="#9a9a9a", ec="#555", lw=0.5, alpha=0.9))
    ax.add_patch(Rectangle((g.coil_r0, g.coil_z0), g.coil_r1 - g.coil_r0, g.coil_z1 - g.coil_z0,
                           fc="#d08a2e", ec="#7a4d00", lw=0.8, alpha=0.95))
    if ds.s.conc is not None:
        rr = np.linspace(g.coil_r0, r1, 110)
        zz = np.linspace(z0, z1, 190)
        RR, ZZ = np.meshgrid(rr, zz, indexing="ij")
        M = ds.s.conc.mask(g, RR, ZZ).astype(float)
        ax.contourf(RR, ZZ, M, levels=[0.5, 1.5], colors=["#3a6ea5"], alpha=0.9)
    sc_air = ax.scatter([], [], s=4, c="#333333", lw=0, alpha=0.75)
    sc_stk = ax.scatter([], [], s=9, c="#d62728", lw=0)
    ax.set_xlim(0, r1)
    ax.set_ylim(z1, z0)                      # z 轴倒置：上方为出料方向，视觉符合现场
    ax.set_xlabel("r (mm)")
    ax.set_ylabel("z (mm)")
    return sc_air, sc_stk


def update_scatter(ax, sc_air, sc_stk, ds, title_prefix=""):
    """刷新粒子与标题统计。"""
    sc_air.set_offsets(ds.pos if len(ds.pos) else np.zeros((0, 2)))
    sc_stk.set_offsets(ds.stuck if len(ds.stuck) else np.zeros((0, 2)))
    st = ds.stats()
    ax.set_title(f"{title_prefix}t={st['t']:.0f} s｜空气中 {st['airborne']}｜粘附 {st['stuck']}｜"
                 f"最大层厚 {st['layer_um']:.0f} µm｜拉弧风险【{st['risk']}】", fontsize=10)
    return st
