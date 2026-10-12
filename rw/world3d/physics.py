"""刚体物理（MuJoCo, Apache-2.0）+ 受力体检。
流程：Scene → MJCF → 让它在重力下落、碰撞、摩擦、堆叠，直到静止或倒塌 → 取接触力 →
  (1) 稳定性：位移/倾倒角/是否滑走；
  (2) 地面压强、接触压强 vs 材料屈服（接触面积由接触点包络估算，点接触不可靠，会标出）；
  (3) 单个梁/板：用接触力+自重求弯矩、应力、挠度（Euler-Bernoulli，沿长轴一维）。
所有单位：scene 用 mm，MuJoCo 用 m/kg/s。"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .scene import MATS, MU

G = 9.81


def _quat_wxyz(rot):
    q = rot.as_quat()  # x y z w
    return [q[3], q[0], q[1], q[2]]


def build_xml(scene, dt=0.002, floor_mu=0.7):
    groups = {}
    order = []
    for i, p in enumerate(scene.parts):
        key = p.group if p.group else f"__{i}"
        if p.fixed:
            key = "__world__"
        if key not in groups:
            groups[key] = []; order.append(key)
        groups[key].append(i)
    xml = [f'<mujoco><option timestep="{dt}" gravity="0 0 -{G}" integrator="implicitfast" cone="elliptic" impratio="20" noslip_iterations="10"/>',
           '<default><geom solref="0.004 1" solimp="0.95 0.99 0.001"/><pair solref="0.004 1" solimp="0.95 0.99 0.001"/></default><worldbody>',
           f'<geom name="__floor__" type="plane" size="20 20 0.1" contype="0" conaffinity="0"/>']

    def geom_xml(i, rel_pos=None, rel_quat=None):
        p = scene.parts[i]
        d = p.dims
        m = p.mass
        if p.kind == "box": g = f'type="box" size="{d[0]/2000} {d[1]/2000} {d[2]/2000}"'
        elif p.kind == "cyl": g = f'type="cylinder" size="{d[0]/2000} {d[1]/2000}"'
        elif p.kind == "tube": g = f'type="cylinder" size="{d[0]/2000} {d[2]/2000}"'
        else: g = f'type="sphere" size="{d[0]/2000}"'
        mu = MU.get(p.mat, 0.5)
        pos = rel_pos if rel_pos is not None else [0, 0, 0]
        quat = rel_quat if rel_quat is not None else [1, 0, 0, 0]
        return (f'<geom name="{p.name}" {g} mass="{m}" contype="0" conaffinity="0" '
                f'pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"/>')

    bodies = []
    for key in order:
        idx = groups[key]
        if key == "__world__":
            for i in idx:
                p = scene.parts[i]
                pos = np.array(p.pos) / 1000
                xml.append(geom_xml(i, pos, _quat_wxyz(p.R)))
            continue
        p0 = scene.parts[idx[0]]
        anchor = np.array(p0.pos) / 1000
        bname = "b_" + key
        bodies.append((bname, idx))
        xml.append(f'<body name="{bname}" pos="{anchor[0]} {anchor[1]} {anchor[2]}"><freejoint/>')
        for i in idx:
            p = scene.parts[i]
            rel = np.array(p.pos) / 1000 - anchor
            xml.append(geom_xml(i, rel, _quat_wxyz(p.R)))
        xml.append('</body>')
    xml.append('</worldbody>')
    # 接触对：摩擦系数取两种材料中较小者（保守，粗估）；固定件之间不用算
    bodyof = {}
    for key in order:
        for i in groups[key]:
            bodyof[i] = key if key != "__world__" else f"__w{i}"
    xml.append('<contact>')
    n = len(scene.parts)
    fm = MU.get(scene.floor_mat, 0.7)
    for i in range(n):
        pi = scene.parts[i]
        if not pi.fixed:
            mu = min(fm, MU.get(pi.mat, 0.5))
            xml.append(f'<pair geom1="__floor__" geom2="{pi.name}" friction="{mu} {mu} 0.005 0.0001 0.0001"/>')
        for j in range(i + 1, n):
            pj = scene.parts[j]
            if bodyof[i] == bodyof[j] or (pi.fixed and pj.fixed):
                continue
            mu = min(MU.get(pi.mat, 0.5), MU.get(pj.mat, 0.5))
            xml.append(f'<pair geom1="{pi.name}" geom2="{pj.name}" friction="{mu} {mu} 0.005 0.0001 0.0001"/>')
    xml.append('</contact></mujoco>')
    return "\n".join(xml), bodies


class Result:
    pass


def simulate(scene, tmax=6.0, fps=30, settle_v=2e-3, dt=0.002, floor_mu=0.7):
    xml, bodies = build_xml(scene, dt, floor_mu)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    res = Result(); res.scene = scene; res.model_xml = xml
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b, _ in bodies]
    frames, times = [], []
    nstep = int(tmax / dt); every = max(1, int(1 / fps / dt))
    quiet = 0; t_settle = None
    acc_forces = None
    for k in range(nstep):
        mujoco.mj_step(m, d)
        if k % every == 0:
            frames.append(np.array([np.r_[d.xpos[b], d.xquat[b]] for b in bid])); times.append(d.time)
        if bodies:
            vmax = max(np.abs(d.cvel[b]).max() for b in bid)
            quiet = quiet + 1 if vmax < settle_v else 0
            if quiet * dt > 0.4 and t_settle is None:
                t_settle = d.time - 0.4
                break
    # 再平均 0.1 s 的接触力
    nav = 50; F = {}
    for _ in range(nav):
        mujoco.mj_step(m, d)
        for ci in range(d.ncon):
            c = d.contact[ci]
            f6 = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1); g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            F.setdefault((g1, g2), []).append((c.pos.copy(), fw.copy()))
    contacts = {}
    for k, lst in F.items():
        pts = {}
        for pos, fw in lst:
            key = tuple(np.round(pos, 3))
            a = pts.setdefault(key, [pos, np.zeros(3), 0]); a[1] += fw; a[2] += 1
        contacts[k] = [(v[0] * 1000, v[1] / nav) for v in pts.values()]   # 对 nav 步时间平均（不是对出现次数平均）   # 位置 mm, 力 N (geom1→geom2)
    res.contacts = contacts
    res.frames = np.array(frames); res.times = np.array(times)
    res.bodies = bodies
    res.t_settle = t_settle
    res.settled = t_settle is not None
    final = np.array([np.r_[d.xpos[b], d.xquat[b]] for b in bid])
    res.final = final
    res.geom_pose = {}
    for gi in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if n and n != "__floor__":
            res.geom_pose[n] = (d.geom_xpos[gi].copy() * 1000, d.geom_xmat[gi].reshape(3, 3).copy())
    return res


# --------------------------------------------------------------------------- 体检
def check(res):
    sc = res.scene
    out = dict(stability=[], bearing=[], beams=[], summary=[])
    # --- 1 稳定性
    init = {}
    for bname, idx in res.bodies:
        p0 = sc.parts[idx[0]]
        init[bname] = (np.array(p0.pos), p0.R)
    for bi, (bname, idx) in enumerate(res.bodies):
        pos0, R0 = init[bname]
        pf = res.final[bi][:3] * 1000
        q = res.final[bi][3:]
        Rf = R.from_quat([q[1], q[2], q[3], q[0]])
        tilt_axis = np.degrees(Rf.magnitude())   # 物体坐标系初始为单位旋转，所以转角就是倾转角
        disp = np.linalg.norm((pf - pos0)[:2])      # 水平位移；竖向下落单独看 dz
        name = ",".join(sc.parts[i].name for i in idx)
        dz = pf[2] - pos0[2]
        status = "稳定"
        if tilt_axis > 25 or disp > 150: status = "倾倒/滑落"
        elif disp > 5 or tilt_axis > 1 or abs(dz) > 5: status = "有位移/下落"
        out["stability"].append(dict(body=name, disp_mm=float(disp), dz_mm=float(dz), tilt_deg=float(tilt_axis), status=status))
    # --- 2 压强
    for (g1, g2), pts in res.contacts.items():
        fn = sum(max(0, -f[2]) if g1 == "__floor__" else 0 for _, f in pts)   # 占位，下面统一算
        P = np.array([p for p, _ in pts]); Fv = np.array([f for _, f in pts])
        Fmag = np.linalg.norm(Fv.sum(axis=0))
        # 法向力：力矢量和的大小（对静力平衡的竖直接触≈竖向分量）
        Fz = abs(Fv[:, 2].sum())
        if Fz < 1: continue
        if len(P) >= 3:
            ext = P.max(axis=0) - P.min(axis=0); ext = np.sort(ext)[::-1]
            area = max(ext[0] * ext[1], 1.0)
        else:
            area = 1.0
        press = Fz / area  # N/mm2 = MPa
        if g1 == "__floor__":
            mats = [MATS[sc.floor_mat]["fy"], MATS[sc.get(g2).mat].get("creep_fy", MATS[sc.get(g2).mat]["fy"])]
            nm1 = "地面"
        else:
            p1, p2 = sc.get(g1), sc.get(g2)
            mats = [p.material.get("creep_fy", p.material["fy"]) for p in (p1, p2)]
            nm1 = g1
        lim = min(mats) / 1e6
        reliable = len(P) >= 3 and area >= 50
        out["bearing"].append(dict(pair=f"{nm1}↔{g2}", F=float(Fz), area=float(area), p=float(press), lim=float(lim),
                                   ok=bool(press <= lim), reliable=bool(reliable), n=len(P)))
    # --- 3 梁
    out["beams"] = [beam_check(res, p) for p in sc.parts if p.kind == "box" and not p.group and not p.fixed]
    out["beams"] = [b for b in out["beams"] if b]
    # 冲击放大（材料力学落重冲击公式，保守上界）：n = 1 + sqrt(1 + 2h/δst)
    drops = {}
    for bi, (bname, idx) in enumerate(res.bodies):
        dz = init[bname][0][2] - res.final[bi][2] * 1000
        for i in idx:
            drops[sc.parts[i].name] = max(0.0, dz)
    for b in out["beams"]:
        hmax = 0.0
        for (g1, g2) in res.contacts:
            if b["name"] in (g1, g2):
                other = g2 if g1 == b["name"] else g1
                hmax = max(hmax, drops.get(other, 0.0))
        b["drop_h"] = hmax
        if hmax > 20 and b["defl"] > 1e-6:
            n = 1 + np.sqrt(1 + 2 * hmax / b["defl"])
            b["n_dyn"] = float(n); b["sigma_dyn"] = float(n * b["sigma"]); b["ok_dyn"] = bool(b["sigma_dyn"] <= b["fy"])
    # 汇总
    bad = [s for s in out["stability"] if s["status"] != "稳定"]
    out["unstable"] = bad
    out["floor"] = floor_pressure(res)
    return out


def floor_pressure(res):
    pts = []
    for (g1, g2), lst in res.contacts.items():
        if g1 == "__floor__":
            pts += [(p, f) for p, f in lst]
    if not pts: return None
    P = np.array([p for p, _ in pts]); Fz = sum(abs(f[2]) for _, f in pts)
    return dict(total_N=float(Fz), n=len(pts))


def beam_check(res, part):
    """一根箱型梁/板：沿长轴分析弯矩。只对"自由体"可靠（接触力即全部外力）。"""
    if part.group: return None
    gp = res.geom_pose.get(part.name)
    if gp is None: return None
    c, Rm = gp
    dims = np.array(part.dims, float)
    ax = int(np.argmax(dims)); L = dims[ax]
    e = Rm[:, ax]                       # 长轴方向(世界)
    if abs(e[2]) > 0.5: return None      # 竖向构件(柱)不当梁算
    others = [i for i in range(3) if i != ax]
    zc = [abs(Rm[2, i]) for i in others]
    hi = others[int(np.argmax(zc))]; bi = others[1 - int(np.argmax(zc))]
    h, b = dims[hi], dims[bi]
    mat = part.material
    E = mat["E"] / 1e6
    I = b * h ** 3 / 12                   # mm4
    W = I / (h / 2)
    # 载荷点
    loads = []   # (s, Fz) s 以长轴方向从中心起, mm
    for (g1, g2), pts in res.contacts.items():
        if part.name not in (g1, g2): continue
        sign = 1 if g2 == part.name else -1   # 力 geom1→geom2：作用在 geom2 上为 +F
        for p, f in pts:
            s = float(np.dot(p - c, e))
            loads.append((s, sign * f[2]))
    if not loads: return None
    w = part.mass * G / L                  # N/mm 向下
    s_grid = np.linspace(-L / 2, L / 2, 401)
    # 从左端起算弯矩 M(s) = Σ_{s_i<s} Fz_i (s - s_i) - w (s+L/2)^2/2
    M = np.zeros_like(s_grid)
    for si, Fi in loads:
        M += np.where(s_grid > si, Fi * (s_grid - si), 0)
    M -= w * (s_grid + L / 2) ** 2 / 2
    Mmax = float(np.abs(M).max()); sig = Mmax / W
    # 挠度：两端最外侧支点处为 0
    ups = sorted((si, Fi) for si, Fi in loads if Fi > 0)
    s0, s1 = -L / 2, L / 2
    if len(ups) >= 2:
        cl = [[ups[0]]]
        for u in ups[1:]:
            (cl[-1] if u[0] - cl[-1][-1][0] < 0.15 * L else cl.append([]) or cl[-1]).append(u)
        cen = [(sum(a * f for a, f in c_) / sum(f for _, f in c_), sum(f for _, f in c_)) for c_ in cl]
        cen = [x for x in cen if x[1] > 0.05 * max(y[1] for y in cen)]
        if len(cen) >= 2:
            s0, s1 = cen[0][0], cen[-1][0]
    kappa = M / (E * I)
    ds = s_grid[1] - s_grid[0]
    th = np.cumsum(kappa) * ds
    wdef = np.cumsum(th) * ds
    # 线性修正使 s0,s1 处为 0
    w0, w1 = np.interp(s0, s_grid, wdef), np.interp(s1, s_grid, wdef)
    wdef = wdef - (w0 + (w1 - w0) * (s_grid - s0) / (s1 - s0))
    inside = (s_grid >= s0) & (s_grid <= s1)
    dmax = float(np.abs(wdef[inside]).max()) if inside.any() else 0.0
    span = float(s1 - s0)
    return dict(name=part.name, L=float(L), span=span, b=float(b), h=float(h), Mmax=Mmax, sigma=float(sig),
                fy=mat["fy"] / 1e6, ok=bool(sig <= mat["fy"] / 1e6), defl=dmax, defl_ratio=span / dmax if dmax > 1e-9 else 1e9,
                s_grid=s_grid, M=M, w=wdef, s0=float(s0), s1=float(s1))


def float_check(scene, T=20.0):
    """液体里每个零件：整体密度 vs 液体密度 → 浮/沉，浮时吃水比例（阿基米德）。scene.fluid=(key,液面mm)。"""
    from ..world.fluids import get_fluid
    if not scene.fluid: return []
    f = get_fluid(scene.fluid[0]); rho_f = f["rho"](T)
    out = []
    for p in scene.parts:
        if p.fixed: continue
        # 管/空心件：用外轮廓体积算浮力(封闭空心管不是这样，这里按"两端开口、水可进入"→ 用材料体积)
        rho = p.material["rho"]
        out.append(dict(name=p.name, rho_part=rho, rho_fluid=float(rho_f), floats=bool(rho < rho_f),
                        draft=float(min(1.0, rho / rho_f)) if rho < rho_f else 1.0,
                        net_N=float((rho - rho_f) * p.volume_mm3() * 1e-9 * G)))
    return out
