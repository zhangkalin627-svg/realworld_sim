"""刚体物理（MuJoCo, Apache-2.0）+ 受力体检。
流程：Scene → MJCF → 让它在重力下落、碰撞、摩擦、堆叠，直到静止或倒塌 → 取接触力 →
  (1) 稳定性：位移/倾倒角/是否滑走；
  (2) 地面压强、接触压强 vs 材料屈服（接触面积由接触点包络估算，点接触不可靠，会标出）；
  (3) 单个梁/板/型材：用接触力+自重求弯矩、应力、挠度（Euler-Bernoulli，沿长轴一维）；
  (4) v0.12 螺栓组校核（接头两侧静力平衡 → 螺栓拉力/剪力）、轴承反力与 L10 寿命、
      赫兹接触应力（球/圆柱）、断裂判据（应力 > 抗拉强度 fu）。
所有单位：scene 用 mm，MuJoCo 用 m/kg/s。
v0.12: 型材零件在 MJCF 里分解成长方体 primitive（质量/惯量/接触都按真实截面），
       轴承用 MuJoCo hinge 关节；铰链反力由刚体运动学反推（不依赖求解器内部数组）。"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .scene import MATS, MU
from . import profile as _prof
from . import joints as _jnt

G = 9.81


def _quat_wxyz(rot):
    q = rot.as_quat()  # x y z w
    return [q[3], q[0], q[1], q[2]]


def _prim_volume(kind, dims):
    d = dims
    if kind == "box": return d[0] * d[1] * d[2]
    if kind == "cyl": return np.pi * (d[0] / 2) ** 2 * d[1]
    if kind == "tube": return np.pi * ((d[0] / 2) ** 2 - (d[1] / 2) ** 2) * d[2]
    if kind == "sphere": return 4 / 3 * np.pi * (d[0] / 2) ** 3
    raise ValueError(kind)


def part_com_local(part):
    """零件局部坐标下的质心（primitive 质量加权；对称件 = 几何中心）。"""
    prims = part.primitives()
    if len(prims) == 1: return np.zeros(3)
    V = np.array([_prim_volume(k, d) for k, d, _ in prims])
    O = np.array([o for _, _, o in prims])
    return (V[:, None] * O).sum(axis=0) / V.sum()


def _prim_inertia_diag(pk, d, m):
    """图元关于自身质心、图元局部坐标（圆柱轴=局部 z）的惯量对角，SI 单位（kg·m²）。
    issue 1929：tube 必须用圆环公式——MuJoCo 只能按实心圆柱几何推惯量，与环形质量不符。"""
    if pk == "box":
        a, b, c = d[0] / 1000, d[1] / 1000, d[2] / 1000
        return np.array([m * (b * b + c * c), m * (a * a + c * c), m * (a * a + b * b)]) / 12
    if pk == "cyl":
        r, L = d[0] / 2000, d[1] / 1000
        return np.array([m * (3 * r * r + L * L) / 12, m * (3 * r * r + L * L) / 12, m * r * r / 2])
    if pk == "tube":
        ro, ri, L = d[0] / 2000, d[1] / 2000, d[2] / 1000
        return np.array([m * (3 * (ro * ro + ri * ri) + L * L) / 12,
                         m * (3 * (ro * ro + ri * ri) + L * L) / 12,
                         m * (ro * ro + ri * ri) / 2])
    r = d[0] / 2000
    return np.array([2 * m * r * r / 5.0] * 3)


def _body_inertial_xml(scene, idx, anchor):
    """body 内含 tube 图元时，全 body 改用解析惯量生成显式 <inertial>（box/cyl/sphere 公式
    与 MuJoCo 几何推导一致，tube 用圆环公式），覆盖 geom 几何推导（issue 1929）。
    无 tube 的 body 保持原有 MuJoCo 几何推导路径，行为不变。"""
    entries = []
    has_tube = False
    for i in idx:
        p = scene.parts[i]
        rel = np.array(p.pos) / 1000 - anchor
        for pk, pd, po in p.primitives():
            m = _prim_volume(pk, pd) * 1e-9 * p.material["rho"]
            pos = rel + p.R.apply(np.array(po, float) / 1000)
            entries.append((m, pos, p.R.as_matrix(), _prim_inertia_diag(pk, pd, m)))
            has_tube = has_tube or pk == "tube"
    if not has_tube:
        return None
    M = sum(e[0] for e in entries)
    if M <= 0:
        return None
    com = sum(e[0] * e[1] for e in entries) / M
    I = np.zeros((3, 3))
    for m, pos, Rm, Id in entries:
        I += Rm @ np.diag(Id) @ Rm.T
        dv = pos - com
        I += m * ((dv @ dv) * np.eye(3) - np.outer(dv, dv))
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]
    qx, qy, qz, qw = R.from_matrix(V).as_quat()
    return ('<inertial pos="%.9g %.9g %.9g" quat="%.9g %.9g %.9g %.9g" mass="%.9g" diaginertia="%.9g %.9g %.9g"/>'
            % (com[0], com[1], com[2], qw, qx, qy, qz, M, w[0], w[1], w[2]))


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
    hinge_of = {}                       # body key -> hinge spec
    for h in scene.hinges:
        for i, p in enumerate(scene.parts):
            if p.name == h["part"]:
                for key in order:
                    if i in groups[key] and key != "__world__":
                        hinge_of[key] = h
    xml = [f'<mujoco><option timestep="{dt}" gravity="0 0 -{G}" integrator="implicitfast" cone="elliptic" impratio="20" noslip_iterations="10"/>',
           '<default><geom solref="0.004 1" solimp="0.95 0.99 0.001"/><pair solref="0.004 1" solimp="0.95 0.99 0.001"/></default><worldbody>',
           f'<geom name="__floor__" type="plane" size="20 20 0.1" contype="0" conaffinity="0"/>']

    geom_part = {}                      # geom 名 -> part 名（型材分解体的 "#k" 后缀 geom 也映射回本零件）

    def geom_xml(i, rel_pos=None, rel_quat=None):
        p = scene.parts[i]
        mu = MU.get(p.mat, 0.5)
        out = []
        prims = p.primitives()
        for k, (pk, pd, po) in enumerate(prims):
            d = pd
            m = _prim_volume(pk, pd) * 1e-9 * p.material["rho"]
            if pk == "box": g = f'type="box" size="{d[0]/2000} {d[1]/2000} {d[2]/2000}"'
            elif pk == "cyl": g = f'type="cylinder" size="{d[0]/2000} {d[1]/2000}"'
            elif pk == "tube": g = f'type="cylinder" size="{d[0]/2000} {d[2]/2000}"'
            else: g = f'type="sphere" size="{d[0]/2000}"'
            gname = p.name if k == 0 else f"{p.name}#{k}"
            geom_part[gname] = p.name
            pos = np.array(rel_pos if rel_pos is not None else [0, 0, 0]) + p.R.apply(np.array(po) / 1000)
            quat = rel_quat if rel_quat is not None else [1, 0, 0, 0]
            out.append(f'<geom name="{gname}" {g} mass="{m}" contype="0" conaffinity="0" '
                       f'pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"/>')
        return "\n".join(out)

    bodies = []
    body_anchor = {}
    for key in order:
        idx = groups[key]
        if key == "__world__":
            for i in idx:
                p = scene.parts[i]
                pos = np.array(p.pos) / 1000
                xml.append(geom_xml(i, pos, _quat_wxyz(p.R)))
            continue
        h = hinge_of.get(key)
        p0 = scene.parts[idx[0]]
        anchor = np.array(h["anchor"], float) / 1000 if h else np.array(p0.pos) / 1000
        body_anchor[key] = anchor
        bname = "b_" + key
        bodies.append((bname, idx))
        xml.append(f'<body name="{bname}" pos="{anchor[0]} {anchor[1]} {anchor[2]}">')
        _inert = _body_inertial_xml(scene, idx, anchor)
        if _inert:
            xml.append(_inert)
        if h:
            ax = np.array(h["axis"], float); ax = ax / np.linalg.norm(ax)
            damp = h.get("damp", 0.0)
            xml.append(f'<joint name="j_{bname}" type="hinge" axis="{ax[0]} {ax[1]} {ax[2]}" pos="0 0 0" damping="{damp}" limited="false"/>')
        else:
            xml.append('<freejoint/>')
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

    def gnames(p):
        # 多图元零件（型材分解为 翼缘/腹板 等多个 box）每个图元是独立 geom，接触对必须逐一列出
        return [p.name if k == 0 else f"{p.name}#{k}" for k in range(len(p.primitives()))]

    for i in range(n):
        pi = scene.parts[i]
        if not pi.fixed:
            mu = min(fm, MU.get(pi.mat, 0.5))
            for g2 in gnames(pi):
                xml.append(f'<pair geom1="__floor__" geom2="{g2}" friction="{mu} {mu} 0.005 0.0001 0.0001"/>')
        for j in range(i + 1, n):
            pj = scene.parts[j]
            if bodyof[i] == bodyof[j] or (pi.fixed and pj.fixed):
                continue
            mu = min(MU.get(pi.mat, 0.5), MU.get(pj.mat, 0.5))
            for g1 in gnames(pi):
                for g2 in gnames(pj):
                    xml.append(f'<pair geom1="{g1}" geom2="{g2}" friction="{mu} {mu} 0.005 0.0001 0.0001"/>')
    xml.append('</contact></mujoco>')
    return "\n".join(xml), bodies, geom_part


class Result:
    pass


def simulate(scene, tmax=6.0, fps=30, settle_v=2e-3, dt=0.002, floor_mu=0.7):
    xml, bodies, geom_part = build_xml(scene, dt, floor_mu)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    res = Result(); res.scene = scene; res.model_xml = xml; res.geom_part = geom_part
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b, _ in bodies]
    # v0.12 轴承铰链：初速（spin_rpm）与运动学反推所需的句柄
    hinges = []
    for bi, (bname, idx) in enumerate(bodies):
        spec = next((h for h in scene.hinges if h["part"] in [scene.parts[i].name for i in idx]), None)
        if spec is None: continue
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "j_" + bname)
        dof = int(m.jnt_dofadr[jid])
        ax = np.array(spec["axis"], float); ax /= np.linalg.norm(ax)
        if spec.get("spin_rpm"):
            d.qvel[dof] = float(spec["spin_rpm"]) * 2 * np.pi / 60
        hinges.append(dict(spec=spec, bi=bi, dof=dof, axis_local=ax, body_id=bid[bi]))
    frames, times = [], []
    nstep = int(tmax / dt); every = max(1, int(1 / fps / dt))
    quiet = 0; t_settle = None
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
    # 再平均 nav 步：接触力 + 铰链反力（运动学反推，逐步记录）
    nav = 50; F = {}
    hinge_series = {h["spec"]["part"]: [] for h in hinges}
    for _ in range(nav):
        mujoco.mj_step(m, d)
        body_contact = {}
        for ci in range(d.ncon):
            c = d.contact[ci]
            f6 = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1); g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            F.setdefault((g1, g2), []).append((c.pos.copy(), fw.copy()))
            # 按 geom→part→body 归集接触力（作用在 geom2 上为 +f，geom1 上为 -f）
            for gg, sgn in ((g1, -1.0), (g2, 1.0)):
                pn = geom_part.get(gg)
                if pn is None: continue
                body_contact[pn] = body_contact.get(pn, np.zeros(3)) + sgn * fw
        for h in hinges:
            b = h["body_id"]
            omega = d.cvel[b][:3]
            ax_w = d.xmat[b].reshape(3, 3) @ h["axis_local"]
            alpha = ax_w * d.qacc[h["dof"]]
            r = d.xipos[b] - d.xpos[b]
            a_com = np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
            mss = m.body_mass[b]
            Fc = mss * (a_com - np.array([0, 0, -G]))
            for i in bodies[h["bi"]][1]:
                Fc -= body_contact.get(scene.parts[i].name, np.zeros(3))
            hinge_series[h["spec"]["part"]].append(Fc.copy())
    contacts = {}
    for k, lst in F.items():
        pts = {}
        for pos, fw in lst:
            key = tuple(np.round(pos, 3))
            a = pts.setdefault(key, [pos, np.zeros(3), 0]); a[1] += fw; a[2] += 1
        contacts[k] = [(v[0] * 1000, v[1] / nav) for v in pts.values()]   # 对 nav 步时间平均（不是对出现次数平均）   # 位置 mm, 力 N (geom1→geom2)
    res.contacts = contacts
    res.hinge_series = {k: np.array(v) for k, v in hinge_series.items()}
    res.frames = np.array(frames); res.times = np.array(times)
    res.bodies = bodies
    res.t_settle = t_settle
    res.settled = t_settle is not None
    final = np.array([np.r_[d.xpos[b], d.xquat[b]] for b in bid])
    res.final = final
    # 每个零件的世界位姿（由所属刚体变换得到；型材分解不改变 body 划分）
    res.part_pose = {}
    for bi, (bname, idx) in enumerate(bodies):
        key = bname[2:]
        # body 的锚点(初始 body pos) = anchor；part 初始 rel = p.pos/1000 - anchor
        anchor = None
        for h in scene.hinges:
            if h["part"] in [scene.parts[i].name for i in idx]:
                anchor = np.array(h["anchor"], float) / 1000
        if anchor is None:
            anchor = np.array(scene.parts[idx[0]].pos) / 1000
        xq = d.xquat[bid[bi]]
        Rb = R.from_quat([xq[1], xq[2], xq[3], xq[0]]).as_matrix()
        for i in idx:
            p = scene.parts[i]
            rel = np.array(p.pos) / 1000 - anchor
            pw = d.xpos[bid[bi]] + Rb @ rel
            res.part_pose[p.name] = (pw * 1000, Rb @ p.R.as_matrix())
    return res


# --------------------------------------------------------------------------- 体检
def check(res):
    sc = res.scene
    out = dict(stability=[], bearing=[], beams=[], summary=[])
    gp = res.geom_part
    pn = lambda g: gp.get(g, g)                      # geom 名 → part 名（去掉型材 "#k" 分解）
    # --- 1 稳定性（铰接 body 按设计旋转，不判倾倒）
    hinged_names = {h["part"] for h in sc.hinges}
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
        if any(sc.parts[i].name in hinged_names for i in idx):
            out["stability"].append(dict(body=name, disp_mm=float(disp), dz_mm=float(dz), tilt_deg=float(tilt_axis), status="铰接"))
            continue
        status = "稳定"
        if tilt_axis > 25 or disp > 150: status = "倾倒/滑落"
        elif disp > 5 or tilt_axis > 1 or abs(dz) > 5: status = "有位移/下落"
        out["stability"].append(dict(body=name, disp_mm=float(disp), dz_mm=float(dz), tilt_deg=float(tilt_axis), status=status))
    # --- 2 压强
    for (g1, g2), pts in res.contacts.items():
        P = np.array([p for p, _ in pts]); Fv = np.array([f for _, f in pts])
        Fz = abs(Fv[:, 2].sum())
        if Fz < 1: continue
        if len(P) >= 3:
            ext = P.max(axis=0) - P.min(axis=0); ext = np.sort(ext)[::-1]
            area = max(ext[0] * ext[1], 1.0)
        else:
            area = 1.0
        press = Fz / area  # N/mm2 = MPa
        if g1 == "__floor__":
            mats = [MATS[sc.floor_mat]["fy"], MATS[sc.get(pn(g2)).mat].get("creep_fy", MATS[sc.get(pn(g2)).mat]["fy"])]
            nm1 = "地面"
        else:
            p1, p2 = sc.get(pn(g1)), sc.get(pn(g2))
            mats = [p.material.get("creep_fy", p.material["fy"]) for p in (p1, p2)]
            nm1 = pn(g1)
        lim = min(mats) / 1e6
        reliable = len(P) >= 3 and area >= 50
        out["bearing"].append(dict(pair=f"{nm1}↔{pn(g2)}", F=float(Fz), area=float(area), p=float(press), lim=float(lim),
                                   ok=bool(press <= lim), reliable=bool(reliable), n=len(P)))
    # --- 3 梁/型材
    out["beams"] = [beam_check(res, p) for p in sc.parts if p.kind in BEAM_KINDS and not p.group and not p.fixed]
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
            if b["name"] in (pn(g1), pn(g2)):
                other = pn(g2) if pn(g1) == b["name"] else pn(g1)
                hmax = max(hmax, drops.get(other, 0.0))
        b["drop_h"] = hmax
        if hmax > 20 and b["defl"] > 1e-6:
            n = 1 + np.sqrt(1 + 2 * hmax / b["defl"])
            b["n_dyn"] = float(n); b["sigma_dyn"] = float(n * b["sigma"]); b["ok_dyn"] = bool(b["sigma_dyn"] <= b["fy"])
    # --- 4 v0.12 螺栓组校核
    out["bolts"] = [joint_check(res, j) for j in sc.joints]
    out["bolts"] = [b for b in out["bolts"] if b]
    # --- 5 v0.12 轴承反力与 L10 寿命
    out["hinges"] = []
    for h in sc.hinges:
        ser = res.hinge_series.get(h["part"])
        if ser is None or len(ser) == 0: continue
        rpm = abs(float(h.get("spin_rpm", 0.0)))
        ax = np.array(h["axis"], float); ax /= np.linalg.norm(ax)
        F_mean = ser.mean(axis=0)
        rad = float(np.linalg.norm(F_mean - np.dot(F_mean, ax) * ax))
        amp = float(np.linalg.norm(ser - F_mean, axis=1).max())      # 旋转不平衡力幅值
        P_eq = rad + amp                                              # 当量动载荷：静载 + 不平衡幅值（保守）
        rec = dict(part=h["part"], bearing=h.get("bearing"), rpm=rpm,
                   F_mean=[float(x) for x in F_mean], F_radial=rad, F_unbalance=amp, P=float(P_eq))
        if h.get("bearing") and rpm > 0:
            rec["life"] = _jnt.bearing_life(h["bearing"], P_eq, rpm)
        elif h.get("bearing"):
            rec["life"] = dict(S0=float(_jnt.BEARINGS[h["bearing"]]["C0"] / max(P_eq, 1e-9)),
                               C0=_jnt.BEARINGS[h["bearing"]]["C0"], P=float(P_eq), rpm=0.0, code=h["bearing"])
        out["hinges"].append(rec)
    # --- 6 v0.12 赫兹接触应力（球/圆柱 vs 平面类零件）
    out["hertz"] = hertz_check(res)
    # 汇总
    bad = [s for s in out["stability"] if s["status"] != "稳定"]
    out["unstable"] = bad
    out["floor"] = floor_pressure(res)
    out["broken"] = [b for b in out["beams"] if b.get("status") == "断裂"]
    out["yielded"] = [b for b in out["beams"] if b.get("status") == "屈服"]
    return out


def floor_pressure(res):
    pts = []
    for (g1, g2), lst in res.contacts.items():
        if g1 == "__floor__":
            pts += [(p, f) for p, f in lst]
    if not pts: return None
    P = np.array([p for p, _ in pts]); Fz = sum(abs(f[2]) for _, f in pts)
    return dict(total_N=float(Fz), n=len(pts))


def joint_check(res, j):
    """螺栓组校核：side 一侧的所有外力（重力+接触力）对接头中心求和 → F, M → 螺栓组分配。"""
    sc = res.scene
    gp = res.geom_part
    side = j.get("side") or [j["part"]]
    try:
        parts = [sc.get(nm) for nm in side]
    except KeyError:
        return None
    center = np.array(j["center"], float)
    F = np.zeros(3); M = np.zeros(3)
    for p in parts:                                   # 重力（质心处）
        com = np.array(p.pos) + p.R.apply(part_com_local(p))
        Fg = np.array([0, 0, -p.mass * G])
        F += Fg; M += np.cross(com - center, Fg)
    for (g1, g2), pts in res.contacts.items():        # 接触力（只取作用在 side 零件上的）
        for gg, sgn in ((g1, -1.0), (g2, 1.0)):
            if gp.get(gg, gg) not in side: continue
            for pos, f in pts:
                F += sgn * f; M += np.cross(pos - center, sgn * f)
    if j.get("weld"):
        w = j["weld"]
        fy = sc.get(j["part"]).material["fy"] / 1e6
        r = _jnt.weld_check(F, M, j["normal"], w["legs"], w["hf"], fy, j.get("e1"))
        r.update(base=j["base"], part=j["part"], side=side, weld=True,
                 F=[float(x) for x in F], M=[float(x) for x in M])
        return r
    r = _jnt.bolt_group_check(F, M, center, j["normal"], j["coords"], j["size"], j["grade"], j.get("e1"))
    r.update(base=j["base"], part=j["part"], side=side,
             F=[float(x) for x in F], M=[float(x) for x in M])
    return r


def hertz_check(res):
    """球/圆柱零件的赫兹接触应力（与平面类零件 box/型材/地面接触时）。点接触给 p0，线接触给 p0。
    赫兹压力常超过屈服而不失效（三向受压），这里只做报告并与屈服并列表述，不断言破坏。"""
    sc = res.scene
    gp = res.geom_part
    out = []
    for (g1, g2), pts in res.contacts.items():
        n1, n2 = gp.get(g1, g1), gp.get(g2, g2)
        if g1 == "__floor__": n1 = None
        pair_parts = []
        for nm in (n1, n2):
            if nm is None:
                pair_parts.append(("__floor__", MATS[sc.floor_mat]))
            else:
                p = sc.get(nm); pair_parts.append((p, p.material))
        (o1, m1), (o2, m2) = pair_parts
        Fz = abs(sum(f[2] for _, f in pts))
        if Fz < 1: continue
        for (oth, mo), (self_p, ms) in (((o1, m1), (o2, m2)), ((o2, m2), (o1, m1))):
            if not hasattr(self_p, "kind"): continue
            if self_p.kind == "sphere":
                h = _jnt.hertz_point(Fz, self_p.dims[0] / 2, ms["E"], ms["nu"], mo["E"], mo["nu"])
                out.append(dict(kind="点接触", part=self_p.name, F=float(Fz), p0=h["p0"], a=h["a"],
                                with_=(oth if isinstance(oth, str) else oth.name),
                                fy=ms["fy"] / 1e6, note="赫兹点接触；p0>屈服不一定失效（三向应力）"))
            elif self_p.kind in ("cyl", "tube"):
                # 只在线接触姿态（轴线水平）时有意义
                pp = res.part_pose.get(self_p.name)
                if pp is None: continue
                axidx = 1 if self_p.kind == "cyl" else 2
                e = pp[1][:, axidx]
                if abs(e[2]) > 0.3: continue
                Lc = self_p.dims[axidx]
                h = _jnt.hertz_line(Fz, Lc, self_p.dims[0] / 2, ms["E"], ms["nu"], mo["E"], mo["nu"])
                out.append(dict(kind="线接触", part=self_p.name, F=float(Fz), p0=h["p0"], b=h["b"],
                                with_=(oth if isinstance(oth, str) else oth.name),
                                fy=ms["fy"] / 1e6, note="赫兹线接触；按全长接触估算"))
    return out


BEAM_KINDS = ("box", "tube", "ibeam", "channel", "angle", "sqtube")


def _beam_section(part, Rm):
    """返回 (ax, e, h, b, I, W, axis_note)：梁长轴索引、长轴单位向量(世界)、截面高宽、弯曲惯性矩/系数。
    弯曲轴取世界水平方向对应的形心主轴（竖向载荷 → 绕水平形心轴弯）。"""
    d = np.array(part.dims, float)
    if part.kind == "box":
        ax = int(np.argmax(d))
        e = Rm[:, ax]
        others = [i for i in range(3) if i != ax]
        zc = [abs(Rm[2, i]) for i in others]
        hi = others[int(np.argmax(zc))]; bi = others[1 - int(np.argmax(zc))]
        h, b = d[hi], d[bi]
        I = b * h ** 3 / 12; W = I / (h / 2)
        return ax, e, h, b, I, W, "矩形 b·h³/12"
    if part.kind == "tube":
        ax = 2
        e = Rm[:, ax]
        Do, Di = d[0], d[1]
        I = np.pi * (Do ** 4 - Di ** 4) / 64; W = I / (Do / 2)
        return ax, e, Do, Do, I, W, "圆管 π(D⁴-d⁴)/64"
    # 型材：长度沿局部 X（局部坐标的第 0 轴）；截面在局部 Y-Z（H 沿局部 Z，B 沿局部 Y）
    ax = len(d) - 1                     # dims 里的长度槽位（取 L 用）
    e = Rm[:, 0]
    sm = _prof.section_moments(part.kind, d)
    if abs(Rm[2, 2]) >= abs(Rm[2, 1]):     # 局部 Z(H) 更接近竖直 → 绕局部 Y(强轴)弯
        return ax, e, d[0], d[1], sm["Iy"], sm["Wy"], "型材强轴 Iy"
    return ax, e, d[0], d[1], sm["Iz"], sm["Wz"], "型材弱轴 Iz"


def beam_check(res, part):
    """一根梁/板/型材：沿长轴分析弯矩。只对"自由体"可靠（接触力即全部外力）。
    v0.12: 支持型材（真实截面 I/W）与圆管；应力分级：σ≤fy 正常 / fy<σ≤fu 屈服(塑性变形) / σ>fu 断裂。"""
    if part.group: return None
    gp = res.part_pose.get(part.name)
    if gp is None: return None
    c, Rm = gp
    dims = np.array(part.dims, float)
    ax, e, h, b, I, W, note = _beam_section(part, Rm)
    L = dims[ax]
    if abs(e[2]) > 0.5: return None      # 竖向构件(柱)不当梁算
    mat = part.material
    E = mat["E"] / 1e6
    gp2 = res.geom_part
    # 载荷点
    loads = []   # (s, Fz) s 以长轴方向从中心起, mm
    for (g1, g2), pts in res.contacts.items():
        n1, n2 = gp2.get(g1, g1), gp2.get(g2, g2)
        if part.name not in (n1, n2): continue
        sign = 1 if n2 == part.name else -1   # 力 geom1→geom2：作用在 geom2 上为 +F
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
    fy = mat["fy"] / 1e6; fu = mat.get("fu", mat["fy"]) / 1e6
    status = "正常" if sig <= fy else ("屈服" if sig <= fu else "断裂")
    return dict(name=part.name, L=float(L), span=span, b=float(b), h=float(h), Mmax=Mmax, sigma=float(sig),
                fy=fy, fu=float(fu), status=status, section=note,
                ok=bool(sig <= fy), defl=dmax, defl_ratio=span / dmax if dmax > 1e-9 else 1e9,
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
