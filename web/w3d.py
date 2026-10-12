"""三维装配世界 Web 模块 (v0.18)：预设场景 → MuJoCo 仿真 → {零件 mesh, 逐帧位姿, 受力体检}，供 three.js 查看器。
数据约定：长度一律 mm（MuJoCo 内部是 m，×1000）；位姿四元数 xyzw（three.js 约定）；帧率 15 fps。"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from .modules import _clean

MAT_COLOR = {"Q235": 0x8b9bb4, "45": 0x7f8fa8, "6061": 0xc0c8d0, "6063": 0xc0c8d0, "SUS304": 0xaab4c0,
             "2Cr13": 0x97a3b2, "Cu": 0xb87333, "Brass": 0xc8a24a, "PP": 0xe8e8e8, "POM": 0xf2f2f2,
             "PU": 0xd8cf7a, "PC": 0xbfe0ea, "MIX": 0xb0b0b0, "wood": 0xa5723a, "concrete": 0x9a9a9a,
             "glass": 0xaed6f1, "rubber": 0x3a3a3a, "ice": 0xcfe8f5}


def _T(pos_mm, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    T[:3, 3] = pos_mm
    return T


def run_world3d(params, progress):
    from rw.world3d import presets
    from rw.world3d.physics import simulate, check
    name = params.get("preset") or next(iter(presets.PRESETS))
    if name not in presets.PRESETS:
        raise ValueError(f"未知预设 {name}")
    tmax = min(float(params.get("tmax", 8.0)), 15.0)
    scene = presets.PRESETS[name]()
    progress(0.05)
    res = simulate(scene, tmax=tmax, fps=15)
    progress(0.7)
    chk = check(res)
    progress(0.85)

    meshes = {}
    for p in scene.parts:
        m = p.local_mesh()
        meshes[p.name] = {"v": [round(float(x), 3) for x in m.vertices.reshape(-1)],
                          "f": [int(x) for x in m.faces.reshape(-1)]}

    # 逐帧零件位姿：body 帧 × 零件在体内的初始相对位姿（零件可能与其它零件焊成同一 body）
    rel = {}
    for bi, (bname, idx) in enumerate(res.bodies):
        Tb0i = np.linalg.inv(_T(res.frames[0][bi][:3] * 1000, res.frames[0][bi][3:]))
        for i in idx:
            p = scene.parts[i]
            Tp0 = np.eye(4); Tp0[:3, :3] = p.R.as_matrix(); Tp0[:3, 3] = p.pos
            rel[(bi, i)] = Tb0i @ Tp0
    poses = {p.name: [] for p in scene.parts}
    for f in range(len(res.frames)):
        for bi, (bname, idx) in enumerate(res.bodies):
            Tb = _T(res.frames[f][bi][:3] * 1000, res.frames[f][bi][3:])
            for i in idx:
                T = Tb @ rel[(bi, i)]
                q = R.from_matrix(T[:3, :3]).as_quat()      # xyzw
                poses[scene.parts[i].name].append(
                    [round(float(v), 2) for v in T[:3, 3]] + [round(float(v), 4) for v in q])
    # 固定在地、不属于任何仿真 body 的零件：用初始位姿补齐所有帧（查看器要求每个零件都有完整位姿序列）
    for p in scene.parts:
        if not poses[p.name]:
            Tp0 = np.eye(4); Tp0[:3, :3] = p.R.as_matrix(); Tp0[:3, 3] = p.pos
            q0 = R.from_matrix(Tp0[:3, :3]).as_quat()
            row = [round(float(v), 2) for v in Tp0[:3, 3]] + [round(float(v), 4) for v in q0]
            poses[p.name] = [row] * len(res.frames)
    progress(0.95)

    # 零件体检状态（查看器着色）：梁/型材 正常/屈服/断裂；其余看稳定性
    status = {}
    for b in chk["beams"]:
        status[b["name"]] = b.get("status", "正常")
    stab_by_body = {s["body"]: s["status"] for s in chk["stability"]}
    for bname, idx in res.bodies:
        nm = ",".join(scene.parts[i].name for i in idx)
        for i in idx:
            p = scene.parts[i]
            if p.name not in status:
                status[p.name] = stab_by_body.get(nm, "稳定")

    mtot, com = scene.total_mass()
    lines = [f"{scene.name}：{len(scene.parts)} 个零件，总重 {mtot:.1f} kg；"
             f"{len(res.frames)} 帧（{res.times[-1]:.2f} s）" + (f"，静止于 t={res.t_settle:.2f} s" if res.settled else "，未完全静止")]
    for s in chk["stability"]:
        if s["status"] != "稳定":
            lines.append(f"· {s['body']}：{s['status']}（水平位移 {s['disp_mm']:.0f} mm，倾转 {s['tilt_deg']:.1f}°，下落 {s['dz_mm']:.0f} mm）")
    for b in chk["beams"]:
        mark = {"正常": "✓", "屈服": "⚠", "断裂": "💥"}.get(b.get("status"), "·")
        lines.append(f"{mark} 梁 {b['name']}：{b.get('status','?')}，σ={b['sigma']:.0f} MPa（fy={b['fy']:.0f}），挠度 {b['defl']:.1f} mm")
    for j in chk["bolts"]:
        if j.get("weld"):
            lines.append(f"· 焊缝 {j['part']}↔{j['base']}：" + ("合格" if j.get("ok", True) else "不合格"))
        else:
            lines.append(f"· 螺栓组 {j['part']}↔{j['base']}（{j['size']} {j['grade']}）：最大利用率 {j['worst']:.2f}"
                         + ("（合格）" if j["ok"] else "（超许用！）"))
    for h in chk["hinges"]:
        if h.get("life") and h["life"].get("L10h"):
            lines.append(f"· 轴承 {h.get('bearing')}（{h['part']}）：径向力 {h['F_radial']:.0f} N，L10 ≈ {h['life']['L10h']:.0f} h")
    for hz in chk.get("hertz", []):
        lines.append(f"· 赫兹{hz['kind']} {hz['part']}↔{hz['with_']}：p0={hz['p0']:.0f} MPa（屈服 {hz['fy']:.0f}）")
    fl = chk.get("floor")
    if fl:
        lines.append(f"· 地面总压力 {fl['total_N']:.0f} N（{fl['n']} 个接触点）")
    warnings = [f"梁 {b['name']} 断裂！（GUI ⑨页可演示断裂后果）" for b in chk["broken"]] + \
               [f"梁 {b['name']} 屈服（塑性变形）" for b in chk["yielded"]]
    viewer = {"name": scene.name,
              "parts": [{"name": p.name, "mat": p.mat, "color": MAT_COLOR.get(p.mat, 0xb0b0b0),
                         "fixed": bool(p.fixed), "glass": p.mat in ("glass", "ice", "PC")} for p in scene.parts],
              "meshes": meshes, "poses": poses,
              "times": [round(float(t), 3) for t in res.times], "status": status}
    progress(1.0)
    return _clean({"lines": lines, "warnings": warnings, "plots": [], "viewer": viewer, "raw": {}})
