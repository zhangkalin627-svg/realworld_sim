"""v0.13 断裂后果演示：把 check() 判为"断裂"的梁在最大弯矩截面处分成两段，重新仿真看后果
（塌下来、被支座托住、还是弹开）。分裂位置取 |M| 最大处（s_grid 上），两段保持原姿态，
断面粗糙 → 不粘合（两段是独立刚体）。只演示静力后果，不含碎片飞散与裂纹路径。"""
import numpy as np
from .scene import Scene
from .physics import simulate, check


def split_scene(res, chk, names=None, gap=1.0):
    """返回一个新 Scene：被判断裂的梁换成两段（断口在 |M|max 处，留 gap mm 缝）。
    型材也支持（两段同截面型材）。其余零件原样复制（用世界位姿）。"""
    broken = [b for b in chk["beams"] if b.get("status") == "断裂"]
    if names is not None:
        broken = [b for b in broken if b["name"] in names]
    sc = res.scene
    out = Scene(sc.name + "（断裂后）")
    out.fluid, out.floor_mat = sc.fluid, sc.floor_mat
    out.joints = [dict(j) for j in sc.joints]
    out.hinges = [dict(h) for h in sc.hinges]
    split_names = {b["name"]: b for b in broken}
    for p in sc.parts:
        if p.name not in split_names:
            c, Rm = res.part_pose.get(p.name, (np.array(p.pos), p.R.as_matrix()))
            from scipy.spatial.transform import Rotation as _R
            q = _R.from_matrix(Rm).as_euler("xyz", degrees=True)
            out.add(p.name, p.kind, list(p.dims), p.mat, pos=list(c), rot=list(q),
                    fixed=p.fixed, group=p.group, color=p.color)
            continue
        b = split_names[p.name]
        c, Rm = res.part_pose[p.name]
        from scipy.spatial.transform import Rotation as _R
        dims = list(p.dims)
        L = dims[-1] if p.kind not in ("box", "tube") else dims[int(np.argmax(dims))] if p.kind == "box" else dims[2]
        # 断口位置（沿长轴，离中心 s*）
        s_star = float(b["s_grid"][int(np.argmax(np.abs(b["M"])))])
        e = Rm[:, 0] if p.kind not in ("box", "tube") else (Rm[:, int(np.argmax(dims))] if p.kind == "box" else Rm[:, 2])
        for sgn, tag in ((-1, "左段"), (1, "右段")):
            L2 = (L / 2 + sgn * s_star) - gap / 2 if sgn < 0 else (L / 2 - sgn * s_star) - gap / 2
            # 左段: 从 -L/2 到 s_star-gap/2; 右段: 从 s_star+gap/2 到 L/2
            if sgn < 0:
                a0, a1 = -L / 2, s_star - gap / 2
            else:
                a0, a1 = s_star + gap / 2, L / 2
            Lseg = a1 - a0
            if Lseg < 1.0: continue
            c_seg_local = (a0 + a1) / 2
            c_seg = c + e * c_seg_local
            dims2 = list(dims)
            if p.kind == "box": dims2[int(np.argmax(dims))] = Lseg
            elif p.kind == "tube": dims2[2] = Lseg
            else: dims2[-1] = Lseg
            q = _R.from_matrix(Rm).as_euler("xyz", degrees=True)
            out.add(f"{p.name}-{tag}", p.kind, dims2, p.mat, pos=list(c_seg), rot=list(q), color=p.color)
    return out


def break_and_rerun(res, chk, tmax=4.0, names=None):
    """断裂 → 分裂 → 再仿真 → 再体检。返回 (新scene, 新res, 新chk)。没有断裂件时返回 None。"""
    if not chk.get("broken"):
        return None
    s2 = split_scene(res, chk, names)
    r2 = simulate(s2, tmax=tmax)
    c2 = check(r2)
    return s2, r2, c2
