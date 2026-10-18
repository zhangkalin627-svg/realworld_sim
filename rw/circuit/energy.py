"""能量守恒自检（对标改进 N7）：电源输出能量 = 电阻耗散 + 储能变化 + 其余器件耗散。
全部是仿真结果自检，不是实测。思路来自对仿真工具"能量核算"功能的对标，代码为自写。

规则：
- 只有 R/C/L(+K)/V 源的线性网络：残差必须≈0，否则黄牌（说明步长或积分有问题）。
- 含二极管/开关/MOS/行为源/子电路时，这些器件的耗散未单独算，残差=它们的总耗散，必须 ≥ 0
  （无源器件不会凭空产生能量）；残差明显为负则黄牌。
- 含电流源时本检查不适用（返回"未核"）。
"""
import numpy as np
from .spice import parse_value


def _trapz(t, y):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


def _get(w, key):
    try:
        return np.asarray(w[key], dtype=float)
    except Exception:
        return None


def _v(w, n):
    return np.zeros_like(np.asarray(w["time"], dtype=float)) if str(n) == "0" else _get(w, "v(%s)" % n)


class _E:
    """导入网表的元件是原始文本行（nodes 为空），这里统一成 name/nodes/value。"""
    def __init__(self, e):
        self.name, self.kind = e.name, e.kind
        if e.nodes or e.kind == "K":
            self.nodes, self.value = e.nodes, e.value
            if e.kind == "K" and not self.value:
                self.value = " ".join(e.line().split()[1:])
        else:
            tok = e.line().split()
            self.nodes = tok[1:3]; self.value = tok[3] if len(tok) > 3 else ""
        self.line = e.line


def energy_check(circuit, w, tol=0.02):
    t = np.asarray(w["time"], dtype=float)
    out = {"status": "未核", "notes": [], "e_src": 0.0, "e_R": 0.0, "d_stored": 0.0, "residual": None, "others": {}}
    inds, ks, caps, res, srcs = {}, [], [], [], []
    for e in map(_E, circuit.elements):
        k = e.kind
        if k == "R":
            res.append(e)
        elif k == "C":
            caps.append(e)
        elif k == "L":
            inds[e.name.lower()] = e
        elif k == "K":
            ks.append(e)
        elif k == "V":
            srcs.append(e)
        elif k == "I":
            out["notes"].append("含电流源 %s，能量核算不适用" % e.name); return out
        elif k in "DQMSEGBX":
            out["others"][k] = out["others"].get(k, 0) + 1
        elif k == ".":
            pass
    # 源
    for e in srcs:
        i = _get(w, "i(%s)" % e.name)
        vp, vm = _v(w, e.nodes[0]), _v(w, e.nodes[1])
        if i is None or vp is None or vm is None:
            out["notes"].append("源 %s 的电流/电压缺失" % e.name); return out
        out["e_src"] += _trapz(t, -(vp - vm) * i)
    # 电阻
    for e in res:
        r = parse_value(e.value)
        if not r:
            out["others"]["R(参数化)"] = out["others"].get("R(参数化)", 0) + 1; continue
        a, b = _v(w, e.nodes[0]), _v(w, e.nodes[1])
        if a is None or b is None:
            out["notes"].append("电阻 %s 节点电压缺失" % e.name); return out
        out["e_R"] += _trapz(t, (a - b) ** 2 / r)
    # 储能（始末差）
    d = 0.0
    for e in caps:
        c = parse_value(e.value)
        if not c:
            out["notes"].append("电容 %s 值无法解析" % e.name); return out
        a, b = _v(w, e.nodes[0]), _v(w, e.nodes[1])
        if a is None or b is None:
            out["notes"].append("电容 %s 节点电压缺失" % e.name); return out
        vc = a - b
        d += 0.5 * c * (vc[-1] ** 2 - vc[0] ** 2)
    names = list(inds)
    cur = {}
    for n in names:
        i = _get(w, "i(%s)" % n)
        if i is None:
            out["notes"].append("电感 %s 电流缺失" % n); return out
        cur[n] = i
    Lm = {n: parse_value(inds[n].value) for n in names}
    if any(not v for v in Lm.values()):
        out["notes"].append("有电感值无法解析"); return out
    M = {}
    for e in ks:
        tok = (e.value or "").split()
        if len(tok) >= 3:
            a, b, k = tok[-3].lower(), tok[-2].lower(), parse_value(tok[-1])
            if a in Lm and b in Lm and k:
                M[(a, b)] = M[(b, a)] = k * np.sqrt(Lm[a] * Lm[b])
    def mag(idx):
        s = sum(0.5 * Lm[n] * cur[n][idx] ** 2 for n in names)
        for (a, b), m in M.items():
            if a < b:
                s += m * cur[a][idx] * cur[b][idx]
        return s
    d += mag(-1) - mag(0)
    out["d_stored"] = d
    resid = out["e_src"] - out["e_R"] - d
    out["residual"] = resid
    scale = max(abs(out["e_src"]), abs(out["e_R"]), abs(d), 1e-30)
    out["rel"] = resid / scale
    if not out["others"]:
        out["status"] = "通过" if abs(out["rel"]) <= tol else "黄牌"
        out["notes"].append("线性网络：残差应≈0，实际 %.3g（相对 %.2f%%）" % (resid, 100 * out["rel"]))
    else:
        out["status"] = "通过" if out["rel"] >= -tol else "黄牌"
        out["notes"].append("含器件 %s：残差=这些器件总耗散，应≥0，实际 %.3g（相对 %.2f%%）" % (out["others"], resid, 100 * out["rel"]))
    return out
