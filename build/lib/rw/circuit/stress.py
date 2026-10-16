"""v0.12 器件应力校核（MIT，公开）：每个器件的峰值电压/电流 vs 额定值，>80 % 黄、>100 % 红。

    rows = stress.check(circuit, waveforms)          # 导入的电路：按 circuit.ratings 和节点电压自动算
    rows = stress.table([("V8", "反压", 617, 600, "C3D06060"), ...])   # 已知峰值时直接组表
每行 dict：ref, qty, peak, rating, pct, level('ok'|'warn'|'over'), color, note
阈值可改：WARN=0.8, OVER=1.0（常见降额设计取 80 %；这里不替代具体行业降额规范）。
"""
import numpy as np

WARN, OVER = 0.80, 1.00
COLORS = {"ok": "#2e9e4f", "warn": "#e0b000", "over": "#d62828"}
LABEL = {"ok": "正常", "warn": ">80 %", "over": "超额定"}


def level(pct):
    return "over" if pct > OVER * 100 else ("warn" if pct > WARN * 100 else "ok")


def row(ref, qty, peak, rating, note=""):
    pct = 100.0 * abs(peak) / rating if rating else float("nan")
    lv = level(pct) if rating else "ok"
    return dict(ref=ref, qty=qty, peak=float(peak), rating=float(rating) if rating else None, pct=pct, level=lv,
                color=COLORS[lv], note=note)


def table(items):
    """items: [(ref, 量, 峰值, 额定, 备注), ...]"""
    return [row(*it) for it in items]


def _wv(w, n):
    if n == "0":
        return 0.0
    try:
        return np.asarray(w[n])
    except KeyError:
        return None


def check(c, w, t_from=None):
    """导入电路 c（kicad.load 得到，含 c.ratings {ref: {'V':..,'A':..}}、元件节点）+ 仿真波形 w。
    电压 = 两端节点电压差的峰值绝对值；电阻电流 = ΔV/R。t_from 之后才统计（跳过上电过程）。"""
    rat = getattr(c, "ratings", {}) or {}
    els = {e.name.upper(): e for e in c.elements}
    t = np.asarray(w.time) if hasattr(w, "time") else None
    sel = slice(None) if (t is None or t_from is None) else (t >= t_from)
    out = []
    for ref, r in sorted(rat.items()):
        e = None
        for pre in ("", "R", "C", "L", "D", "Q", "M", "X"):
            e = els.get((pre + ref).upper())
            if e is not None:
                break
        sn = (getattr(c, "stress_nodes", {}) or {}).get(ref)
        nodes = list(sn) if sn else (e.nodes if e is not None and e.nodes else None)
        if nodes is None and e is not None:
            nodes = e.line().split()[1:3]
        if not nodes or len(nodes) < 2:
            continue
        a, b = _wv(w, nodes[0]), _wv(w, nodes[1])
        if a is None or b is None:
            continue
        dv = np.asarray(a - b)[sel] if np.ndim(a - b) else np.array([a - b])
        if "V" in r:
            out.append(row(ref, "电压", float(np.max(np.abs(dv))), r["V"], r.get("src", "")))
        if "A" in r and e is not None and e.kind == "R":
            try:
                R = float(e.value)
                out.append(row(ref, "电流", float(np.max(np.abs(dv))) / R, r["A"], "ΔV/R"))
            except ValueError:
                pass
    return out


def summary(rows):
    n = {k: sum(1 for r in rows if r["level"] == k) for k in COLORS}
    worst = max(rows, key=lambda r: r["pct"] if r["pct"] == r["pct"] else -1) if rows else None
    return n, worst


def text(rows):
    lines = ["%-6s %-14s %9s %9s %7s  %s" % ("位号", "量", "峰值", "额定", "%", "判定")]
    for r in rows:
        lines.append("%-6s %-14s %9.4g %9.4g %6.0f%%  %s %s" % (r["ref"], r["qty"], r["peak"], r["rating"] or 0, r["pct"], LABEL[r["level"]], r["note"]))
    return "\n".join(lines)
