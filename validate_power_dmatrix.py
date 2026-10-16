# -*- coding: utf-8 -*-
"""D 矩阵抽样：相移全桥 + 串联谐振槽 ngspice 时域 vs 平均模型。单个工作点两种 m 语义。"""
import math, subprocess, sys
NG = "/home/sandbox/ngbuild/root/usr/bin/ngspice"

def pwl_source(name, node, Vbus, f, delay, ncyc=18):
    """50% 方波腿，延迟 delay 秒起始，显式 PWL 边沿列表。"""
    T = 1.0 / f; tr = 100e-9; pts = []; t = 0.0
    pts.append((0.0, 0.0))
    t = delay
    if delay > 0: pts.append((delay, 0.0))
    for c in range(ncyc):
        t0 = delay + c * T
        pts.append((t0 + tr, Vbus)); pts.append((t0 + T / 2, Vbus)); pts.append((t0 + T / 2 + tr, 0.0)); pts.append((t0 + (c + 1) * T - delay * 0 + (c * 0), 0.0))
        pts[-1] = (t0 + T, 0.0)
    body = " ".join("%.9g %.4f" % p for p in pts)
    return "%s %s 0 PWL( %s )" % (name, node, body)

def run(tag, Vbus, f, L, C, R, m, mode):
    T = 1.0 / f
    if mode == "phase":   # m = 相移角/π
        delay = m * T / 2.0
    else:                  # m = 幅值等效：sin(φ/2)=m → 移相角 φ=2asin(m)
        delay = math.asin(min(m, 1.0)) * T / math.pi
    net = "\n".join([
        "* D-matrix %s %s" % (tag, mode),
        pwl_source("VA", "a", Vbus, f, 0.0),
        pwl_source("VB", "b", Vbus, f, delay),
        "La a x %.6e" % L,
        "Cc x y %.6e" % C,
        "Rt y b %.6e" % R,
        "Bcap vc 0 v = v(x)-v(y)",
        "Bab vab 0 v = v(a)-v(b)",
        ".control",
        "tran 50n 500u",
        "meas tran IRMS RMS i(VA) from=350u to=500u",
        "meas tran VCRMS RMS v(vc) from=350u to=500u",
        "meas tran VABRMS RMS v(vab) from=350u to=500u",
        ".endc",
        ".end", ""])
    fn = "/tmp/dm_%s_%s.cir" % (tag, mode)
    open(fn, "w").write(net)
    out = subprocess.run([NG, "-b", fn], capture_output=True, text=True, timeout=300).stdout
    r = {}
    for line in out.splitlines():
        for k in ("irms", "vcrms", "vabrms"):
            if line.strip().lower().startswith(k) and "=" in line:
                r[k] = float(line.split("=")[1].split()[0])
    I, Vc = r.get("irms", float("nan")), r.get("vcrms", float("nan"))
    P = I * I * R
    print("  [%s] 移相 %.4f° | I1 %.1f Arms | Uc %.0f Vrms | P %.1f kW | Vab基波占比 %.3f" % (
        mode, delay / T * 360.0, I, Vc, P / 1e3, r.get("vabrms", 0) / (Vbus / math.sqrt(2))))
    return I, Vc, P

if __name__ == "__main__":
    Vbus, f, L, C, R, m = (float(x) for x in sys.argv[1:7])
    tag = sys.argv[7]
    print("工作点 %s：Vbus=%.1f f=%.1fkHz L=%.2fuH C=%.3fuF R=%.3fΩ m=%.6f" % (tag, Vbus, f / 1e3, L * 1e6, C * 1e6, R, m))
    for mode in ("phase", "ampsin"):
        run(tag, Vbus, f, L, C, R, m, mode)

# D 矩阵 v1 结论（2026-10-16，ngspice 36 本地）：
# 528Vac 锚点（Vbus=702.1V, f=30kHz, 热态槽路 L=45.41uH/C=0.667uF/R=2.891Ω, m=0.949173）：
#   [ampsin] φ=2·asin(m)：I1 203.2A(-0.5%) / Uc 1616V(-2.2%) / P 119.4kW(-0.9%) —— 与 B 轨锚点一致，
#     证实 B 轨"固定相移 m"语义=幅值等效（V1_rms=0.9003·Vbus·m），v0.5 平均模型线性 m 口径正确。
#   [phase] φ=m·π（几何移相角）：I1 213.5A(+4.6%) / P 131.8kW(+9.4%) —— 若固件把 m 当几何移相比例实现，
#     满载点将超限 27.5A。控制板固件必须用幅值等效映射（或 φ=2·asin(m)）。已列入设计注意事项。
# 口径：ngspice 36（本机）vs B 轨 ngspice 42；测量为真 RMS（含少量谐波），梯形边沿 100ns。
