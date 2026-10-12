"""电路板案例批处理 + 器件热模型（通用代码；具体板子的案例数据放在 models/private_boards/<板>/，不公开）。

流程：ngspice -b 跑网表（替换 {RS}/{TAG} 等模板参数）→ 解析 .meas 与 wrdata 波形 → P(t)=V·I →
两级 Foster 热网络 → Tj(t)、稳态 Tj → 判定（正常/警告/超 Tj max 烧毁/峰值电流 > IFSM 击穿）。

热模型假设（如实）：总 Rth_ja 默认 100 K/W（数据手册/板面积假设）；拆成 结→壳 15 K/W（τ=2 ms）+ 壳→板→空气 85 K/W
（τ=20 s）。两个时间常数是保守的“假设值”，没有实测；稳态 Tj=Ta+P·Rth 与时间常数无关。
"""
import os, re, shutil, subprocess, tempfile
import numpy as np
from . import spice as _sp


def run_cir(path, workdir=None, params=None, timeout=200):
    """跑一个 .cir；params 替换文件里的 {NAME}。返回 (meas dict, 波形 dict: 名→(t,v))。"""
    exe = _sp.find_ngspice()
    if not exe:
        raise RuntimeError("找不到 ngspice")
    txt = open(path, encoding="utf-8", errors="ignore").read()
    for k, v in (params or {}).items():
        txt = txt.replace("{%s}" % k, str(v))
    wd = workdir or tempfile.mkdtemp(prefix="rwboard_")
    os.makedirs(wd, exist_ok=True)
    cir = os.path.join(wd, "case.cir")
    open(cir, "w").write(txt)
    r = subprocess.run([exe, "-b", "case.cir"], cwd=wd, capture_output=True, text=True, timeout=timeout, env=_sp._env_for(exe))
    meas = {}
    for m in re.finditer(r"^([a-z0-9_]+)\s+=\s+(-?[0-9.]+(?:e[+-]?\d+)?)", r.stdout, re.M | re.I):
        meas[m.group(1).lower()] = float(m.group(2))
    waves = {}
    mw = re.search(r"^\s*wrdata\s+(\S+)\s+(.+)$", txt, re.M | re.I)
    if mw:
        fn = os.path.join(wd, mw.group(1))
        names = mw.group(2).split()
        if os.path.exists(fn):
            a = np.loadtxt(fn)
            for i, n in enumerate(names):
                waves[n.lower()] = (a[:, 2 * i], a[:, 2 * i + 1])
    if not waves and not meas:
        raise RuntimeError("ngspice 无输出：" + r.stdout[-300:] + r.stderr[-300:])
    return meas, waves


class Thermal:
    """两级 Foster 热网络：Tj = Ta + P1·R1 状态 + P2·R2 状态。"""
    def __init__(self, rjc=15.0, rca=85.0, tau1=2e-3, tau2=20.0, ta=40.0, tjmax=150.0, ifsm=10.0):
        self.r1, self.r2, self.t1, self.t2, self.ta, self.tjmax, self.ifsm = rjc, rca, tau1, tau2, ta, tjmax, ifsm

    @property
    def rth(self):
        return self.r1 + self.r2

    def tj(self, t, p):
        """t 递增，p(t) 分段线性；返回 Tj(t)。每段 P 取均值，指数精确更新。"""
        t = np.asarray(t, float); p = np.asarray(p, float)
        x1 = x2 = 0.0
        out = np.empty_like(t); out[0] = self.ta
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            pm = 0.5 * (p[i] + p[i - 1])
            a1, a2 = np.exp(-dt / self.t1), np.exp(-dt / self.t2)
            x1 = x1 * a1 + pm * self.r1 * (1 - a1)
            x2 = x2 * a2 + pm * self.r2 * (1 - a2)
            out[i] = self.ta + x1 + x2
        return out


def downsample(t, y, n=4000):
    """按时间等间隔取每格最大绝对值之外的均值（功率用均值保能量）。"""
    if len(t) <= n:
        return np.asarray(t), np.asarray(y)
    edges = np.linspace(t[0], t[-1], n + 1)
    idx = np.searchsorted(t, edges)
    ts, ys = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        if b > a:
            ts.append(0.5 * (t[a] + t[min(b, len(t)) - 1])); ys.append(np.mean(y[a:b]))
    return np.array(ts), np.array(ys)


def thermal_response(t, p, th, hold_s=300.0, hold_frac=0.2, pend=None):
    """仿真窗口内用 P(t)，之后保持窗口末段均值 P 至 hold_s（持续故障看稳态）。返回 (t, Tj, Pend, 稳态Tj)。"""
    t, p = downsample(t, p)
    tail = p[t >= t[-1] - hold_frac * (t[-1] - t[0])]
    pend = float(np.mean(tail)) if pend is None else float(pend)
    tt = np.concatenate([t, np.geomspace(max(t[-1] * 1.5, 1e-3), hold_s, 150)])
    pp = np.concatenate([p, np.full(150, pend)])
    tj = th.tj(tt, pp)
    return tt, tj, pend, th.ta + pend * th.rth


def judge(th, tj_peak_burst, tj_ss, i_peak, ifsm=None, persistent=True):
    ifsm = th.ifsm if ifsm is None else ifsm
    if i_peak is not None and i_peak > ifsm:
        return "击穿", "峰值电流 %.1f A > IFSM %.0f A" % (i_peak, ifsm), "red"
    tj = max(tj_ss if persistent else 0, tj_peak_burst)
    if tj > th.tjmax:
        return "烧毁", "Tj %.0f °C > Tj max %.0f °C" % (tj, th.tjmax), "red"
    if tj > th.tjmax - 50 or (i_peak is not None and i_peak > 0.7 * ifsm):
        return "警告", "Tj %.0f °C，离 Tj max 不足 50 K（或电流 >70%% IFSM）" % tj, "orange"
    return "正常", "Tj %.0f °C" % tj, "green"
