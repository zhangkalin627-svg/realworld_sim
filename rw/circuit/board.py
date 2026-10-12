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
    _sp.check_exit(r, os.path.basename(path))   # 被杀/崩溃 → 明确报错，不解析残缺输出
    meas = {}
    for m in re.finditer(r"^([a-z0-9_]+)\s+=\s+(-?[0-9.]+(?:e[+-]?\d+)?)", r.stdout, re.M | re.I):
        meas[m.group(1).lower()] = float(m.group(2))
    waves = {}
    mw = re.search(r"^\s*wrdata\s+(\S+)\s+(.+)$", txt, re.M | re.I)
    if mw:
        fn = os.path.join(wd, mw.group(1))
        names = mw.group(2).split()
        if os.path.exists(fn):
            try:
                a = np.loadtxt(fn)
            except ValueError as e:
                raise RuntimeError("ngspice 的 wrdata 文件残缺（%s）：%s" % (os.path.basename(path), e))
            for i, n in enumerate(names):
                waves[n.lower()] = (a[:, 2 * i], a[:, 2 * i + 1])
    if not waves and not meas:
        raise RuntimeError("ngspice 无输出：" + r.stdout[-300:] + r.stderr[-300:])
    return meas, waves


class Thermal:
    """两级 Foster 热网络：Tj = Ta + P1·R1 状态 + P2·R2 状态。

    v0.11.1 烧毁截止：Tj 超过 burn_c（默认 Tj max + 100 °C，假设值：焊料/封装/硅片在这个温度以上会永久损坏）后，
    记为“烧毁”（self.burned=True, self.t_burn），之后器件按 fail_mode 假设处理并明确标注：
      "open"（默认）开路 → 器件不再耗功率（P=0），Tj 按热网络冷却；
      "short" 短路 → 器件只剩导通电阻的小功耗，P = short_frac·P（默认 0.05，假设值）。
    这样不再出现 654 °C 这类远超熔化的温度。"""
    def __init__(self, rjc=15.0, rca=85.0, tau1=2e-3, tau2=20.0, ta=40.0, tjmax=150.0, ifsm=10.0,
                 burn_c=None, fail_mode="open", short_frac=0.05):
        self.r1, self.r2, self.t1, self.t2, self.ta, self.tjmax, self.ifsm = rjc, rca, tau1, tau2, ta, tjmax, ifsm
        self.burn_c = tjmax + 100.0 if burn_c is None else burn_c
        self.fail_mode, self.short_frac = fail_mode, short_frac
        self.burned, self.t_burn, self.tj_raw_peak = False, None, None

    @property
    def rth(self):
        return self.r1 + self.r2

    @property
    def fail_note(self):
        if not self.burned:
            return ""
        m = "开路（P=0）" if self.fail_mode == "open" else "短路（P×%.2g）" % self.short_frac
        return "t=%.3g ms Tj 超过烧毁温度 %.0f °C → 记为烧毁，之后按%s假设计算" % (self.t_burn * 1e3, self.burn_c, m)

    def tj(self, t, p, burn=True):
        """t 递增，p(t) 是每个采样点所在小段的平均功率；第 i 段 (t[i-1], t[i]] 用 0.5·(p[i-1]+p[i])，指数精确更新。
        burn=True 时启用烧毁截止（见类说明）。"""
        t = np.asarray(t, float); p = np.asarray(p, float)
        x1 = x2 = 0.0
        out = np.empty_like(t); out[0] = self.ta
        self.burned, self.t_burn, peak = False, None, self.ta
        k = 1.0
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            pm = 0.5 * (p[i] + p[i - 1]) * k
            a1, a2 = np.exp(-dt / self.t1), np.exp(-dt / self.t2)
            x1 = x1 * a1 + pm * self.r1 * (1 - a1)
            x2 = x2 * a2 + pm * self.r2 * (1 - a2)
            out[i] = self.ta + x1 + x2
            peak = max(peak, out[i])
            if burn and not self.burned and out[i] > self.burn_c:
                self.burned, self.t_burn = True, float(t[i])
                f = (self.burn_c - self.ta) / (out[i] - self.ta)  # 截到烧毁温度（该步内到达）
                x1, x2 = x1 * f, x2 * f; out[i] = self.burn_c
                k = 0.0 if self.fail_mode == "open" else self.short_frac
        self.tj_raw_peak = peak
        return out


def downsample(t, y, n=4000):
    """按时间等间隔分 n 格，每格取均值（功率用均值保能量）。"""
    if len(t) <= n:
        return np.asarray(t), np.asarray(y)
    edges = np.linspace(t[0], t[-1], n + 1)
    idx = np.searchsorted(t, edges)
    ts, ys = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        if b > a:
            ts.append(0.5 * (t[a] + t[min(b, len(t)) - 1])); ys.append(np.mean(y[a:b]))
    return np.array(ts), np.array(ys)


def thermal_response(t, p, th, hold_s=300.0, hold_frac=0.2, pend=None, burn=True):
    """仿真窗口内用 P(t)，之后保持窗口末段均值 P 至 hold_s（持续故障看稳态）。返回 (t, Tj, Pend, 稳态Tj)。

    v0.11.1 拼接修复：以前窗口末点 t_end 直接连到 1.5·t_end，那一大段用“末点瞬时 P 与保持 P 的平均”，
    开关纹波下末点 P 可能≈0 → Tj 曲线出现台阶/凹口。现在：窗口末点之后立即（同一时刻）切到保持功率，
    保持段从 t_end 开始按几何间隔加密（第一步 = 窗口内的平均步长），Tj 在拼接处连续。
    稳态 Tj 若器件已烧毁（开路），稳态 = Ta。"""
    t, p = downsample(np.asarray(t, float), np.asarray(p, float))
    p = np.maximum(p, 0.0)  # 器件耗散功率非负；负值（数值噪声/输入错）钳 0，避免 Tj<绝对零度
    tail = p[t >= t[-1] - hold_frac * (t[-1] - t[0])]
    pend = float(np.mean(tail)) if pend is None else max(float(pend), 0.0)
    dt0 = max((t[-1] - t[0]) / max(len(t) - 1, 1), 1e-9)
    th_ = np.geomspace(dt0, max(hold_s - t[-1], 2 * dt0), 400)
    # 在 t_end 处放一个与末点同时刻的“切换点”（p 从窗口值切到保持值），之后都是 pend
    tt = np.concatenate([t, [t[-1]], t[-1] + th_])
    pp = np.concatenate([p, [pend], np.full(len(th_), pend)])
    tj = th.tj(tt, pp, burn=burn)
    ss = th.ta + pend * th.rth
    if th.burned:
        ss = th.ta + (0.0 if th.fail_mode == "open" else pend * th.short_frac * th.rth)
    return tt, tj, pend, ss


def judge(th, tj_peak_burst, tj_ss, i_peak, ifsm=None, persistent=True):
    ifsm = th.ifsm if ifsm is None else ifsm
    if i_peak is not None and i_peak > ifsm:
        return "击穿", "峰值电流 %.1f A > IFSM %.0f A" % (i_peak, ifsm) + ("；" + th.fail_note if getattr(th, "burned", False) else ""), "red"
    if getattr(th, "burned", False):
        return "烧毁", th.fail_note, "red"
    tj = max(tj_ss if persistent else 0, tj_peak_burst)
    if tj > th.tjmax:
        return "烧毁", "Tj %.0f °C > Tj max %.0f °C" % (tj, th.tjmax), "red"
    if tj > th.tjmax - 50 or (i_peak is not None and i_peak > 0.7 * ifsm):
        return "警告", "Tj %.0f °C，离 Tj max 不足 50 K（或电流 >70%% IFSM）" % tj, "orange"
    return "正常", "Tj %.0f °C" % tj, "green"
