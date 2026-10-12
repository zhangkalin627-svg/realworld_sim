"""虚拟示波器引擎（不含界面）：按真实示波器的操作逻辑处理一段仿真波形记录。

- 屏幕：横 10 格 × 纵 8 格；
- 通道（默认 4 个）：信号源可选任意网络、两网络差分 'a-b'、电流 'i(V1)'；伏/格、偏移（V）、DC/AC 耦合、开关；
- 时基：s/格、水平位置（触发点相对屏幕中心的时间偏移）；
- 边沿触发：源通道、上升/下降、电平、Auto / Normal / Single；
- 光标：两条时间光标（Δt、1/Δt）+ 两条电压光标（ΔV）；
- 自动测量：Vpp Vmax Vmin Vavg Vrms 频率 周期 占空比 上升/下降时间（10–90%）；
- FFT：加窗（矩形/汉宁/汉明/布莱克曼/平顶），dBV（相对 1 Vrms）。

说明（如实）：
- 仿真结果是一次性的“记录”，Auto/Normal 下每次 acquire() 取记录里第 n 个触发点来显示，相当于示波器连续刷新；
  Single 只取第一个触发点，之后锁屏直到 arm()。
- AC 耦合用“减去整段记录平均值”近似，不是真实示波器的 ~10 Hz 高通（低频信号或刚上电的瞬态会和真机不同）。
- 测量在“屏幕显示的那一段”上做（和多数台式示波器默认一致）；上升/下降时间的 0%/100% 电平取该段的 Vmin/Vmax
  （真机常用直方图求顶/底，波形有过冲时两者会有差别）。
"""
import numpy as np

COLORS = ["#f5d000", "#00d4ff", "#ff4fd8", "#4cff6a"]  # 常见示波器 CH1 黄 CH2 青 CH3 品红 CH4 绿
HDIV, VDIV = 10, 8
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy<2 兼容
WINDOWS = {"矩形": "rect", "汉宁": "hann", "汉明": "hamming", "布莱克曼": "blackman", "平顶": "flattop"}


def nice_steps(lo=1e-9, hi=100):
    """1-2-5 序列（伏/格、秒/格的档位）。"""
    out, d = [], 10 ** np.floor(np.log10(lo))
    while d <= hi * 1.0001:
        for m in (1, 2, 5):
            v = m * d
            if lo * 0.9999 <= v <= hi * 1.0001:
                out.append(float("%.3g" % v))
        d *= 10
    return out


def eng(v, unit="", digits=4):
    """工程计数法：0.00123 → '1.23 m'。"""
    if v is None or not np.isfinite(v):
        return "—"
    if v == 0:
        return f"0 {unit}".strip()
    p = int(np.floor(np.log10(abs(v)) / 3))
    p = max(-5, min(4, p))
    pre = {-5: "f", -4: "p", -3: "n", -2: "µ", -1: "m", 0: "", 1: "k", 2: "M", 3: "G", 4: "T"}[p]
    return f"{v / 10 ** (3 * p):.{digits}g} {pre}{unit}".strip()


class Channel:
    def __init__(self, n, source="", vdiv=1.0, offset=0.0, coupling="DC", on=True):
        self.n, self.source, self.vdiv, self.offset, self.coupling, self.on = n, source, vdiv, offset, coupling, on
        self.color = COLORS[(n - 1) % len(COLORS)]


class Trigger:
    def __init__(self):
        self.source = 1        # 通道号
        self.slope = "rise"    # rise / fall
        self.level = 0.5       # V（按通道耦合后的电压）
        self.mode = "Auto"     # Auto / Normal / Single
        self.holdoff = 0.0


class Scope:
    def __init__(self, n_channels=4):
        self.ch = [Channel(i + 1, on=(i == 0)) for i in range(n_channels)]
        self.tdiv = 1e-3
        self.hpos = 0.0        # 触发点相对屏幕中心的时间偏移（正值 = 屏幕往后看）
        self.trig = Trigger()
        self.t = None
        self.wave = None
        self.frame_index = None  # None = 用记录里最后一个合适的触发点（最接近稳态）
        self.single_done = False
        self.last = None       # 上一帧（Normal 无触发 / Single 锁屏时保持显示）
        self.cursors = dict(t1=None, t2=None, v1=None, v2=None, vch=1)

    # ------------------------------------------------ 数据
    def load(self, waveforms):
        """waveforms：spice.Waveforms（瞬态结果），或 dict{'time':..., 名字: 数组}。"""
        self.wave = waveforms
        self.t = np.asarray(waveforms.x if hasattr(waveforms, "x") else waveforms["time"], float)
        self.frame_index, self.single_done, self.last = None, False, None

    def raw(self, ch):
        c = self.ch[ch - 1] if isinstance(ch, int) else ch
        if not c.source:
            return None
        try:
            y = np.real(np.asarray(self.wave[c.source], float))
        except KeyError:
            self.warn = "没有网络 %s，该通道显示为空" % c.source
            return None
        if c.coupling == "AC":
            y = y - _tavg(self.t, y)
        return y

    # ------------------------------------------------ 触发
    def trigger_times(self):
        y = self.raw(self.trig.source)
        if y is None:
            return np.array([])
        lv = self.trig.level
        s = np.sign(y - lv)
        s[s == 0] = -1 if self.trig.slope == "rise" else 1
        if self.trig.slope == "rise":
            idx = np.nonzero((s[:-1] < 0) & (s[1:] > 0))[0]
        else:
            idx = np.nonzero((s[:-1] > 0) & (s[1:] < 0))[0]
        t0, t1, y0, y1 = self.t[idx], self.t[idx + 1], y[idx], y[idx + 1]
        tt = t0 + (lv - y0) * (t1 - t0) / np.where(y1 != y0, y1 - y0, 1)
        if self.trig.holdoff > 0 and len(tt):
            keep, last = [], -np.inf
            for x in tt:
                if x - last >= self.trig.holdoff:
                    keep.append(x); last = x
            tt = np.array(keep)
        # 只要能在记录里放下整屏（或至少触发点在屏内）的触发点
        lo = self.t[0] + 5 * self.tdiv - self.hpos
        hi = self.t[-1] - 5 * self.tdiv - self.hpos
        ok = tt[(tt >= lo) & (tt <= hi)]
        return ok if len(ok) else tt

    def arm(self):
        self.single_done = False
        self.frame_index = None

    def autoset(self, chans=None):
        """像示波器的 Auto Set：按记录后半段（接近稳态）定伏/格、偏移、时基、触发电平。"""
        if self.t is None:
            return
        m = self.t >= self.t[0] + 0.5 * (self.t[-1] - self.t[0])
        steps = nice_steps(1e-6, 1000)
        on = [c for c in self.ch if c.on and c.source]
        n = len(on)
        band = VDIV / max(n, 1)  # 多通道时上下分区摆放，互不重叠（真机 Auto Set 会叠在一起，这里为了好看）
        for i, c in enumerate(on):
            y = self.raw(c.n)[m]
            lo, hi = float(np.min(y)), float(np.max(y))
            big = max(abs(hi), abs(lo))
            center_div = VDIV / 2 - band * (i + 0.5)
            if hi - lo < 0.05 * big:  # 基本是直流：让 0 V 和波形都在本区里
                c.vdiv = next((s for s in steps if s * band * 0.9 >= big), steps[-1])
                ground_div = center_div - np.sign(hi + lo) * band * 0.4  # 0 V 放在本区的一端，波形在另一端
                c.offset = ground_div * c.vdiv
                continue
            c.vdiv = next((s for s in steps if s * band * 0.85 >= hi - lo), steps[-1])
            c.offset = -(hi + lo) / 2 + center_div * c.vdiv
            c.offset = round(c.offset / (c.vdiv / 10)) * (c.vdiv / 10)
        src = next((c for c in self.ch if c.on and c.source), None)
        if src is None:
            return
        y = self.raw(src.n)[m]
        r = measure(self.t[m], y)
        tsteps = nice_steps(1e-9, 100)
        dur = self.t[-1] - self.t[0]
        want = 2.5 / r["频率"] / HDIV if r.get("频率") else dur / HDIV
        self.tdiv = next((s for s in tsteps if s >= want), tsteps[-1])
        if self.tdiv * HDIV > dur:
            self.tdiv = max([s for s in tsteps if s * HDIV <= dur] or [tsteps[0]])
        self.hpos = 0.0
        self.trig.source, self.trig.slope = src.n, "rise"
        self.trig.level = 0.5 * (float(np.max(y)) + float(np.min(y)))
        self.frame_index = None

    def acquire(self, advance=False):
        """取一帧。返回 dict：t0（屏幕左边对应的绝对时间）、tt（触发时刻或 None）、traces、状态文字。"""
        if self.t is None:
            return None
        if not (self.tdiv > 0):  # tdiv≤0 无意义，钳到最小 1 ns
            self.tdiv = 1e-9
        if self.trig.mode == "Single" and self.single_done and self.last is not None:
            return dict(self.last, status="Stop（Single 已触发，按“单次”重新等待）")
        tt = self.trigger_times()
        if advance and len(tt):
            self.frame_index = 0 if self.frame_index is None else (self.frame_index + 1) % len(tt)
        if len(tt):
            if self.frame_index is None:
                k = 0 if self.trig.mode == "Single" else len(tt) - 1  # 单次取第一个触发点，其余取最后一个（接近稳态）
            else:
                k = min(self.frame_index, len(tt) - 1)
            ttrig = tt[k]
            fr_k = (k, len(tt))
            status = "Trig'd"
        elif self.trig.mode == "Auto":
            ttrig = None
            status = "Auto（未找到触发，自由显示）"
        else:
            if self.last is not None:
                return dict(self.last, status="Wait（等待触发）")
            return dict(t0=None, tt=None, traces={}, status="Wait（等待触发）")
        if ttrig is None:
            center = self.t[0] + 5 * self.tdiv + self.hpos
        else:
            center = ttrig + self.hpos
        t0 = center - 5 * self.tdiv
        fr = dict(t0=t0, tt=ttrig, traces={}, status=status, center=center,
                  trig_no=fr_k if ttrig is not None else None)
        for c in self.ch:
            if c.on and c.source and self.raw(c.n) is not None:
                x, y = self.window(c.n, t0, t0 + HDIV * self.tdiv)
                fr["traces"][c.n] = (x, y)
        self.last = fr
        if self.trig.mode == "Single" and ttrig is not None:
            self.single_done = True
        return fr

    def window(self, ch, ta, tb, n_min=2000):
        """取 [ta, tb] 段波形（带两端插值点），x 为相对 ta 的时间。"""
        y = self.raw(ch)
        t = self.t
        m = (t > ta) & (t < tb)
        tw = np.concatenate([[ta], t[m], [tb]])
        yw = np.interp(tw, t, y, left=np.nan, right=np.nan)
        if m.sum() < n_min:  # 点太少时加密（线性插值），让屏幕上看着平滑；测量也基于此
            tg = np.linspace(ta, tb, n_min)
            tw = np.union1d(tw, tg)
            yw = np.interp(tw, t, y, left=np.nan, right=np.nan)
        return tw - ta, yw

    # ------------------------------------------------ 屏幕坐标
    def to_div(self, ch, y):
        c = self.ch[ch - 1]
        return (y + c.offset) / c.vdiv

    # ------------------------------------------------ 测量
    def measure(self, ch, frame=None):
        fr = frame or self.last or self.acquire()
        if not fr or ch not in fr["traces"]:
            return {}
        x, y = fr["traces"][ch]
        ok = np.isfinite(y)
        return measure(x[ok], y[ok])

    def cursor_readout(self, frame=None):
        fr = frame or self.last
        c = self.cursors
        out = {}
        if c["t1"] is not None and c["t2"] is not None:
            dt = c["t2"] - c["t1"]
            out["Δt"] = dt; out["1/Δt"] = 1 / dt if dt else np.inf
        if c["v1"] is not None and c["v2"] is not None:
            out["ΔV"] = c["v2"] - c["v1"]
        if fr and fr.get("t0") is not None:
            for k in ("t1", "t2"):
                if c[k] is not None:
                    for n, (x, y) in fr["traces"].items():
                        out[f"CH{n}@{k}"] = float(np.interp(c[k], x, y))
        return out

    # ------------------------------------------------ FFT
    def fft(self, ch, window="hann", span="screen", frame=None):
        if span == "screen":
            fr = frame or self.last or self.acquire()
            x, y = fr["traces"][ch]
            ok = np.isfinite(y); x, y = x[ok], y[ok]
        else:
            x, y = self.t - self.t[0], self.raw(ch)
        return fft_dbv(x, y, window)


# ---------------------------------------------------------------- 纯函数（validate.py 也直接用）
def _tavg(t, y):
    T = t[-1] - t[0]
    return float(_trapz(y, t) / T) if T > 0 else float(np.mean(y))


def crossings(t, y, level, slope):
    s = y > level
    if slope == "rise":
        idx = np.nonzero(~s[:-1] & s[1:])[0]
    else:
        idx = np.nonzero(s[:-1] & ~s[1:])[0]
    y0, y1 = y[idx], y[idx + 1]
    return t[idx] + (level - y0) * (t[idx + 1] - t[idx]) / np.where(y1 != y0, y1 - y0, 1)


def measure(t, y):
    """对一段（可能不等间隔的）波形做自动测量。量不出来的项为 None。"""
    if len(t) < 3:
        return {}
    vmax, vmin = float(np.max(y)), float(np.min(y))
    T = t[-1] - t[0]
    vavg = _tavg(t, y)
    vrms = float(np.sqrt(_trapz(y * y, t) / T)) if T > 0 else float(np.sqrt(np.mean(y * y)))
    r = dict(Vpp=vmax - vmin, Vmax=vmax, Vmin=vmin, Vavg=vavg, Vrms=vrms,
             频率=None, 周期=None, 占空比=None, 上升时间=None, 下降时间=None)
    amp = vmax - vmin
    if amp <= 1e-12 * max(1, abs(vmax)):
        return r
    mid = vmin + 0.5 * amp
    # 带回差的 50% 电平穿越，避免噪声/振铃产生假穿越
    hy = 0.05 * amp
    state, ups, dns = None, [], []  # 先确认的事件
    i_up = crossings(t, y, mid + hy, "rise"); i_dn = crossings(t, y, mid - hy, "fall")
    ev = sorted([(x, 1) for x in i_up] + [(x, -1) for x in i_dn])
    for x, k in ev:
        if k != state:
            state = k
            (ups if k == 1 else dns).append(x)
    up50 = crossings(t, y, mid, "rise"); dn50 = crossings(t, y, mid, "fall")
    # 用回差确认过的事件附近的 50% 穿越作为精确时刻
    def snap(ev_list, exact):
        out = []
        for x in ev_list:
            c = exact[exact <= x]
            if len(c):
                out.append(c[-1])
        return np.unique(np.array(out))
    ups, dns = snap(ups, up50), snap(dns, dn50)
    if len(ups) >= 2:
        per = float(np.mean(np.diff(ups)))
        r["周期"], r["频率"] = per, 1 / per
        hs = []
        for a, b in zip(ups[:-1], ups[1:]):
            d = dns[(dns > a) & (dns < b)]
            if len(d):
                hs.append(d[0] - a)
        if hs:
            r["占空比"] = float(np.mean(hs) / per)
    elif len(dns) >= 2:
        per = float(np.mean(np.diff(dns)))
        r["周期"], r["频率"] = per, 1 / per
    l10, l90 = vmin + 0.1 * amp, vmin + 0.9 * amp
    a, b = crossings(t, y, l10, "rise"), crossings(t, y, l90, "rise")
    rt = [b[b > x][0] - x for x in a if np.any(b > x)]
    if rt:
        r["上升时间"] = float(rt[0])
    a, b = crossings(t, y, l90, "fall"), crossings(t, y, l10, "fall")
    ft = [b[b > x][0] - x for x in a if np.any(b > x)]
    if ft:
        r["下降时间"] = float(ft[0])
    return r


def _window(name, n):
    k = np.arange(n)
    if name == "rect": return np.ones(n)
    if name == "hann": return 0.5 - 0.5 * np.cos(2 * np.pi * k / n)
    if name == "hamming": return 0.54 - 0.46 * np.cos(2 * np.pi * k / n)
    if name == "blackman": return 0.42 - 0.5 * np.cos(2 * np.pi * k / n) + 0.08 * np.cos(4 * np.pi * k / n)
    if name == "flattop":
        a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
        return sum((-1) ** i * a[i] * np.cos(2 * np.pi * i * k / n) for i in range(5))
    raise ValueError(name)


def fft_dbv(t, y, window="hann", n=None):
    """等间隔重采样 → 加窗 → 单边幅度谱（Vrms，已按窗的相干增益修正）。返回 (f, dBV, Vrms)。
    dBV = 20·log10(Vrms / 1 V)。直流分量（f=0）按直流值本身算。"""
    t = np.asarray(t, float); y = np.asarray(y, float)
    if n is None:
        n = int(2 ** np.ceil(np.log10(max(len(t), 1024)) / np.log10(2)))
        n = min(max(n, 1024), 2 ** 18)
    tg = np.linspace(t[0], t[-1], n, endpoint=False)
    yg = np.interp(tg, t, y)
    w = _window(window, n)
    cg = w.sum() / n
    Y = np.fft.rfft(yg * w) / n / cg
    amp = np.abs(Y)
    amp[1:] *= 2  # 单边
    vrms = amp.copy(); vrms[1:] /= np.sqrt(2)
    f = np.fft.rfftfreq(n, tg[1] - tg[0])
    return f, 20 * np.log10(np.maximum(vrms, 1e-15)), vrms


def peak_near(f, vrms, f0, rel=0.15):
    """在 f0±rel 范围内找最高的谱线，返回 (频率, Vrms)。"""
    m = (f > f0 * (1 - rel)) & (f < f0 * (1 + rel))
    if not m.any():
        return None, None
    i = np.nonzero(m)[0][np.argmax(vrms[m])]
    return float(f[i]), float(vrms[i])
