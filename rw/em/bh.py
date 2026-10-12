"""非线性 B-H 曲线 + 描述函数 (谐波平衡/基波法) + 导磁体材料库

物理:
  铁磁体磁化曲线用 Fröhlich–Kennelly 形式 (各向同性、无磁滞的"首次磁化/平均磁化"曲线):
        B(H, T) = mu0*H + Js(T) * H / (|H| + Hk(T))
        Js(T)   = Js0 * (1 - (T+273.15)/(Tc+273.15))^beta        (T<Tc;  T>=Tc → 0, 纯顺磁 μr=1)
        Hk(T)   = Js(T) / (mu0*(mu_i-1))                          (保持初始磁化率 mu_i-1 不随温度变)
  这是工程近似: 真实钢的曲线要用该炉钢的实测 B-H，曲线形状(尤其膝点)和 Hopkinson 峰都没有建模。
  感应加热时表面磁场强度 H ~ 1e4–1e5 A/m，早已深度饱和，所以决定性的是 Js 与居里点，而不是膝点细节。

  谐波平衡 (描述函数法): 磁场按 H(t)=H0 sin(wt) 变化时，磁感应强度 B(t) 非正弦，
  取其基波幅值 B1(H0) = (1/pi) ∫ B(H0 sin φ) sin φ dφ，
  有效磁导率 mu_eff = B1 / (mu0*H0)。用于时谐有限元的单元磁导率。
  比"B峰值/H峰值"更准确（后者在深度饱和时高估 μ 约 30–40%）。
  由有限元得到 |B1| 后反查 H0 (表格插值)，迭代到自洽。
"""
import numpy as np

MU0 = 4e-7 * np.pi

# 描述函数 g(x) = (4/pi) ∫0^{pi/2} x sin^2φ/(x sinφ+1) dφ ,  x = H0/Hk
_X = np.logspace(-8, 9, 700)
_PHI, _W = np.polynomial.legendre.leggauss(64)
_PHI = 0.25 * np.pi * (_PHI + 1.0)
_W = _W * 0.25 * np.pi
_s = np.sin(_PHI)[None, :]
_G = (4 / np.pi) * (_W[None, :] * (_X[:, None] * _s**2 / (_X[:, None] * _s + 1.0))).sum(axis=1)


def g_fun(x):
    """Fröhlich 磁化部分的基波系数 (M1/Js)。x=H0/Hk"""
    x = np.asarray(x, float)
    return np.interp(np.log(np.maximum(x, 1e-8)), np.log(_X), _G)


class BHCurve:
    """温度相关的饱和铁磁体 B-H 模型。参数:
    Js0: 0 K 外推附近的饱和极化强度 (T)，在常温下近似取 20°C 值
    mu_i: 初始(弱场)相对磁导率   Tc: 居里点 °C    beta: 临界指数(Fe≈0.37)
    """

    def __init__(self, name="45钢", Js0=2.0, mu_i=200.0, Tc=770.0, beta=0.37, T_ref=20.0):
        self.name, self.Js0, self.mu_i, self.Tc, self.beta = name, Js0, mu_i, Tc, beta
        # 让 20°C 时的 Js 恰好等于 Js0
        self._norm = (1 - (T_ref + 273.15) / (Tc + 273.15)) ** beta if np.isfinite(Tc) else 1.0

    def Js(self, T):
        T = np.asarray(T, float)
        if not np.isfinite(self.Tc):
            return np.full_like(T, self.Js0)
        x = 1 - (T + 273.15) / (self.Tc + 273.15)
        return self.Js0 * np.clip(x, 0, None) ** self.beta / self._norm

    def Hk(self, T):
        return np.maximum(self.Js(T), 1e-12) / (MU0 * (self.mu_i - 1.0))

    def B(self, H, T):
        """静态(无磁滞)B-H。H 可为负。"""
        H = np.asarray(H, float)
        return MU0 * H + self.Js(T) * H / (np.abs(H) + self.Hk(T))

    def B1(self, H0, T):
        """幅值为 H0 的正弦磁场下，B 的基波幅值 (T)。"""
        H0 = np.asarray(H0, float)
        return MU0 * H0 + self.Js(T) * g_fun(H0 / self.Hk(T))

    def mu_eff(self, H0, T):
        """基波有效相对磁导率"""
        H0 = np.maximum(np.asarray(H0, float), 1e-3)
        return self.B1(H0, T) / (MU0 * H0)

    def H0_from_B1(self, B1, T):
        """给定基波磁密幅值，反查磁场幅值 H0 (A/m)。逐元素二分(对数域)。"""
        B1 = np.atleast_1d(np.asarray(B1, float))
        T = np.broadcast_to(np.asarray(T, float), B1.shape)
        lo = np.full(B1.shape, 1e-3)
        hi = np.full(B1.shape, 1e9)
        for _ in range(60):
            mid = np.sqrt(lo * hi)
            hot = self.B1(mid, T) > B1
            hi = np.where(hot, mid, hi)
            lo = np.where(hot, lo, mid)
        return np.sqrt(lo * hi)

    def mur_from_B1(self, B1, T):
        """给定基波磁密幅值 → 有效相对磁导率 (自洽)。"""
        B1 = np.maximum(np.asarray(B1, float), 1e-9)
        H0 = self.H0_from_B1(B1, T)
        return np.maximum(B1 / (MU0 * H0), 1.0)


# 45 钢: 常温饱和极化强度约 2.0 T (含碳量 0.45% 略低于纯铁 2.15 T)；居里点约 770°C
STEEL45_BH = BHCurve("45钢", Js0=2.0, mu_i=200.0, Tc=770.0, beta=0.37)


# ---------------------------------------------------------------- 导磁体 (磁集中器)
class FluxMaterial:
    """导磁体材料: 磁化曲线(上面的模型，不随温度变) + 损耗模型。
    损耗密度 (W/m3, B 为基波幅值 T, f 为频率):
        P_hys = 4*Hc*B*f              矩形磁滞回线的上限估计
        P_eddy = pi^2 d^2 f^2 B^2 / (6*rho)   经典涡流损耗 (d=叠片厚度或等效颗粒/层厚, rho=电阻率)
    ⚠ 下面三种是"典型量级"预设，不是某个牌号的数据；请按供应商数据表改。
    """

    def __init__(self, name, mu_i, Js, Hc, d, rho, note=""):
        self.name, self.Hc, self.d, self.rho, self.note = name, Hc, d, rho, note
        self.curve = BHCurve(name, Js0=Js, mu_i=mu_i, Tc=np.inf)

    def loss_density(self, B, f):
        B = np.asarray(B, float)
        ph = 4.0 * self.Hc * B * f
        pe = (np.pi * self.d * f) ** 2 * B**2 / (6.0 * self.rho)
        return ph + pe, ph, pe


FLUX_MATERIALS = {
    "sife": FluxMaterial("硅钢叠片 (0.1 mm 薄带)", mu_i=1500.0, Js=2.0, Hc=30.0, d=0.1e-3, rho=4.7e-7,
                         note="高 μ、高 Bs，但涡流损耗随 f² 增长，30 kHz 以上很难用"),
    "smc": FluxMaterial("软磁复合材料 SMC (绝缘铁粉压制)", mu_i=25.0, Js=1.4, Hc=250.0, d=0.15e-3, rho=1e-4,
                        note="μ 低但各向同性、损耗随 f 增长较慢，几 kHz~几百 kHz 常用"),
    "ferrite": FluxMaterial("MnZn 铁氧体", mu_i=1500.0, Js=0.45, Hc=15.0, d=0.2e-3, rho=1.0,
                            note="μ 高、几乎无涡流，但 Bs 只有 0.4–0.5 T，感应加热强场下立刻饱和，且居里点低(~200°C)易碎"),
}
