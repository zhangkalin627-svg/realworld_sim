"""一维半无限平板的非线性涡流参考解 (用来检验二维/三维有限元的非线性处理)

两种解法，材料完全相同 (B-H 来自 rw.em.bh):
  1) 时域 (参考真值): 磁扩散方程  dB/dt = d/dx( rho dH/dx ), B=B(H) 无磁滞静态曲线，
     表面 H(0,t)=H0 sin(wt)，积分到周期稳态，功率 = 一个周期内焦耳热的平均 ∫ rho J^2 dx。
     (没有任何"等效磁导率"假设，B 的非正弦、谐波全部自动包含。)
  2) 谐波平衡 (描述函数) : 只保留基波，每个点的 μ 由当地 |B1| 自洽迭代得到
     —— 这正是有限元里 `nonlinear=True` 用的方法，所以两者的差就是该近似的误差。
线性极限: μr 常数时两者都应回到解析解 P = 0.5*H0^2*sqrt(pi f mu rho)  (单位面积)。
"""
import numpy as np
from scipy.integrate import solve_ivp
from .bh import MU0


def linear_power(H0, f, mur, rho):
    """半无限线性导体单位面积吸收功率 (W/m2): 0.5*H0^2*Rs, Rs=sqrt(pi f mu rho)"""
    return 0.5 * H0**2 * np.sqrt(np.pi * f * MU0 * mur * rho)


def _grid(delta, depth_factor=8.0, n=200):
    # 表面密、向内几何增长
    L = depth_factor * delta
    x = np.concatenate([[0.0], np.geomspace(delta / 400, L, n)])
    return x


def time_domain(curve, T, rho, f, H0, n_per=200, periods=5, n=120):
    """时域参考解。curve: BHCurve; T: 恒定温度 °C; rho: 电阻率; 返回单位面积平均功率 W/m2"""
    d_lin = np.sqrt(rho / (np.pi * f * MU0))            # 真空磁导率下的透入深度，作为尺度上限
    x = _grid(d_lin, 8.0, n)
    N = len(x)
    Tn = float(T)

    def dBdH(H):
        e = np.maximum(np.abs(H) * 1e-4, 1e-3)
        return (curve.B(H + e, Tn) - curve.B(H - e, Tn)) / (2 * e)

    h = np.diff(x)

    def rhs(t, Hi):
        H = np.concatenate([[H0 * np.sin(2 * np.pi * f * t)], Hi, [0.0]])
        dH = np.diff(H) / h
        d2 = (dH[1:] - dH[:-1]) / (0.5 * (h[1:] + h[:-1]))
        return rho * d2 / dBdH(Hi)

    T_per = 1.0 / f
    y0 = np.zeros(N - 2)
    tt = np.linspace(0, periods * T_per, periods * n_per + 1)
    import scipy.sparse as sp
    sps = sp.diags([1, 1, 1], [-1, 0, 1], shape=(N - 2, N - 2))
    sol = solve_ivp(rhs, (0, tt[-1]), y0, t_eval=tt, method="BDF", rtol=1e-5, atol=1e-2, jac_sparsity=sps)
    H = np.vstack([H0 * np.sin(2 * np.pi * f * sol.t), sol.y, np.zeros_like(sol.t)])
    J = np.diff(H, axis=0) / h[:, None]
    # 最后一个周期: 逐时刻焦耳功率, 单元 J 取区间值
    last = sol.t >= (periods - 1) * T_per - 1e-12
    q = rho * np.sum(J[:, last] ** 2 * h[:, None], axis=0)
    return float(np.mean(q))


def harmonic(curve, T, rho, f, H0, n=120, iters=60):
    """谐波平衡 (基波) 解: 每点 μ 自洽。返回 (P W/m2, 深度坐标 x, 当地 mur)"""
    w = 2 * np.pi * f
    d_lin = np.sqrt(rho / (np.pi * f * MU0))
    x = _grid(d_lin, 8.0, n)
    N = len(x)
    # 节点 H 的 ODE:  d/dx( rho dH/dx ) = j w mu H  (H 复数幅值) ; 单元上 μ 取单元中点
    h = np.diff(x)
    mur = np.full(N - 1, 100.0)
    Tn = np.full(N - 1, float(T))
    for it in range(iters):
        mu = MU0 * mur
        # 三对角: 节点 i=1..N-2 未知
        main = np.zeros(N, complex); lo = np.zeros(N, complex); up = np.zeros(N, complex)
        # -rho*(H_{i+1}-H_i)/h_i + rho*(H_i-H_{i-1})/h_{i-1} + j w mu_i H_i * hbar = 0
        for i in range(1, N - 1):
            hl, hr = h[i - 1], h[i]
            hb = 0.5 * (hl + hr)
            mb = (mu[i - 1] * hl + mu[i] * hr) / (hl + hr)
            main[i] = rho / hl + rho / hr + 1j * w * mb * hb
            lo[i] = -rho / hl
            up[i] = -rho / hr
        A = np.zeros((N, N), complex)
        for i in range(1, N - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = lo[i], main[i], up[i]
        A[0, 0] = 1.0
        A[N - 1, N - 1] = 1.0
        b = np.zeros(N, complex); b[0] = H0
        Hc = np.linalg.solve(A, b)
        # 单元内: J = dH/dx (幅值), B1 = mu*|H|  (单元中点 H 取平均)
        Hm = 0.5 * np.abs(Hc[:-1] + Hc[1:])
        Hmid = np.maximum(0.5 * (np.abs(Hc[:-1]) + np.abs(Hc[1:])), 1.0)
        # 自洽: 需要 |H1| → mur ;  B1(H0=|H|)=mu0*mur*|H|  → mur = mu_eff(|H|)
        new = curve.mu_eff(Hmid, Tn)
        new = np.maximum(new, 1.0)
        mur_new = np.exp(0.5 * np.log(mur) + 0.5 * np.log(new))
        if np.max(np.abs(np.log(mur_new / mur))) < 1e-4:
            mur = mur_new
            break
        mur = mur_new
    J = np.diff(Hc) / h
    P = 0.5 * rho * np.sum(np.abs(J) ** 2 * h)
    return float(P), x, mur
