"""射流（喷嘴出口速度、抛物轨迹、冲击力）与水箱放空（Torricelli 解析解）。"""
import math
import numpy as np
from ..world.fluids import G


def jet_speed(dp, rho, Cv=0.97):
    """孔前后压差 dp(Pa) → 射流速度（Cv≈0.97 速度系数）"""
    return Cv * math.sqrt(2 * dp / rho)


def trajectory(v, angle_deg, h0=0.0, n=200):
    """无空气阻力的抛物线射流（短距离、粗水柱够用；细雾化射流会被空气减速，见 notes）。
    angle 从水平线向上为正。返回 x,z,range,apex,t_flight"""
    th = math.radians(angle_deg)
    vx, vz = v * math.cos(th), v * math.sin(th)
    disc = vz ** 2 + 2 * G * h0
    tf = (vz + math.sqrt(disc)) / G
    t = np.linspace(0, tf, n)
    x = vx * t; z = h0 + vz * t - 0.5 * G * t ** 2
    return dict(x=x, z=z, range=vx * tf, apex=h0 + vz ** 2 / (2 * G), t_flight=tf, t=t)


def aim_angle(v, dx, dz):
    """打到相对喷嘴 (dx, dz) 的目标需要的仰角(两个解: 低弹道/高弹道)；够不到返回 None"""
    a = G * dx ** 2 / (2 * v ** 2)
    disc = dx ** 2 - 4 * a * (a + dz)
    if disc < 0:
        return None
    return tuple(math.degrees(math.atan((dx + s * math.sqrt(disc)) / (2 * a))) for s in (-1, 1))


def jet_force(rho, Q, v, theta_deg=90.0, deflect="plate"):
    """射流冲击平板的法向力 F = ρ Q v sinθ（θ=射流与板面夹角，90°=垂直；平板、水沿板面流走）。
    deflect='bucket' 为 180° 折返时 2ρQv。"""
    f = rho * Q * v
    return 2 * f if deflect == "bucket" else f * math.sin(math.radians(theta_deg))


def stagnation_pressure(rho, v):
    return 0.5 * rho * v ** 2


def drain_time(A_tank, a_hole, H0, H1=0.0, Cd=0.62):
    """水箱经底部小孔放空/降液位 H0→H1 的时间(s)，Torricelli: t = A/(Cd a) √(2/g) (√H0 − √H1)"""
    return A_tank / (Cd * a_hole) * math.sqrt(2 / G) * (math.sqrt(H0) - math.sqrt(H1))
