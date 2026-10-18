#!/usr/bin/env python3
"""场景C锚点: 无限长螺线管+圆柱工件 1D 解析(贝塞尔严格解, scipy ive 复数)
仅含"体内屏蔽/透入"分量, 不含端部效应(短圈端部聚磁需 2D FEM).
"""
import numpy as np
from scipy.special import ive
MU0 = 4e-7 * np.pi

def fac_workpiece(a, f, sig, mur):
    w = 2*np.pi*f
    g = np.sqrt(1j*w*sig*MU0*mur)*a
    if abs(g) < 1e-6: return complex(mur)
    return complex(mur*(2/g)*ive(1,g)/ive(0,g))

def L_per_len(a, R, n1, f, sig, mur):
    fac = fac_workpiece(a, f, sig, mur)
    Z_over_jw = MU0*n1**2*(np.pi*(R**2-a**2) + np.pi*a**2*fac)
    return Z_over_jw.real, -2*np.pi*f*Z_over_jw.imag

a, R, n1, f = 0.03, 0.04, 25.0, 15e3
L_air, _ = L_per_len(0.0, R, n1, f, 0.0, 1.0)
print(f"空心 L(40mm单匝,无限长近似)={L_air*0.04*1e6:.4f} µH (Nagaoka 短圈修正应≈0.083)")
print(f"{'工况':<12}{'μr':>5}{'δ(mm)':>8}{'L(µH)':>9}{'L/Lair':>8}{'f0偏移':>9}")
for name, mur, sig in [("冷态",100,5.0e6),("近居里",30,1.0e6),("弱磁",10,1.0e6),("热态",1,8.7e5)]:
    d = np.sqrt(2/(2*np.pi*f*MU0*mur*sig))
    L, Rr = L_per_len(a, R, n1, f, sig, mur)
    print(f"{name:<12}{mur:>5}{d*1e3:>8.3f}{L*0.04*1e6:>9.4f}{L/L_air:>8.3f}{(np.sqrt(L_air/L)-1)*100:>+8.1f}%")
print("\nμr 扫描(15kHz, σ=5e6):")
for mur in [200,100,50,30,10,5,2,1]:
    L, Rr = L_per_len(a, R, n1, f, 5.0e6, mur)
    print(f"  μr={mur:>4}: L={L*0.04*1e6:.4f} µH ({L/L_air*100:.1f}%), f0 {(np.sqrt(L_air/L)-1)*100:+.2f}%")
print("\n低频 1kHz 对照(弱屏蔽,磁芯效应显现):")
for mur in [100,30,1]:
    L, Rr = L_per_len(a, R, n1, 1e3, 5.0e6, mur)
    print(f"  μr={mur:>4}: L={L*0.04*1e6:.4f} µH ({L/L_air*100:.1f}%)")
