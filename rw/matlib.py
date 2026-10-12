"""结构材料库（典型值）。E: Pa, rho: kg/m3, fy: 屈服 Pa, Tmax: 长期使用温度上限 °C
magnetic: 是否铁磁（会被高频磁场强烈加热/吸磁）, resist: 电阻率 Ω·m"""

MAT = {
    "Q235":    dict(name="Q235 碳钢",        E=206e9, nu=0.3,  rho=7850, fy=235e6, fu=4e+08, Tmax=350, magnetic=True,  resist=1.6e-7, mur=200),
    "45":      dict(name="45 钢(调质)",       E=206e9, nu=0.3,  rho=7850, fy=355e6, fu=6.5e+08, Tmax=400, magnetic=True,  resist=1.7e-7, mur=200),
    "6061":    dict(name="6061-T6 铝",       E=69e9,  nu=0.33, rho=2700, fy=240e6, fu=3.1e+08, Tmax=150, magnetic=False, resist=4.0e-8, mur=1),
    "6063":    dict(name="6063-T5 铝型材",    E=69e9,  nu=0.33, rho=2700, fy=110e6, fu=1.6e+08, Tmax=150, magnetic=False, resist=3.3e-8, mur=1),
    "SUS304":  dict(name="SUS304 不锈钢",     E=193e9, nu=0.29, rho=7930, fy=205e6, fu=5.2e+08, Tmax=800, magnetic=False, resist=7.2e-7, mur=1.02),
    "2Cr13":   dict(name="2Cr13 不锈钢(调质)", E=200e9, nu=0.3,  rho=7750, fy=440e6, fu=7e+08, Tmax=500, magnetic=True,  resist=5.5e-7, mur=300),
    "Cu":      dict(name="T2 紫铜",           E=117e9, nu=0.34, rho=8940, fy=70e6, fu=2.1e+08,  Tmax=200, magnetic=False, resist=1.72e-8, mur=1),
    "Brass":   dict(name="H62 黄铜",          E=100e9, nu=0.34, rho=8500, fy=150e6, fu=3.3e+08, Tmax=250, magnetic=False, resist=7.1e-8, mur=1),
    "PP":      dict(name="PP 聚丙烯",         E=1.3e9, nu=0.42, rho=905,  fy=30e6, fu=3e+07,  Tmax=80,  magnetic=False, resist=None,   mur=1, creep_fy=6e6),
    "POM":     dict(name="POM 聚甲醛",        E=2.8e9, nu=0.38, rho=1410, fy=65e6, fu=6.5e+07,  Tmax=100, magnetic=False, resist=None,   mur=1, creep_fy=15e6),
    "PU":      dict(name="PU 聚氨酯 A90",     E=0.05e9, nu=0.48, rho=1200, fy=30e6, fu=3e+07, Tmax=80,  magnetic=False, resist=None,   mur=1),
    "PC":      dict(name="PC 耐力板",         E=2.3e9, nu=0.37, rho=1200, fy=60e6, fu=7e+07,  Tmax=115, magnetic=False, resist=None,   mur=1),
    "MIX":     dict(name="外购件(按样本重量)", E=200e9, nu=0.3,  rho=3000, fy=200e6, fu=2e+08, Tmax=80,  magnetic=True,  resist=None,   mur=1),
}

# 螺栓 (公称应力截面 As mm2, 保证/屈服强度 Pa)
BOLT_AS = {4: 8.78, 5: 14.2, 6: 20.1, 8: 36.6, 10: 58.0, 12: 84.3, 16: 157, 20: 245}
BOLT_GRADE = {
    "8.8":   dict(name="8.8 级碳钢", fy=640e6, magnetic=True),
    "A2-70": dict(name="A2-70 (304)", fy=450e6, magnetic=False),
    "4.8":   dict(name="4.8 级", fy=320e6, magnetic=True),
    "PA66":  dict(name="尼龙 PA66 螺钉", fy=60e6, magnetic=False, Tmax=100),
}
