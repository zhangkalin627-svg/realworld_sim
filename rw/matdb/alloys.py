"""常用牌号合金：名义成分 + 实测/标准性能。

comp: 名义成分 wt%（余量元素写进来凑成 100）
rho kg/m3, E GPa, Rp 屈服(Rp0.2) MPa, Rm 抗拉 MPa, A 断后伸长 %, k W/m/K, resist Ω·m, melt (固相线, 液相线) °C, cte 1/K
basis: "min" = 国标/ASTM 规定最小值（设计可用）；"typ" = 手册典型值（MatWeb/ASM/厂家数据表）
所有数值是公开手册数据的常见值，不同炉批、规格、状态会有差别，设计时以材质证明书为准。
"""

GRADES = {
    # ---------- 钢铁
    "Q235B": dict(name="Q235B 碳素结构钢", std="GB/T 700", state="热轧",
                  comp={"Fe": 98.93, "C": 0.17, "Mn": 0.60, "Si": 0.30},
                  rho=7850, E=206, Rp=235, Rm=(370, 500), A=26, k=50, resist=1.6e-7, melt=(1420, 1510), cte=12e-6, basis="min"),
    "45": dict(name="45 优质碳素钢", std="GB/T 699", state="正火",
               comp={"Fe": 98.36, "C": 0.45, "Mn": 0.65, "Si": 0.27, "Cr": 0.15, "Ni": 0.12},
               rho=7850, E=209, Rp=355, Rm=600, A=16, k=48, resist=1.7e-7, melt=(1410, 1495), cte=11.6e-6, basis="min"),
    "40Cr": dict(name="40Cr 合金结构钢", std="GB/T 3077", state="调质(850 油淬+520 回火)",
                 comp={"Fe": 97.37, "C": 0.40, "Cr": 0.95, "Mn": 0.65, "Si": 0.27, "Ni": 0.20, "Mo": 0.10, "Cu": 0.06},
                 rho=7820, E=211, Rp=785, Rm=980, A=9, k=44, resist=2.2e-7, melt=(1410, 1490), cte=11.0e-6, basis="min"),
    "42CrMo": dict(name="42CrMo 合金结构钢", std="GB/T 3077", state="调质(850 油淬+560 回火)",
                   comp={"Fe": 96.95, "C": 0.42, "Cr": 1.05, "Mo": 0.20, "Mn": 0.75, "Si": 0.27, "Ni": 0.20, "Cu": 0.16},
                   rho=7850, E=210, Rp=930, Rm=1080, A=12, k=42.6, resist=2.2e-7, melt=(1416, 1480), cte=11.2e-6, basis="min"),
    "65Mn": dict(name="65Mn 弹簧钢", std="GB/T 1222", state="淬火+中温回火",
                 comp={"Fe": 97.68, "C": 0.65, "Mn": 1.05, "Si": 0.27, "Cr": 0.20, "Ni": 0.15},
                 rho=7810, E=200, Rp=785, Rm=980, A=8, k=46, resist=2.0e-7, melt=(1400, 1480), cte=11.0e-6, basis="min"),
    "GCr15": dict(name="GCr15 轴承钢", std="GB/T 18254", state="球化退火",
                  comp={"Fe": 96.73, "C": 1.0, "Cr": 1.50, "Mn": 0.35, "Si": 0.25, "Ni": 0.17},
                  rho=7810, E=208, Rp=420, Rm=650, A=20, k=40, resist=2.2e-7, melt=(1350, 1450), cte=11.9e-6, basis="typ"),
    "HT200": dict(name="HT200 灰铸铁", std="GB/T 9439", state="铸态",
                  comp={"Fe": 93.6, "C": 3.3, "Si": 1.8, "Mn": 0.8, "P": 0.2, "S": 0.1, "Cr": 0.2},
                  rho=7200, E=110, Rp=None, Rm=200, A=0.5, k=50, resist=8.0e-7, melt=(1150, 1250), cte=10.5e-6, basis="min"),
    "QT500-7": dict(name="QT500-7 球墨铸铁", std="GB/T 1348", state="铸态",
                    comp={"Fe": 93.4, "C": 3.6, "Si": 2.5, "Mn": 0.4, "Mg": 0.05, "Cu": 0.05},
                    rho=7100, E=169, Rp=320, Rm=500, A=7, k=36, resist=5.0e-7, melt=(1150, 1200), cte=11.0e-6, basis="min"),
    "304": dict(name="06Cr19Ni10 (304) 不锈钢", std="GB/T 20878 / ASTM A240", state="固溶",
                comp={"Fe": 70.65, "Cr": 18.5, "Ni": 9.0, "Mn": 1.5, "Si": 0.4, "C": 0.05, "N": 0.05, "Mo": 0.1, "Cu": 0.15, "P": 0.03, "S": 0.02, "Co": 0.1, "Ti": 0.0, "V": 0.0} ,
                rho=7930, E=193, Rp=205, Rm=520, A=40, k=16.2, resist=7.2e-7, melt=(1400, 1450), cte=17.2e-6, basis="min"),
    "316L": dict(name="022Cr17Ni12Mo2 (316L) 不锈钢", std="GB/T 20878 / ASTM A240", state="固溶",
                 comp={"Fe": 66.47, "Cr": 17.0, "Ni": 12.0, "Mo": 2.5, "Mn": 1.5, "Si": 0.5, "C": 0.03},
                 rho=7980, E=193, Rp=170, Rm=485, A=40, k=16.3, resist=7.4e-7, melt=(1375, 1400), cte=16.0e-6, basis="min"),
    "Invar36": dict(name="4J36 殷钢 (Invar 36)", std="GB/T 15018", state="退火",
                    comp={"Fe": 63.6, "Ni": 36.0, "Mn": 0.3, "Si": 0.1},
                    rho=8050, E=141, Rp=276, Rm=490, A=42, k=10.4, resist=8.2e-7, melt=(1427, 1427), cte=1.2e-6, basis="typ",
                    note="热膨胀只有普通钢的 1/10：这种反常（磁致伸缩抵消热膨胀）任何混合法则都算不出来"),
    # ---------- 铝 / 镁 / 钛
    "1060": dict(name="1060 工业纯铝", std="GB/T 3190", state="O 退火",
                 comp={"Al": 99.6, "Fe": 0.3, "Si": 0.1},
                 rho=2705, E=69, Rp=28, Rm=69, A=43, k=234, resist=2.83e-8, melt=(646, 657), cte=23.6e-6, basis="typ"),
    "6061-T6": dict(name="6061-T6 铝合金", std="GB/T 3190 / ASTM B209", state="T6",
                    comp={"Al": 97.92, "Mg": 1.0, "Si": 0.6, "Cu": 0.28, "Cr": 0.2},
                    rho=2700, E=68.9, Rp=240, Rm=260, A=8, k=167, resist=3.99e-8, melt=(582, 652), cte=23.6e-6, basis="min"),
    "7075-T6": dict(name="7075-T6 超硬铝", std="ASTM B209", state="T6",
                    comp={"Al": 90.07, "Zn": 5.6, "Mg": 2.5, "Cu": 1.6, "Cr": 0.23},
                    rho=2810, E=71.7, Rp=503, Rm=572, A=11, k=130, resist=5.15e-8, melt=(477, 635), cte=23.6e-6, basis="typ"),
    "2024-T3": dict(name="2024-T3 硬铝", std="ASTM B209", state="T3",
                    comp={"Al": 93.5, "Cu": 4.4, "Mg": 1.5, "Mn": 0.6},
                    rho=2780, E=73.1, Rp=345, Rm=483, A=18, k=121, resist=5.82e-8, melt=(502, 638), cte=23.2e-6, basis="typ"),
    "AZ31B": dict(name="AZ31B 镁合金", std="ASTM B90", state="O 退火板",
                  comp={"Mg": 96.0, "Al": 3.0, "Zn": 1.0},
                  rho=1770, E=45, Rp=150, Rm=255, A=21, k=96, resist=9.2e-8, melt=(605, 630), cte=26e-6, basis="typ"),
    "TA2": dict(name="TA2 工业纯钛 (Gr2)", std="GB/T 3620 / ASTM B265", state="退火",
                comp={"Ti": 99.6, "Fe": 0.3, "O": 0.1},
                rho=4510, E=105, Rp=275, Rm=345, A=20, k=16.4, resist=5.6e-7, melt=(1665, 1665), cte=8.6e-6, basis="min"),
    "TC4": dict(name="TC4 钛合金 (Ti-6Al-4V)", std="GB/T 3620 / ASTM B265", state="退火",
                comp={"Ti": 90.0, "Al": 6.0, "V": 4.0},
                rho=4430, E=113.8, Rp=880, Rm=950, A=14, k=6.7, resist=1.78e-6, melt=(1604, 1660), cte=8.6e-6, basis="typ"),
    # ---------- 铜
    "T2": dict(name="T2 紫铜", std="GB/T 5231", state="M 软态",
               comp={"Cu": 99.9, "Ag": 0.05, "O": 0.05},
               rho=8890, E=115, Rp=69, Rm=205, A=30, k=391, resist=1.72e-8, melt=(1083, 1083), cte=17.0e-6, basis="min"),
    "H62": dict(name="H62 黄铜", std="GB/T 5231", state="M 软态",
                comp={"Cu": 62.0, "Zn": 38.0},
                rho=8430, E=100, Rp=110, Rm=294, A=40, k=109, resist=7.1e-8, melt=(899, 905), cte=20.6e-6, basis="min"),
    "H68": dict(name="H68 黄铜", std="GB/T 5231", state="M 软态",
                comp={"Cu": 68.0, "Zn": 32.0},
                rho=8530, E=110, Rp=75, Rm=294, A=40, k=120, resist=6.2e-8, melt=(915, 955), cte=19.9e-6, basis="min"),
    "QSn6.5-0.1": dict(name="QSn6.5-0.1 锡磷青铜", std="GB/T 5231", state="M 软态",
                       comp={"Cu": 93.4, "Sn": 6.5, "P": 0.1},
                       rho=8800, E=110, Rp=130, Rm=315, A=40, k=62, resist=1.15e-7, melt=(950, 1050), cte=17.8e-6, basis="min"),
    "QAl9-4": dict(name="QAl9-4 铝青铜", std="GB/T 5231", state="挤制",
                   comp={"Cu": 87.0, "Al": 9.0, "Fe": 4.0},
                   rho=7500, E=110, Rp=200, Rm=540, A=15, k=59, resist=1.23e-7, melt=(1040, 1060), cte=16.2e-6, basis="min"),
    "Constantan": dict(name="6J40 康铜 (Cu55Ni45)", std="GB/T 6145", state="退火",
                       comp={"Cu": 55.0, "Ni": 44.0, "Mn": 1.0},
                       rho=8900, E=162, Rp=None, Rm=420, A=25, k=21, resist=4.9e-7, melt=(1225, 1300), cte=14.9e-6, basis="typ",
                       note="电阻率是纯铜的 28 倍、纯镍的 7 倍——固溶体的电阻率不能按比例算"),
    "Monel400": dict(name="Monel 400 蒙乃尔", std="ASTM B127", state="退火",
                     comp={"Ni": 65.5, "Cu": 31.5, "Fe": 1.7, "Mn": 1.1, "Si": 0.2},
                     rho=8800, E=179, Rp=240, Rm=550, A=40, k=21.8, resist=5.47e-7, melt=(1300, 1350), cte=13.9e-6, basis="typ"),
    # ---------- 镍基 / 电热
    "Cr20Ni80": dict(name="Cr20Ni80 镍铬电热丝", std="GB/T 1234", state="退火",
                     comp={"Ni": 78.5, "Cr": 20.5, "Si": 1.0},
                     rho=8400, E=214, Rp=None, Rm=650, A=20, k=11.3, resist=1.09e-6, melt=(1400, 1400), cte=14e-6, basis="min"),
    "IN718": dict(name="GH4169 / Inconel 718 高温合金", std="GB/T 14992 / AMS 5662", state="固溶+时效",
                  comp={"Ni": 53.0, "Cr": 19.0, "Fe": 18.0, "Nb": 5.1, "Mo": 3.0, "Ti": 0.9, "Al": 0.5, "Co": 0.5},
                  rho=8190, E=200, Rp=1034, Rm=1241, A=12, k=11.4, resist=1.25e-6, melt=(1260, 1336), cte=13.0e-6, basis="min"),
    # ---------- 焊料 / 贵金属 / 粉末冶金
    "Sn63Pb37": dict(name="Sn63Pb37 共晶焊锡", std="GB/T 3131", state="铸态",
                     comp={"Sn": 63.0, "Pb": 37.0},
                     rho=8400, E=32, Rp=None, Rm=52, A=40, k=50, resist=1.45e-7, melt=(183, 183), cte=24.7e-6, basis="typ"),
    "SAC305": dict(name="SAC305 无铅焊锡", std="GB/T 20422 / JIS Z3282", state="铸态",
                   comp={"Sn": 96.5, "Ag": 3.0, "Cu": 0.5},
                   rho=7380, E=51, Rp=None, Rm=45, A=25, k=58, resist=1.32e-7, melt=(217, 220), cte=21.6e-6, basis="typ"),
    "Ag925": dict(name="925 银 (Sterling)", std="GB 11887", state="退火",
                  comp={"Ag": 92.5, "Cu": 7.5},
                  rho=10360, E=76, Rp=None, Rm=None, A=None, k=360, resist=2.0e-8, melt=(802, 899), cte=19.0e-6, basis="typ"),
    "WCu70": dict(name="W70Cu30 钨铜（粉末冶金，渗铜）", std="GB/T 8320", state="烧结+熔渗",
                  comp={"W": 70.0, "Cu": 30.0},
                  rho=13800, E=None, Rp=None, Rm=None, A=None, k=200, resist=3.6e-8, melt=(1083, 3422), cte=9.0e-6, basis="min",
                  note="钨和铜完全不互溶，熔炼做不出来；靠压制钨骨架再渗入液态铜——不互溶的金属也能“合”起来用"),
}


def grade_list():
    return [(k, g["name"]) for k, g in GRADES.items()]


def norm_comp(comp):
    s = sum(v for v in comp.values() if v > 0)
    return {k: v * 100.0 / s for k, v in comp.items() if v > 0}


def match_grade(comp_wt, tol_abs=0.3, tol_rel=0.15):
    """配比与哪个牌号成分一致 → 返回牌号 key 或 None。
    每个元素偏差 ≤ max(tol_abs, tol_rel×含量) wt%；碳 ≤ max(0.06, 15%)；
    用户没写的、牌号里 <1% 的杂质元素不计较。"""
    c = norm_comp(comp_wt)
    best = None
    for k, g in GRADES.items():
        gc = norm_comp(g["comp"])
        keys = set(gc) | set(c)
        ok = True; err = 0.0
        for e in keys:
            a, b = c.get(e, 0.0), gc.get(e, 0.0)
            if e not in c and b < 1.0:
                continue
            tol = max(0.06, tol_rel * b) if e == "C" else max(tol_abs, tol_rel * b)
            if abs(a - b) > tol:
                ok = False; break
            err += abs(a - b)
        if ok and (best is None or err < best[1]):
            best = (k, err)
    return None if best is None else best[0]
