"""生成 elements.json（118 种元素基础物性）。只在开发时运行一次，结果随软件发布。

数据来源（均为 MIT 许可的开源数据库）：
  - pymatgen (Materials Project) periodic_table.json：密度、熔点、沸点、电阻率、导热、弹性模量、
    剪切/体积模量、泊松比、线膨胀、布氏硬度、金属半径
  - mendeleev (L. Mentel) elements.db：晶体结构、晶格常数、比热、电负性、Miedema 参数、参考价格、放射性
用法:  python3 -m rw.matdb.build_elements <pymatgen periodic_table.json.gz> <mendeleev elements.db>
"""
import gzip, json, sqlite3, sys, os

ZH = ('氢氦锂铍硼碳氮氧氟氖钠镁铝硅磷硫氯氩钾钙钪钛钒铬锰铁钴镍铜锌镓锗砷硒溴氪铷锶钇锆铌钼锝钌铑钯银镉铟锡锑碲碘氙'
      '铯钡镧铈镨钕钷钐铕钆铽镝钬铒铥镱镥铪钽钨铼锇铱铂金汞铊铅铋钋砹氡钫镭锕钍镤铀镎钚镅锔锫锎锿镄钔锘铹'
      '𬬻𬭊𬭳𬭛𬭶鿏𫟼𬬭鿔鿭𫓧镆𫟷鿬鿫')
STRUCT = {"BCC": "体心立方 bcc", "FCC": "面心立方 fcc", "HEX": "密排六方 hcp", "DIA": "金刚石结构", "CUB": "立方(复杂)",
          "SC": "简单立方", "TET": "四方", "ORC": "正交", "MCL": "单斜", "RHL": "菱方"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build(pmg_path, db_path, out):
    P = json.load(gzip.open(pmg_path))
    c = sqlite3.connect(db_path)
    cols = [r[1] for r in c.execute("pragma table_info(elements)")]
    rows = {r[cols.index("atomic_number")]: dict(zip(cols, r)) for r in c.execute("select * from elements")}
    pt = {}
    for r in c.execute("select atomic_number, melting_point, boiling_point from phasetransitions"):
        pt.setdefault(r[0], r[1:])
    els = []
    for Z in range(1, 119):
        m = rows[Z]; s = m["symbol"]; p = P.get(s, {})
        mp = num(p.get("Melting point")) or (pt.get(Z, (None, None))[0])
        bp = num(p.get("Boiling point")) or (pt.get(Z, (None, None))[1])
        rho = num(p.get("Density of solid"))
        rho_src = "solid"
        if rho is None and m["density"] is not None:
            rho = m["density"] * 1000; rho_src = "mendeleev"
        mr = num(p.get("Metallic radius"))
        mr = mr * 100 if mr else m.get("metallic_radius_c12")
        cp = m["specific_heat_capacity"]
        e = dict(
            Z=Z, sym=s, name=m["name"], zh=ZH[Z - 1], mass=m["atomic_weight"], period=m["period"], group=m["group_id"],
            block=m["block"],
            density=rho, density_src=rho_src,                      # kg/m3
            melt_C=None if mp is None else mp - 273.15, boil_C=None if bp is None else bp - 273.15,
            resist=num(p.get("Electrical resistivity")),          # Ω·m
            k=num(p.get("Thermal conductivity")) or m["thermal_conductivity"],  # W/m/K
            cp=None if cp is None else cp * 1000,                  # J/kg/K
            E=num(p.get("Youngs modulus")), G=num(p.get("Rigidity modulus")), K=num(p.get("Bulk modulus")),  # GPa
            nu=num(p.get("Poissons ratio")), cte=num(p.get("Coefficient of linear thermal expansion")),
            HB=num(p.get("Brinell hardness")),                     # MPa
            struct=m["lattice_structure"], a=m["lattice_constant"],
            r_met=mr, chi=m["en_pauling"],
            mied_phi=m["en_miedema"], mied_V=m["miedema_molar_volume"], mied_n=m["miedema_electron_density"],
            price=m["price_per_kg"], radioactive=bool(m["is_radioactive"]),
        )
        if rho_src == "mendeleev":
            if e["boil_C"] is not None and e["boil_C"] < 25:
                e["density_src"] = "gas_25C"          # 气体：25°C、1 atm 下的密度
            elif Z >= 99 or s in ("At", "Fr"):
                e["density_src"] = "predicted"       # 只能理论预测，没人称过
            elif e["melt_C"] is not None and e["melt_C"] < 25:
                e["density_src"] = "liquid"
        e["phase"] = ("气" if e["boil_C"] is not None and e["boil_C"] < 25 else
                      "液" if e["melt_C"] is not None and e["melt_C"] < 25 else
                      "固" if e["melt_C"] is not None else "未知")
        els.append(e)
    meta = dict(sources=["pymatgen periodic_table.json (MIT, Materials Project)", "mendeleev elements.db v1.3 (MIT)"],
                units=dict(density="kg/m3", melt_C="°C", resist="Ω·m", k="W/m/K", cp="J/kg/K", E="GPa", G="GPa",
                           K="GPa", cte="1/K", HB="MPa", a="Å", r_met="pm (CN12)", price="USD/kg"))
    json.dump(dict(meta=meta, elements=els), open(out, "w"), ensure_ascii=False, indent=0)
    print("wrote", out, len(els))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], os.path.join(os.path.dirname(__file__), "elements.json"))
