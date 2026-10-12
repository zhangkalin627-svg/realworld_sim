"""v0.12 通用器件库/导入器自检（公开，不含任何私有板卡数据）：
值字段解析、IGBT/NetTie/驱动/LDO/变压器/保险丝映射、保险丝 I²t 熔断时刻、耦合变压器匝比、
IEC 61000-4-5 1.2/50 µs 波形参数、三相电网激励幅值、IR2181x 行为模型 UVLO、应力表分级。
单独运行：python3 validate_circuit_v012.py；validate_circuit.py 末尾也会调用。失败返回非零。"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NET = '(export (version "E") (design (source "t")) (components %s) (libparts) (nets %s))'


def _comp(ref, value, lib, part):
    return f'(comp (ref "{ref}") (value "{value}") (libsource (lib "{lib}") (part "{part}")))'


def _net(name, *nodes):
    return f'(net (code "1") (name "{name}") ' + " ".join(
        f'(node (ref "{r}") (pin "{p}") (pinfunction "{f}"))' for r, p, f in nodes) + ")"


def run():
    from rw.circuit import devlib as DL, stress as ST, kicad as KI
    from rw.circuit.spice import Circuit, find_ngspice
    print("\n=== 8h) v0.12 器件库 / 导入器 / 应力校核 ===")
    res = []

    def chk(name, cond, detail=""):
        res.append(bool(cond))
        print("  [%s] %s %s" % ("通过" if cond else "失败", name, detail))

    # 1) 值字段带说明文字
    cases = [("C", "MKP21 0.33uF 400V", 0.33e-6), ("C", "Chemi-Con LXM 150uF 450V", 150e-6), ("L", "WE-HCF 2010 150uH 8.4A", 150e-6),
             ("R", "4k7", 4700.0), ("R", "2R2", 2.2), ("R", "68R 2512 '680'", 68.0), ("C", "100n", 100e-9)]
    bad = [(t, DL.value_for(k, t), v) for k, t, v in cases if not (DL.value_for(k, t) and abs(DL.value_for(k, t) / v - 1) < 1e-9)]
    chk("1 值字段抽数（MKP21 不当成 21、4k7/2R2 写法）", not bad, str(bad) if bad else "%d/%d" % (len(cases), len(cases)))
    r = DL.ratings_from_text("Chemi-Con LXM 150uF 450V") ; r2 = DL.ratings_from_text("WE-HCF 150uH 8.4A")
    chk("1 额定电压/电流存为参数", r.get("V") == 450 and r2.get("A") == 8.4, f"{r} {r2}")

    # 2) 小 KiCad 网表：IGBT C/G/E、NetTie、IR2181x、MIC4428、LDO、保险丝、变压器、功率二极管
    comps = " ".join([
        _comp("Q1", "IKW30N60DTP 600V 30A", "Device", "Q_NIGBT_GCE"), _comp("NT1", "NetTie", "Device", "NetTie_2"),
        _comp("U1", "IR21814S", "Driver_FET", "IR21814"), _comp("U2", "MIC4428YM", "Driver_FET", "MIC4428"),
        _comp("U3", "LP2985-5.0", "Regulator_Linear", "LP2985-5.0"), _comp("F1", "T3.15A 250V", "Device", "Fuse"),
        _comp("T1", "TR 12:1", "Device", "Transformer_1P_1S"), _comp("D1", "S4M 1000V 4A", "Device", "D"),
        _comp("D2", "SK54C 40V 5A", "Device", "D_Schottky"), _comp("D3", "bootstrap diode", "Device", "D")])
    nets = " ".join([
        _net("hv", ("Q1", "1", "C"), ("F1", "1", "~"), ("T1", "1", "AA")), _net("g", ("Q1", "2", "G"), ("U1", "7", "HO")),
        _net("sw", ("Q1", "3", "E"), ("NT1", "1", "1"), ("U1", "6", "VS"), ("T1", "2", "AB")), _net("sw2", ("NT1", "2", "2"), ("D1", "2", "A")),
        _net("GND", ("U1", "3", "COM"), ("U2", "3", "GND"), ("U3", "2", "GND"), ("T1", "4", "SB"), ("D2", "2", "A")),
        _net("vcc", ("U1", "1", "VCC"), ("U2", "6", "V+"), ("U3", "5", "VOUT"), ("D3", "2", "A")),
        _net("vb", ("U1", "5", "VB"), ("D3", "1", "K")), _net("pwm", ("U1", "2", "HIN"), ("U1", "4", "LIN"), ("U2", "2", "IN_A"), ("U2", "4", "IN_B")),
        _net("lo", ("U1", "8", "LO")), _net("oa", ("U2", "7", "~{OUT_A}")), _net("ob", ("U2", "5", "OUT_B")),
        _net("vin", ("U3", "1", "VIN"), ("U3", "3", "ON/OFF"), ("F1", "2", "~"), ("D1", "1", "K")), _net("sec", ("T1", "3", "SA"), ("D2", "1", "K"))])
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".net", delete=False) as f:
        f.write(NET % (comps, nets)); pth = f.name
    c = KI.load(pth); os.unlink(pth)
    names = {e.name.upper(): e for e in c.elements}
    w = " | ".join(c.warnings)
    chk("2 IGBT 按 C/G/E 接入（无“找不到引脚”）", "XQ1" in names and names["XQ1"].nodes == ["hv", "g", "sw"] and "找不到引脚" not in w, names.get("XQ1") and names["XQ1"].line())
    chk("2 NetTie → 0 Ω 短接", "RNT1" in names and float(names["RNT1"].value) < 1e-3)
    chk("2 IR21814 / MIC4428 / LP2985 用行为级子电路", all(k in names for k in ("XU1", "XU2", "XU3")) and "没有 SPICE 模型" not in w,
        "; ".join(names[k].line() for k in ("XU1", "XU2", "XU3") if k in names))
    chk("2 MIC4428：A 反相、B 同相", names.get("XU2") is not None and "inva=1" in names["XU2"].line() and "invb=0" in names["XU2"].line())
    chk("2 保险丝 I²t 子电路、In 读成 3.15 A", "XF1" in names and abs(c.ratings["F1"]["A"] - 3.15) < 1e-9, names.get("XF1") and names["XF1"].line())
    chk("2 变压器 → 耦合电感（匝比 12 读自值字段）", any(e.name.upper().startswith("KT1") for e in c.elements)
        and abs(float(names["LT1_P"].value) / float(names["LT1_S1"].value) / 144 - 1) < 1e-4,
        "Lp/Ls = %.4f（理论 144）" % (float(names["LT1_P"].value) / float(names["LT1_S1"].value)))
    chk("2 功率二极管库：S4M/SK54C/自举→快恢复", c.notes.get("D1", "").find("S4M") >= 0 and c.notes.get("D2", "").find("SK54C") >= 0
        and c.notes.get("D3", "").find("ES1J") >= 0, f"{c.notes.get('D1')} / {c.notes.get('D2')} / {c.notes.get('D3')}")
    chk("2 无外部激励时仍然报警（行为级子电路里的 B 源不算）", c.excitation()[0] is False, c.excitation()[1])

    if find_ngspice():
        # 3) 保险丝 I²t：10 A 直流，I²t=4 A²s → 40 ms 熔断
        k = Circuit("fuse")
        k.subckt(DL.FUSE_I2T)
        k.V("1", "a", "0", dc=10.0); k.X("F", ["a", "b"], "FUSE_I2T", i2t=4, r0="10m"); k.R("L", "b", "0", 1.0 - 0.01)
        wv = k.tran(1e-4, 80e-3, tmax=1e-4)
        t, i = np.asarray(wv.time), np.asarray(wv["v(b)"]) / 0.99
        tb = t[np.argmax(i < 5.0)] if np.any(i < 5.0) else None
        chk("3 保险丝 I²t 熔断时刻 = I²t/I²（40 ms）", tb is not None and abs(tb - 0.04) / 0.04 < 0.03, f"仿真 {tb * 1e3 if tb else float('nan'):.2f} ms，理论 40.00 ms")
        chk("3 熔断后不恢复（锁存）", float(np.max(np.abs(i[t > 0.05]))) < 1e-3, "熔断后最大电流 %.2e A" % float(np.max(np.abs(i[t > 0.05]))))

        # 4) 耦合变压器 10:1，10 kHz 100 V 正弦，副边 1 kΩ（Lp=10 mH，k=0.9999）
        k = Circuit("xf")
        k.V("1", "p", "0", sin=(0, 100, 10e3)); k.L("p", "p", "0", 10e-3); k.L("s", "s", "0", 10e-3 / 100); k.K("1", "Lp", "Ls", 0.9999)
        k.R("L", "s", "0", 1e3)
        wv = k.tran(1e-7, 1e-3, tmax=1e-7)
        vs = float(np.max(np.abs(np.asarray(wv["v(s)"])[np.asarray(wv.time) > 5e-4])))
        chk("4 耦合电感匝比 10:1 → 副边峰值 10 V", abs(vs - 10) / 10 < 0.02, "副边峰值 %.3f V" % vs)

        # 5) 三相电网 380 V 额定：线电压峰值 537.4 V
        k = Circuit("grid")
        for ph in ("L1", "L2", "L3"):
            k.R(ph + "y", ph, "y", 1e3)
        desc = DL.add_grid(k, preset="nom", zs=(1e-9, 1e-6))
        wv = k.tran(2e-5, 40e-3, tmax=2e-5)
        vll = float(np.max(np.abs(np.asarray(wv["v(L1)"]) - np.asarray(wv["v(L2)"]))))
        chk("5 三相 380 V 电网激励：线电压峰值 380·√2", abs(vll - 380 * 2 ** 0.5) / 537.4 < 0.01, "%.1f V（理论 537.4 V）" % vll)

        # 6) IR2181x 行为模型：VCC 15 V 时 HO 跟随 HIN；VCC 7 V（< UVLO 8.6 V）时输出全低
        for vcc, expect in ((15.0, True), (7.0, False)):
            k = Circuit("drv")
            k.subckt(DL.IR2181X_BEH)
            k.V("cc", "vcc", "0", dc=vcc); k.V("b", "vb", "vs", dc=vcc); k.V("s", "vs", "0", dc=0)
            k.V("in", "in", "0", pulse=(0, 5, 1e-6, 10e-9, 10e-9, 5e-6, 10e-6))
            k.X("U", ["vcc", "in", "in", "0", "lo", "vb", "ho", "vs"], "IR2181X_BEH")
            k.R("ho", "ho", "vs", 10e3); k.R("lo", "lo", "0", 10e3)
            wv = k.tran(10e-9, 20e-6, tmax=10e-9)
            t, ho = np.asarray(wv.time), np.asarray(wv["v(ho)"])
            hi = float(np.max(ho))
            if expect:
                t50 = t[np.argmax(ho > vcc / 2)]
                chk("6 IR2181x VCC=15 V：HO 高电平≈VB、传播延迟≈180 ns", hi > 0.95 * vcc and 120e-9 < t50 - 1e-6 < 260e-9,
                    "HO 峰值 %.2f V，延迟 %.0f ns" % (hi, (t50 - 1e-6) * 1e9))
            else:
                chk("6 IR2181x VCC=7 V（欠压锁定）：HO 保持低", hi < 0.5, "HO 峰值 %.3f V" % hi)
    else:
        print("  没找到 ngspice，跳过仿真类检查")

    # 7) 1.2/50 µs 浪涌波形参数（IEC 61000-4-5 附录 A 公式）
    t = np.linspace(0, 200e-6, 400001)
    v = DL.surge_v(t, 1000.0)
    pk = float(v.max()); i30, i90 = np.argmax(v > 0.3 * pk), np.argmax(v > 0.9 * pk)
    tf = 1.67 * (t[i90] - t[i30]); t_half = t[np.where(v > 0.5 * pk)[0][-1]] - (t[i30] - 0.3 / 0.6 * (t[i90] - t[i30]))
    chk("7 浪涌：峰值 1000 V、前沿 1.2 µs、半峰 50 µs（±10 %）", abs(pk - 1000) < 20 and abs(tf - 1.2e-6) < 0.12e-6 and abs(t_half - 50e-6) < 5e-6,
        "峰值 %.0f V，T1 %.2f µs，T2 %.1f µs" % (pk, tf * 1e6, t_half * 1e6))

    # 8) 应力表分级：600 V 器件在 380/418/456 V 母线（≈电网线电压峰值）下 90/98/108 %
    rows = ST.table([("D", "反压", 380 * 2 ** 0.5, 600), ("D", "反压", 418 * 2 ** 0.5, 600), ("D", "反压", 456 * 2 ** 0.5, 600), ("R", "功率", 0.5, 1)])
    chk("8 应力分级 >80 % 黄、>100 % 红", [r["level"] for r in rows] == ["warn", "warn", "over", "ok"],
        " / ".join("%.1f%%" % r["pct"] for r in rows))

    # 9) ngspice 异常退出（被信号杀/内存不足）要明确报错，不解析残缺输出（issue 20261012-1946）
    import subprocess, tempfile
    from rw.circuit import spice as SPm, board as BDm
    CP = subprocess.CompletedProcess
    try:
        SPm.check_exit(CP([], -9, "", "Killed")); ok9 = False
    except SPm.SpiceError as e:
        ok9 = "异常退出" in str(e) and "-9" in str(e) and "SIGKILL" in str(e)
    try:
        SPm.check_exit(CP([], 137, "", "")); ok137 = False
    except SPm.SpiceError:
        ok137 = True
    chk("9 check_exit：-9/137 抛带中文说明的错误，0 与普通非零不抛", ok9 and ok137 and SPm.check_exit(CP([], 0, "", "")) is None and "返回码 1" in SPm.check_exit(CP([], 1, "", "")))
    if find_ngspice():
        d = tempfile.mkdtemp(prefix="rwk_"); cir = os.path.join(d, "t.cir")
        open(cir, "w").write("* t\nV1 a 0 1\nR1 a 0 1k\n.tran 1u 10u\n.control\nrun\nwrdata t.dat v(a)\n.endc\n.end\n")
        _orig = BDm.subprocess.run
        try:
            BDm.subprocess.run = lambda *a, **k: CP(a[0], -9, "", "Killed")
            try:
                BDm.run_cir(cir, workdir=d); k1 = False
            except SPm.SpiceError as e:
                k1 = "异常退出" in str(e)
            def trunc(*a, **k):
                open(os.path.join(d, "t.dat"), "w").write(" 0.0 1.0 \n 1e-6 1.0 \n 2e-6 \n")
                return CP(a[0], 0, "", "")
            BDm.subprocess.run = trunc
            try:
                BDm.run_cir(cir, workdir=d); k2 = False
            except RuntimeError as e:
                k2 = "残缺" in str(e)
        finally:
            BDm.subprocess.run = _orig
        chk("9 run_cir：被杀 → 明确报错；wrdata 截断 → 明确报错", k1 and k2, "被杀 %s，截断 %s" % (k1, k2))

    # 1955 第3条：拓扑告警（无直流通路/悬空/电阻跨度），常规示例电路不得误报
    from rw.circuit.spice import Circuit as _C
    from rw.circuit import examples as _EX
    t1 = _C(); t1.I("1", "0", "x", dc=1e-3); t1.C("1", "x", "0", "1u")
    t2 = _C(); t2.V("1", "a", "0", dc=1); t2.R("1", "a", "b", 1e3)
    t3 = _C(); t3.V("1", "a", "0", dc=1); t3.R("1", "a", "0", 1); t3.R("2", "a", "0", 1e-15)
    t4 = _C(); t4.V("1", "a", "0", dc=1); t4.R("1", "a", "b", 1e3); t4.C("1", "b", "0", "1u"); t4.R("2", "b", "0", 1e3)
    fp = [k for k in getattr(_EX, "NAMES", []) if _EX.get(k).circuit.topology_warnings()]
    chk("10 拓扑告警：电流源割集/悬空/电阻跨度各报，正常电路与全部内置示例不误报",
        any("直流通路" in w for w in t1.topology_warnings()) and any("悬空" in w for w in t2.topology_warnings())
        and any("跨度" in w for w in t3.topology_warnings()) and not t4.topology_warnings() and not fp,
        "误报示例 %s" % (fp or "无"))

    n = sum(res)
    print("  v0.12 器件库/导入器/应力：%d/%d 通过" % (n, len(res)))
    return n == len(res)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
