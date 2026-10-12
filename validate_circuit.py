"""v0.11.1 电路导入回归（通用，公开）：6 个电路引擎修复的自检。被 validate.py 调用，也可单独运行：python3 validate_circuit.py"""
import os, re, subprocess, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _ngspice_print(path):
    """直接 ngspice -b 跑（文件自带 .control print），返回 {名: 值}。"""
    from rw.circuit.spice import find_ngspice, _env_for
    exe = find_ngspice()
    r = subprocess.run([exe, "-b", path], capture_output=True, text=True, timeout=60, env=_env_for(exe), cwd=os.path.dirname(path))
    return {m.group(1).lower(): float(m.group(2)) for m in re.finditer(r"^(v\([^)]+\))\s*=\s*([-+0-9.e]+)", r.stdout, re.M)}


NET_HEAD = '(export (version "E") (design (source "t")) (components %s) (libparts %s) (nets %s))'


def _net(comps, libparts, nets):
    return NET_HEAD % (comps, libparts, nets)


def run():
    from rw.circuit.spice import find_ngspice, Waveforms, Circuit
    from rw.circuit import kicad as KI, board as B
    from rw.circuit.scope import Scope
    print("\n=== 8g) v0.11.1 电路导入回归 ===")
    if not find_ngspice():
        print("没找到 ngspice，跳过"); return
    ok_all = []

    def chk(name, cond, detail=""):
        ok_all.append(bool(cond))
        print("  [%s] %s %s" % ("通过" if cond else "失败", name, detail))
        assert cond, name

    # ---- 1) .options 不能被当成 .op 吃掉
    p = os.path.join(HERE, "models", "netlists", "options_test.cir")
    c = KI.load(p)
    ref = _ngspice_print(p)
    w = c.run(KI.default_analysis(c, ".op"))
    v2, v3 = float(w["2"][0]), float(w["3"][0])
    chk("1 .options/.option 不进 analyses、原样进网表", c.analyses == [".op"] and sum(l.lower().startswith(".option") for l in c.extra) == 2,
        "analyses=%s" % c.analyses)
    c0 = KI.load(p); c0.extra = [l for l in c0.extra if not l.lower().startswith(".option")]
    v2_noopt = float(c0.run(".op")["2"][0])
    chk("1 含 .options 的结果 = 直接 ngspice", abs(v2 / ref["v(2)"] - 1) < 1e-5 and abs(v3 - ref["v(3)"]) <= 1e-5 * max(abs(v3), 1e-12) + 1e-12,
        "v(2) 导入 %.6f / ngspice %.6f，v(3) %.4e / %.4e（去掉 .options 会是 v(2)=%.6f）" % (v2, ref["v(2)"], v3, ref["v(3)"], v2_noopt))
    chk("1 .options 确实改变了结果（rshunt 生效）", abs(v2 - v2_noopt) > 0.05)

    # ---- 2/3) .net 类型判断 + 引脚数
    c = KI.load(os.path.join(HERE, "models", "netlists", "iec_designators_kicad.net"))
    want = {"V1": "D", "V2": "PNP", "V3": "NMOS", "D1": "IC", "G1": "V", "F1": "FUSE", "RV1": "POT", "U9": None, "X1": "PASSIVE"}
    got = {r: c.kinds[r][0] for r in want}
    chk("2 IEC 位号：V1=二极管 V2=PNP V3=NMOS D1=隔离芯片(不是二极管) …", got == want, str({k: v for k, v in got.items() if v != want[k]} or ""))
    names = {e.name.upper(): e for e in c.elements}
    chk("2 V 开头二极管导入成 D 元件、D 开头 IC 没被当成二极管", "DV1" in names and not any(n.startswith("D") and "D1" == n for n in names))
    chk("2 判断不出的器件给警告", any(w_.startswith("U9") and "判断不出" in w_ for w_ in c.warnings))
    chk("3 三脚器件按 3 脚：PNP→Q(3 节点)、NMOS→M(3 节点)", len(names["QV2"].nodes or names["QV2"].line().split()[1:4]) == 3 and names["MV3"].kind == "M")
    chk("3 多脚无模型器件给警告", any(w_.startswith("D1") and "没有 SPICE 模型" in w_ for w_ in c.warnings))
    w = c.run(".op")
    vd = float(w["/a"][0] - w["/k"][0])
    chk("2/3 导入后能算：二极管压降 0.5–0.8 V、PNP 导通、NMOS 导通", 0.5 < vd < 0.8 and float(w["/c"][0]) > 10 and float(w["/d"][0]) < 0.1,
        "Vd=%.3f V，V(/c)=%.2f V，V(/d)=%.4f V" % (vd, float(w["/c"][0]), float(w["/d"][0])))
    # 双二极管（A1/A2/K 共阴）+ 4 脚 MOSFET（带 B 脚）+ 同名多脚（SO-8 型 D×4/S×3）
    comps = ('(comp (ref "V9") (value "BAT54C") (libsource (lib "Diode") (part "BAT54C")))'
             '(comp (ref "Q5") (value "IRF7413") (fields (field (name "Sim.Device") "NMOS")) (libsource (lib "Transistor_FET") (part "SO8_NMOS")))'
             '(comp (ref "R1") (value "1k") (libsource (lib "Device") (part "R")))'
             '(comp (ref "V5") (value "5") (fields (field (name "Sim.Device") "V") (field (name "Sim.Params") "dc=5")) (libsource (lib "Simulation_SPICE") (part "VDC")))')
    libp = ('(libpart (lib "Diode") (part "BAT54C") (pins (pin (num "1") (name "A1")) (pin (num "2") (name "A2")) (pin (num "3") (name "K"))))'
            '(libpart (lib "Transistor_FET") (part "SO8_NMOS") (pins (pin (num "1") (name "S")) (pin (num "2") (name "S")) (pin (num "3") (name "S")) (pin (num "4") (name "G"))'
            ' (pin (num "5") (name "D")) (pin (num "6") (name "D")) (pin (num "7") (name "D")) (pin (num "8") (name "D"))))'
            '(libpart (lib "Simulation_SPICE") (part "VDC") (pins (pin (num "1") (name "+")) (pin (num "2") (name "-"))))')
    nets = ('(net (code "1") (name "GND") (node (ref "V5") (pin "2")) (node (ref "Q5") (pin "1")) (node (ref "Q5") (pin "2")) (node (ref "Q5") (pin "3")))'
            '(net (code "2") (name "/p") (node (ref "V5") (pin "1")) (node (ref "V9") (pin "1")) (node (ref "V9") (pin "2")) (node (ref "Q5") (pin "4")))'
            '(net (code "3") (name "/k") (node (ref "V9") (pin "3")) (node (ref "R1") (pin "1")))'
            '(net (code "4") (name "/d") (node (ref "R1") (pin "2")) (node (ref "Q5") (pin "5")) (node (ref "Q5") (pin "6")) (node (ref "Q5") (pin "7")) (node (ref "Q5") (pin "8")))')
    c2 = KI.import_kicad_net(None, _net(comps, libp, nets))
    els = [e.line() for e in c2.elements]
    chk("3 共阴双二极管拆成 2 只、SO-8 MOSFET 多个 D/S 脚合并成 3 端", sum(l.startswith("DV9_") for l in els) == 2 and any(l.startswith("MQ5 /d /p 0") for l in els),
        "; ".join(els))

    # ---- 4) 热模型：烧毁截止 + 拼接处连续
    t = np.linspace(0, 16e-3, 160001); p = np.where((t * 20e3) % 1 < 0.5, 1.4, 0.0)  # 20 kHz 半波，均值 0.7 W，末点落在关断半周
    th = B.Thermal(ta=40.0)
    tt, tj, pe, ss = B.thermal_response(t, p, th)
    n = np.searchsorted(tt, t[-1], side="right")
    dip_new = tj[n - 1] - tj[(tt >= t[-1]) & (tt <= 1.6 * t[-1])].min()
    # 旧算法（v0.11）：窗口末点直接连到 1.5·t_end，那段用 (p末点 + p保持)/2
    td, pd_ = B.downsample(t, p)
    to = np.concatenate([td, np.geomspace(td[-1] * 1.5, 300, 150)]); po = np.concatenate([pd_, np.full(150, pe)])
    tjo = B.Thermal(ta=40.0).tj(to, po, burn=False)
    no = len(td); dip_old = tjo[no - 1] - tjo[no]
    chk("4 拼接处 Tj 连续（无凹口）", dip_new < 0.01, "凹口 旧 %.2f K → 新 %.4f K；稳态 %.2f °C（理论 %.2f）" % (dip_old, max(dip_new, 0), ss, 40 + 0.7 * 100))
    an = lambda x: 40 + 0.7 * (15 * (1 - np.exp(-x / 2e-3)) + 85 * (1 - np.exp(-x / 20)))
    th2 = B.Thermal(ta=40.0); tt2, tj2, _, _ = B.thermal_response(np.linspace(0, 16e-3, 1001), np.full(1001, 0.7), th2)
    chk("4 恒定功率与解析解一致（含保持段）", np.max(np.abs(tj2 - an(tt2))) < 1e-6, "最大误差 %.1e K" % np.max(np.abs(tj2 - an(tt2))))
    th3 = B.Thermal(ta=40.0); t3 = np.linspace(0, 3e-3, 3001); p3 = np.where(t3 < 1e-3, 400.0, 0.0)
    tt3, tj3, _, ss3 = B.thermal_response(t3, p3, th3, pend=0.0)
    v3_ = B.judge(th3, float(tj3.max()), ss3, 1.0)
    chk("4 烧毁截止：Tj 不超过烧毁温度、标记烧毁、之后按开路", th3.burned and tj3.max() <= th3.burn_c + 1e-9 and v3_[0] == "烧毁" and "开路" in v3_[1],
        "烧毁于 %.3f ms，不截止的旧模型会算到 %.0f °C；判定：%s" % (th3.t_burn * 1e3, B.Thermal(ta=40.0).tj(tt3, np.interp(tt3, t3, p3), burn=False).max(), v3_[1]))
    th4 = B.Thermal(ta=40.0, fail_mode="short", short_frac=0.05); B.thermal_response(t3, p3, th4, pend=0.0)
    chk("4 短路假设也明确标注", th4.burned and "短路" in th4.fail_note)

    # ---- 5) 无激励
    c5 = KI.import_spice(None, "passive only\nR1 a 0 1k\nC1 a b 1u\nR2 b 0 1k\n.tran 1u 1m\n.end\n")
    w5 = c5.run(".tran 1u 1m")
    chk("5 无源网表：导入和仿真都报“无激励源，结果全零”", any("无激励源" in x for x in c5.warnings) and any("无激励源" in x for x in w5.warnings), w5.warnings[0] if w5.warnings else "")
    c6 = KI.import_spice(None, "rc\nV1 a 0 PULSE(0 1 0 1n 1n 0.5m 1m)\nR1 a b 1k\nC1 b 0 1u\n.end\n")
    chk("5 有源网表不误报", not c6.warnings and not c6.run(".tran 1u 1m").warnings)
    c7 = KI.import_spice(None, "ic only\nR1 a 0 1k\nC1 a 0 1u IC=5\n.end\n")
    chk("5 只有初始条件（IC=）不算无激励", not c7.warnings)

    # ---- 6) Waveforms 大小写
    x = np.linspace(0, 1e-3, 101)
    wv = Waveforms("t", {"time": x, "V(OUT)": np.sin(x * 1e4), "Vin": x, "V1#branch": -x}, {"time": "time"})
    same = all(np.array_equal(wv[k], np.sin(x * 1e4)) for k in ("out", "OUT", "v(out)", "V(Out)"))
    chk("6 存储/查找大小写不敏感，显示保留原名", same and np.array_equal(wv["vin"], x) and np.array_equal(wv["I(v1)"], -x) and "OUT" in wv.nets() and "Vin" in wv.nets(),
        "nets=%s" % wv.nets())
    sc = Scope(); sc.load(wv); sc.ch[0].source = "Out"; sc.ch[0].on = True
    chk("6 示波器用混合大小写取外部波形", sc.raw(1) is not None and len(sc.raw(1)) == 101)
    print("  v0.11.1 电路导入回归：%d/%d 通过" % (sum(ok_all), len(ok_all)))


if __name__ == "__main__":
    run()
