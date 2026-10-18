"""N7 能量守恒自检的回归：线性 RLC 残差≈0；含二极管残差≥0；故意制造"产能"网络应黄牌。仿真结果，非实测。"""
import sys
from rw.circuit.spice import Circuit, find_ngspice
from rw.circuit.energy import energy_check

def main():
    if not find_ngspice():
        print("没找到 ngspice，跳过"); return
    ok = True
    def chk(name, cond, info=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name, info); ok = ok and cond
    # 1 线性 RLC 串联，电压阶跃
    c = Circuit("rlc"); c.V("1", "a", "0", pulse=(0, 5, 0, 1e-9, 1e-9, 5e-3, 1e-2)); c.R("1", "a", "b", 10); c.L("1", "b", "c", 1e-3); c.C("1", "c", "0", 10e-6)
    w = c.run(".tran 1u 3m 0 1u")
    r = energy_check(c, w); chk("1 线性RLC 残差≈0", r["status"] == "通过", r["notes"][-1])
    # 2 半波整流（含二极管）
    c = Circuit("rect"); c.model("D1N4007"); c.V("1", "a", "0", sin=(0, 10, 1000)); c.D("1", "a", "b", "D1N4007"); c.R("1", "b", "0", 100); c.C("1", "b", "0", 10e-6)
    w = c.run(".tran 1u 5m 0 1u")
    r = energy_check(c, w); chk("2 整流 残差≥0（二极管耗散）", r["status"] == "通过" and r["residual"] > 0, r["notes"][-1])
    # 3 谐振 RLC 带互感耦合（变压器）
    c = Circuit("xfmr"); c.V("1", "a", "0", sin=(0, 5, 5000)); c.R("1", "a", "b", 1); c.L("1", "b", "0", 5e-3); c.L("2", "d", "0", 20e-6); c.K("1", "L1", "L2", 0.99); c.R("2", "d", "0", 0.5)
    w = c.run(".tran 0.5u 4m 0 0.5u")
    r = energy_check(c, w); chk("3 含互感线性网络 残差≈0", r["status"] == "通过", r["notes"][-1])
    # 4 行为源产能：受控电源把电流打回源，残差应为负 → 黄牌
    c = Circuit("gen"); c.V("1", "a", "0", dc=1.0); c.R("1", "a", "b", 1); c.B("1", "b", "0", "V=-2")
    w = c.run(".tran 1u 1m 0 1u")
    r = energy_check(c, w); chk("4 产能网络 黄牌", r["status"] == "黄牌", r["notes"][-1])
    print("ALL PASS" if ok else "SOME FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
