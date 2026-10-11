"""感应加热 → PLC 的耦合表：用 v0.1 的有限元算不同 "加热时间 × 淬火时间" 下工件的
测温点温度和硬度，存成表。虚拟调试时 PLC 实际给了多长的 Y004/Y005，就查表得到
这一件的温度（决定红外测温 X040）和最终硬度。"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(HERE, "models", "demo", "thermal_table.json")
FREQ = 30e3          # 本机规格 ~30 kHz
MEDIUM = "polymer_spray"   # PAG 喷淋
HEAT = [1.5, 2.0, 2.5, 3.0, 3.5]
QUENCH = [1.0, 2.0, 3.0]


def build():
    sys.path.insert(0, HERE)
    from rw.induction import InductionQuenchSim, find_power
    t0 = time.time()
    P = find_power(900, freq=FREQ, t_heat=3.0)   # 按 3.0 s 到 900°C 调好功率（相当于现场试烧定功率）
    out = dict(freq=FREQ, medium=MEDIUM, power_kW=P, heat=HEAT, quench=QUENCH, rows=[])
    curve = None
    for th in HEAT:
        for tq in QUENCH:
            s = InductionQuenchSim(freq=FREQ, power_kw=P, t_heat=th, medium=MEDIUM, t_quench=tq, t_air=3.0)
            r = s.run()
            h = s.hist
            if th == max(HEAT) and curve is None:
                m = h["t"] <= th
                curve = dict(t=h["t"][m][::5].tolist(), Tsurf=h["Tsurf"][m][::5].tolist())
            out["rows"].append(dict(t_heat=th, t_quench=tq, T_surf=r["T_surf_end_heat"], T_in=r["T_inner_end_heat"],
                                    T_zone_min=r["T_zone_min"], HRC=r["HRC_surface"], HRC_in=r["HRC_inner"],
                                    case=r["case_depth_mm"], T_after=float(h["Tsurf"][-1])))
            print(th, tq, round(r["T_surf_end_heat"]), round(r["HRC_surface"], 1), round(r["case_depth_mm"], 2), flush=True)
    out["heat_curve"] = curve
    json.dump(out, open(PATH, "w"), indent=1)
    print("done", time.time() - t0)


class ThermalTable:
    def __init__(self, path=PATH):
        d = json.load(open(path))
        self.d = d
        self.P = d["power_kW"]
        self.H = np.array(d["heat"]); self.Q = np.array(d["quench"])
        self.grid = {k: np.array([[self._get(th, tq)[k] for tq in self.Q] for th in self.H])
                     for k in ("T_surf", "HRC", "case", "T_zone_min")}
        c = d["heat_curve"]
        self.ct, self.cT = np.array(c["t"]), np.array(c["Tsurf"])

    def _get(self, th, tq):
        for r in self.d["rows"]:
            if abs(r["t_heat"] - th) < 1e-6 and abs(r["t_quench"] - tq) < 1e-6:
                return r

    def T_surface(self, t_heat):
        """加热进行到 t_heat 秒时测温点（淬火区中部外表面）温度。"""
        return float(np.interp(t_heat, self.ct, self.cT, left=25.0))

    def result(self, t_heat, t_quench):
        from scipy.interpolate import RegularGridInterpolator
        th = float(np.clip(t_heat, self.H[0], self.H[-1]))
        tq = float(np.clip(t_quench, self.Q[0], self.Q[-1]))
        out = {}
        for k, G in self.grid.items():
            out[k] = float(RegularGridInterpolator((self.H, self.Q), G)((th, tq)))
        if t_heat < self.H[0] * 0.999:      # 加热太短：按比例降温，基本淬不硬
            out["T_surf"] = self.T_surface(t_heat)
            out["HRC"] = min(out["HRC"], 20.0); out["case"] = 0.0
        if t_quench < 0.2:
            out["HRC"] = min(out["HRC"], 25.0); out["case"] = 0.0
        return out


if __name__ == "__main__":
    build()
