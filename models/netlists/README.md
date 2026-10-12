# 公开示例网表

- `npn_switch_kicad.net`：按 KiCad 8 网表格式（Export Netlist → KiCad，S 表达式）**手写**的示例：
  1 kHz 方波 → 10 kΩ → NPN 开关管，集电极 1 kΩ 上拉到 12 V，100 nF 到地。
  不是从真实 KiCad 工程导出的（沙盒里没有 KiCad），字段名按 KiCad 8 的 Sim.Device / Sim.Type / Sim.Params / pinfunction 写。
- `npn_switch.cir`：同一电路的 SPICE 网表（KiCad「导出 SPICE 网表」的样子），带 .tran。

两份网表 MIT 许可，可随意使用。
- `iec_designators_kicad.net`（v0.11.1 回归用，手写）：IEC/DIN 位号（V=半导体、D=集成电路、X=端子、G=电源）的小电路，
  检查 .net 导入按 Sim.Device / 库符号 / 型号 / 引脚名判断器件类型，位号只作最后兜底；含 3 脚 PNP、3 脚 NMOS、电位器、自恢复保险丝、
  没有模型的隔离芯片和一个判断不出的器件（应给警告）。
- `options_test.cir`（v0.11.1 回归用）：含 `.options rshunt=…` 与 `.option`，导入后结果必须和直接 ngspice 一致。
- `no_source_demo.cir`（v0.11.1 演示）：没有任何激励源的无源网表，⑥ 页应显示“无激励源，结果全零”警告。
