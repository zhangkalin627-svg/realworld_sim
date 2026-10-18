# 对标研究 v1：MATLAB/Simulink + PLECS 设计借鉴 → realworld_sim 差距清单与改进项（2026-10-18）

任务来源：用户 11:23/11:24（家机已装 MATLAB R2026b 试用、PLECS 等许可；研究两者设计用来持续更新 realworld_sim；目标=功能做到最好超过它们、界面比它们好看；只借鉴思路与公开文档/行为，自己重新实现，不照抄代码/界面素材/商标）。
本版基于公开文档（docs.plexim.com PLECS 5.0 手册、mathworks.com Simulink/Simscape 求解器文档）+ realworld_sim 仓库现状盘点（commit 05ea17b）。家机实操核对（界面细节/面板截图）另列 §6，待家机桥协助。

## 1. 对方设计要点（公开出处）
**PLECS**（docs.plexim.com/plecs/latest/how-plecs-works、scripting）：
- 三范式同环境：C 方程 + 框图（信号流）+ 物理模型（电气/磁/机械/热四域，集总参数）。
- 核心求解架构：物理模型整体作为一个原子子系统，理想开关+线性元件 → **分段线性状态空间矩阵组（Aσ/Bσ/Cσ/Dσ），开关状态每变一次换一组、带缓存**；Switch Manager 用零交叉信号让 ODE 求解器**精确定位开关时刻**；自然换向器件迭代至稳态。这是电力电子仿真又快又稳的关键设计。
- 代数环 Newton 求解；direct feedthrough 概念管理执行顺序。
- 脚本化：Standalone 内嵌 GNU Octave + **XML-RPC/JSON-RPC 外部接口**（Python 可直接驱动）；Blockset 靠 MATLAB 脚本。
- 参数扫描、Probe/Scope 波形、热描述（损耗→热网络）。

**MATLAB/Simulink**（mathworks.com/help/simulink/ug/choose-a-solver.html）：
- 求解器家族+**autosolver 自动选择**：状态数 <1000 时实算刚度，刚性→ode15s、非刚→ode45；DAE→ode23t/ode14x；Simscape Electrical 固定 ode23tb（电力电子含非线性开关必须刚性求解器）；定步长/变步长/离散全谱系。
- 框图（信号流）+ Simscape（acausal 物理网络，方程自动组装）双范式；库浏览器+mask 子系统封装；parsim/Simulation Manager 批量仿真；Simulation Data Inspector **多 run 叠加对比**；App Designer 做自定义界面。

## 2. realworld_sim 现状（对照基线）
- 求解：自研 FEM（induction.py EM-热耦合，含居里塌缩）+ 电路/PLL/MCU/PLC 模块；固定步长为主，无自动刚度识别、无开关事件精确定位。
- 组织：rw/ 按物理域分包（circuit/em/flow/machine/mcu/plc/world/matdb），gui_*.py 每域一页（手写 Qt），app.py 主程序+Worker 线程，scan.py 参数扫描（ScanJob），viz.py/theme.py 可视化与深色主题，world3d 3D 演示。
- 已有优势（MATLAB/PLECS 没有或弱的）：①EM-热-金相 FEM 直算（感应淬火场景比 PLECS 集总磁路深一个量级）；②validate_*.py 可反驳断言文化+【确定/待证实】标注纪律；③现场问题导向（粉尘/拉弧、扫描淬火、电感估算）；④深色高档 UI 已有主题雏形（Hark 规范）。

## 3. 差距清单 → 改进项（功能维度）
| # | 维度 | 对方做法 | 差距 | 改进项（优先级） |
|---|---|---|---|---|
| F1 | 电力电子求解核心 | PLECS 分段线性状态空间+开关事件精确定位+矩阵缓存 | 我电路模块无事件定位，换向时刻误差=步长级 | **P0**：circuit 模块引入"分段线性拓扑+事件定位"（逆变器换向/二极管自然换向时刻精确到步长内），HFQ6 波形精度直接受益；矩阵缓存加速 |
| F2 | 求解器选择 | MATLAB autosolver 刚度检测自动选 ode | 无 | P1：求解器配置页加"自动/刚性/非刚"档（默认自动，按雅可比特征值粗判），用户可见当前选用与原因 |
| F3 | 建模范式 | 双/三范式（框图+物理+方程） | 单范式（代码模型页） | P2（长期）：元器件级拖拽连线画布（接线即拓扑，正是"元器件级理解"主线的终极形态）；P1（近期）：器件卡片库（每器件一页：作用/正常波形/失效波形，改参数波形即变——ROADMAP 主线直接落地） |
| F4 | 热-电联合 | PLECS 热域（损耗→热网络→温升回读） | induction.py 有 EM-热，但电路器件（IGBT/电容/母排）无损耗→温升链 | P1：器件损耗查表（ datasheet 曲线录入）→3-5 节点热网络→温升读数，接故障库（过热失效模式） |
| F5 | 脚本/自动化 | PLECS Octave+RPC；MATLAB parsim | validate 已脚本化；GUI 会话不可脚本驱动、无 RPC | P1：headless CLI 模式（`python -m rw.run --case xxx.yaml --out dir`）统一 validate 与批量扫描；P2：JSON-RPC 会话接口（家机桥/外部驱动 GUI 同款模型） |
| F6 | 参数扫描 | PLECS/MATLAB 结构化扫描+结果叠加 | scan.py 已有基础 | P1：扫描结果结构化存档（JSON+曲线族）+GUI 扫描配置页完善 |
| F7 | 多 run 对比 | Simulink SDI 多 run 叠加+差异 | 无（replay 单 run） | **P0**：波形页"对比模式"：存基准 run，改参数再跑，双 run 叠加+差异区高亮+关键指标差值表（学习/维修场景杀手锏，对方最弱我们也缺的点） |
| F8 | 组件参数封装 | mask 子系统（参数对话框自动生成） | gui_*.py 手写面板重复 | P1：组件参数 schema 化（YAML 声明参数/单位/范围/说明）→GUI 面板自动生成，新器件接入成本大降 |
| F9 | 结果可信度展示 | 无（对方靠品牌信任） | validate 有但未可视化 | P1（差异化优势）：每个模型页挂 validate 状态徽章（最近通过时间+锚点数+覆盖项），读数旁挂"仿真结果/估算/待查证"三口径角标（已有纪律，做成 UI 元素） |
| F10 | 模型库内容 | 数千模块+行业库 | 我们聚焦感应加热垂直 | 不做广度做深度：感应加热垂直件库（逆变器/谐振槽/匹配变压器/感应器-工件/喷淋/导磁体）做到比对方默认库更贴近现场 |

## 4. 差距清单 → 改进项（界面维度，目标=比 MATLAB R2026b 与 PLECS 更好看）
- U1（P0，出设计稿给用户看）：统一设计 token——theme.py 升级为完整 design system（色板/字号阶梯/间距网格/圆角/阴影/图表配色），深色系高档路线（Hark 规范延伸）；出 2–3 张关键页设计稿截图（主页/波形页/器件卡片页）报用户审。
- U2（P1）：信息层级重构——参数面板分组卡片化、当前仿真状态横幅、危险/越限参数红色描边；波形区现代网格+坐标轴精简+悬停读数。
- U3（P1）：波形交互——光标测量（峰值/周期/相位差读数）、缩放回退、多 run 叠加配色系（与 F7 同步）。
- U4（P2）：动效——参数调节→波形平滑过渡（插值动画，Hark 演示同款物理直觉）；页面切换淡入。
- U5（P2）：演示模式（presentation mode）——大字号只读视图，给用户对外讲时投屏用（呼应成果展示线）。

## 5. 迭代计划并入口径
- ROADMAP 当前优先队列（参数驱动过程动画、冶金结果直算）不动；本清单 P0（F1/F7/U1）插入队列其后，P1 随主线滚动，P2 入长期愿景。
- 每项落地仍按研发纪律：validate 锚点先行、【确定/待证实】标注、全部"仿真结果"口径。
- 借鉴红线：只学设计思路与公开文档/行为，代码全部自研；不复制对方 UI 素材/图标/商标；对比描述用"对标其公开文档所述能力"。

## 6. 待家机桥协助（MATLAB R2026b 实操核对，清单已报 parent）
1. 已装产品家族清单（Simulink/Simscape/Simscape Electrical 是否在列）；
2. Simulink 库浏览器结构 + Solver 配置面板全选项（截图）；
3. Simscape Electrical（Specialized Power Systems）子库树 + 一个电力电子 demo 的模型结构/参数对话框（截图）；
4. Scope 与 Simulation Data Inspector 交互（光标/测量/多 run 对比操作路径+截图）；
5. R2026b 新界面特征（深色主题、面板布局）截图，供 U1 设计稿参考；
6. PLECS 许可到位后同套（库结构/求解器选项/热域/扫描配置）。

---

## 增补 v1.1（2026-10-18 11:47）：MATLAB R2026b 实机核对 · 第一批（家机桥截图 5 张）

已装产品（家机实测）：MATLAB / Simulink / Simscape / Simscape Electrical，版本 26.2（=R2026b）。库树中出现的其他产品名（HDL Coder、Stateflow 等）是浏览器条目，不等于已装。
截图存档：私有备份盘 matlab-plecs-对标实机核对_2026-10-18/（5 张；不进公开仓库，对方界面素材只核对不复制）。

**Solver 配置面板（变步长页，实机）**：
- 类型：变步长/定步长；求解器默认"自动(自动求解器选择)"。
- 变步长求解器全列表：自动、离散(无连续状态)、ode45 (Dormand-Prince)、ode23 (Bogacki-Shampine)、ode113 (Adams)、ode15s (stiff/NDF)、ode23s (stiff/Mod. Rosenbrock)、ode23t (mod. stiff/Trapezoidal)、ode23tb (stiff/TR-BDF2)、odeN (Nonadaptive)、daessc (DAE solver for Simscape)。
- 详细参数：最大/最小/初始步长(auto)、相对容差 1e-3、绝对容差(auto)、自动缩放绝对容差、形状保持(全部禁用)、连续最小步数 1。
- 过零选项：过零控制(使用局部设置)、算法(非自适应)、时间容差 10*128*eps、信号阈值(auto)、连续过零点数 1000。→ 对应本清单 F1（开关事件定位）的配置面设计参考：事件定位精度/算法/防抖（连续过零点数）三参数是成熟做法。
- 高级参数：仿真时间 0 处的日期时间、启用解耦的连续积分、尽量降低过零点对积分的影响。

**库结构（实机）**：
- Simulink 库：Commonly Used Blocks、Continuous、Dashboard、DateTime、Discontinuities、Discrete、Logic and Bit Operations、Lookup Tables、Math Operations、Matrix Operations、Messages & Events、Model Verification、Model-Wide Utilities、Ports & Subsystems、Signal Attributes、Signal Routing、Sinks、Sources、String、User-Defined Functions；另有 Additional Math & Discrete、Fixed-Point Designer(+HDL Support)、HDL Coder、Simscape、Simulink 3D Animation、Simulink Coder、Simulink Extras、Stateflow、最近使用。
- Simscape Electrical 树：Foundation Library、Utilities、Electrical（Connectors & References、Control、DQ0 Components、Electromechanical、Integrated Circuits、Passive、Semiconductors & Converters〔Converters、Sensors & Transducers、Sources、Switches & Breakers、Utilities〕、Additional Components、Network Couplers）。注意：本视图未见经典 Specialized Power Systems(SPS) 子库入口，26.2 可能已并入或需另开；待下一批核实。
- 对我方 F3/F8 的启示：库按"功能域→子域→器件"三级组织，器件对话框是主要参数入口；realworld_sim 器件卡片库（F3）+ 参数 schema 自动生成面板（F8）路线一致，可对标其分组命名。

**对差距清单的修订**：F2（求解器配置页）补充设计目标——提供"自动/刚性/非刚"三档+事件定位三参数（精度/算法/防抖），默认值对标：相对容差 1e-3 量级、连续事件防抖计数。

待下一批核对：定步长求解器列表、Scope 与 Simulation Data Inspector 交互、电力电子 demo 模型结构、SPS 是否在 26.2 保留、PLECS（许可到位后）。

---

## 增补 v1.2（2026-10-18 11:50）：MATLAB R2026b 实机核对 · 第二批 + 产品清单 + SPS 结论修正

**产品清单（ver 实测）**：MATLAB 26.2.0.3386108 (R2026b) + Simulink/Simscape/Simscape Electrical 全 26.2，许可证 DEMO，Windows 11 专业版 26300。桌面为深色主题（R2026b 原生深色，工具带+面板dock 布局）→ U1 方向（深色高档）与行业最新版一致。家机下载目录另有 plecs-standalone-5-0-6 安装包（PLECS 5.0.6，等许可）。

**SPS 结论修正（重要）**：Specialized Power Systems library 在 R2026a 已**移除**（官方正文，家机桥核）：https://www.mathworks.com/help/simscape-electrical/ug/upgrade-sps-models-to-use-simscape-blocks.html ，迁移走 spsConversionAssistant/手工替换。本清单 v1 与增补 v1.1 中"Simscape Electrical(Specialized Power Systems)"表述作废——26.2 只有 Simscape Electrical 新库；电力电子建模全面转向 Simscape 物理网络（半导体+变换器子库）。这对我们反而是好消息：对方砍掉了旧体系，新体系学习成本上升，realworld_sim 的垂直专用路线少了历史包袱。

**定步长求解器全列表（实机）**：自动、离散(无连续状态)、ode8 (Dormand-Prince)、ode5、ode4 (Runge-Kutta)、ode3 (Bogacki-Shampine)、ode2 (Heun)、ode1 (Euler)、ode14x (外插)、ode1be (Backward Euler)。

**Scope 双光标（实机）**：测量页签，光标 1/2 竖线+沿线圆点，读数框 ΔT/ΔY（如 ΔT: 5, ΔY: 337.8316e-03），底部状态栏显示采样基准。→ 我们波形页光标读数（Δt/1/Δt）方向一致，差 ΔY 读数可补。

**Simulation Data Inspector 两 run 比较（实机，F7 的直接对标）**：
- 结构：基线 run 下拉 vs 比较项 run 下拉；结果列（匹配/不匹配 + 最大差值）；上图两 run 叠加（基线蓝/比较橙），下图**差值曲线+容差带**（红=差，绿=容差）。
- 容差字段：绝对容差/相对容差/时间容差，按信号分别设置；匹配统计（N 匹配/N 不匹配/N 未比较）；属性面板含单位、数据类型、采样时间、对齐方式、模型/模块/端口路径。
- → F7 升级项：对比页加①差值子图+容差带（默认相对容差 1e-3 与求解器口径一致）②"匹配/不匹配"判定徽章③按信号设容差。这是 SDI 的核心交互，感应加热场景我们可做得更垂直（容差模板：槽流 ±2%、f0 ±0.5%）。

**BuckConverter_sd（自带转换组件库，非完整 demo）**：
- mask 参数对话框：Forward voltage 0.8 / On-state resistance 0.001 / Off-state conductance 1e-5（开关与二极管两组）/ Snubber 枚举(ee.enum.converters.snubber.exclude)。→ 理想开关四参数（Vf/Ron/Goff/缓冲）与 PLECS 同思路，证实 F1 分段线性理想开关的参数面设计：Vf、Ron、Goff、Snubber 枚举。
- 内部结构：行为级——门信号 g × (BL≤0.5 判定) → 理想开关（G 门控，1+/1− 电气端口），输出 2+/2−。行为级 buck 用乘法+理想开关实现，非器件级物理。

待核对（末批）：正式电力电子 demo（家机桥在找）、PLECS 5.0.6 许可到位后同套（库结构/求解器/热域/扫描）。
