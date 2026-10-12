# 材料数据来源
- elements.json 由 build_elements.py 从以下开源数据生成：
  - pymatgen `periodic_table.json`（Materials Project，MIT License，© The Materials Project）
  - mendeleev `elements.db` v1.3（L. Mentel，MIT License）
- alloys.py：国标/ASTM 标准最小值与公开手册典型值（ASM、MatWeb 等常见数值），以材质证明书为准。
- mixing.py：已知二元相图结论为冶金教科书常识；Miedema 模型参数取自 mendeleev。
