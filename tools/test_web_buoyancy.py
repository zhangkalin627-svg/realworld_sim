"""Real Web world3d output: static buoyancy conclusions, not fluid dynamics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web.w3d import run_world3d
r = run_world3d({'preset': '水里的浮沉', 'tmax': .1}, lambda _: None)
for name, result in [('松木','浮'),('冰','浮'),('PP','浮'),('铝','沉'),('钢','沉'),('铜','沉')]:
    assert any(f'{name}：{result}' in x for x in r['lines']), (name, r['lines'])
assert any('静态' in x and '非液体动力学' in x for x in r['lines'])
assert len(r['raw']['buoyancy']) == 6
for x in r['raw']['buoyancy']:
    assert 0 <= x['draft'] <= 1 and isinstance(x['floats'], bool)
r2=run_world3d({'preset': '钢板桥：两墩 + 铜块','tmax': .1}, lambda _:None)
assert not r2['raw'].get('buoyancy')
print('Web buoyancy/static-only and dry-scene regressions passed')
