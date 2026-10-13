"""Geometry contract regression: core and actual HTTP input path."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rw.induction import BottleNeckGeometry, build_mesh

def rejected(**kwargs):
    try:
        BottleNeckGeometry(**kwargs)
    except ValueError:
        return
    raise AssertionError(f'illegal geometry accepted: {kwargs}')

for kwargs in [dict(neck_od=10, wall=8), dict(coil_id=10),
               dict(wall=0), dict(neck_len=float('nan')), dict(coil_h=-1),
               dict(shoulder_len=0), dict(body_od=4), dict(zone=100,neck_len=10),
               dict(coil_z0=-20,coil_id=24)]:
    rejected(**kwargs)
g=BottleNeckGeometry()
m=build_mesh(g)
assert m.p[0].min() == 0 and g.ri == 8
# Existing zero-inner-radius and zero-gap cases stay available, not certified.
BottleNeckGeometry(neck_od=20,wall=10)
BottleNeckGeometry(coil_id=22)
print('Core geometry invalid relations rejected; defaults and degenerate boundaries preserved')
os.environ['RW_JOBS_DB']='/tmp/rw-geometry-http-test.db'
from fastapi.testclient import TestClient
from web.server import app
with TestClient(app) as client:
    for kwargs in [dict(neck_od=10,wall=8),dict(coil_id=10),dict(neck_len=10,zone=100)]:
        response=client.post('/api/sim/induction',json={'params':kwargs})
        assert response.status_code==400, (kwargs,response.status_code,response.text)
print('Actual HTTP rejects illegal geometry before creating jobs')
from web.modules import run_induction
for kwargs in [dict(neck_od=10,wall=8),dict(coil_id=10)]:
    try:
        run_induction(kwargs,lambda _:None)
    except ValueError:
        pass
    else:
        raise AssertionError('direct runner accepted illegal geometry')
from rw.induction import InductionQuenchSim
result=InductionQuenchSim(t_heat=.1,t_quench=.1,t_air=.1,power_kw=6).run()
assert all(__import__('math').isfinite(result[k]) for k in ('T_max','T_surf_end_heat','T_inner_end_heat'))
print('Direct runner rejects invalid geometry; default full solver short run finite')
