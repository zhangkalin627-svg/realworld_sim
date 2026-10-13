"""API boundary rejects non-finite numbers rather than silently clamping."""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['RW_JOBS_DB']='/tmp/rw-finite-test.db'
from web.server import _validate_params
from fastapi import HTTPException
mod={'fields':[{'name':'x','type':'number','min':0,'max':10}]}
for v in ('nan','inf','-inf',float('nan'),float('inf'),None):
    try: _validate_params(mod,{'x':v})
    except HTTPException as e: assert e.status_code==400
    else: raise AssertionError(f'Non-finite accepted: {v}')
assert _validate_params(mod,{'x':-2})['x']==0
assert _validate_params(mod,{'x':12})['x']==10
assert _validate_params(mod,{'x':'4.5'})['x']==4.5
print('Non-finite rejection, existing finite range clamp passed')
