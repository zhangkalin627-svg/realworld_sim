"""Isolated API/store lifecycle guards; physics is not under test."""
import os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
_tmp=tempfile.TemporaryDirectory(prefix='rw-lifecycle-')
os.environ['RW_JOBS_DB']=str(Path(_tmp.name)/'import.db')
from fastapi.testclient import TestClient
from web import server
from web.jobstore import JobStore

class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='rw-life-case-')
        self.store=JobStore(str(Path(self.tmp.name)/'jobs.db'))
        self.client=TestClient(server.app,raise_server_exceptions=False)
        self.mod={'key':'test-life','name':'test','fields':[],'ckpt':True,'run':lambda *a,**k:{'lines':[]}}
        self.patches=[patch.object(server,'STORE',self.store),patch.object(server,'MODULES',[self.mod])]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def seed(self,jid='old',module='test-life',checkpoint='{}',status='interrupted'):
        self.store.create(jid,module,{})
        if checkpoint is not None:self.store.set_checkpoint(jid,checkpoint)
        self.store.set_error(jid,'stopped')
        if status=='done':self.store.set_done(jid,{'old':True})
    def post(self,params):
        return self.client.post('/api/sim/test-life',json={'params':params})
    def test_bad_resume_does_not_silently_start_fresh(self):
        for case in ('missing','no-checkpoint','wrong-module','done','bad-json','non-object','unsupported'):
            with self.subTest(case=case):
                with self.store._conn() as c:c.execute('DELETE FROM jobs')
                self.seed(checkpoint=None if case=='no-checkpoint' else ('{' if case=='bad-json' else '[]' if case=='non-object' else '{}'),module='other' if case=='wrong-module' else 'test-life',status='done' if case=='done' else 'interrupted')
                self.mod['ckpt']=case!='unsupported'
                with patch.object(server._pool,'submit') as submit:
                    response=self.post({'resume_from':'absent' if case=='missing' else 'old'})
                    self.assertIn(response.status_code,(400,404,409),response.text)
                    submit.assert_not_called()
                with self.store._conn() as c:c.execute('DELETE FROM jobs')
    def test_valid_resume_is_passed_not_replaced(self):
        self.seed(checkpoint='{"marker":17}')
        seen=[]
        self.mod['run']=lambda params,progress,**kw:seen.append(kw['resume']) or {'lines':[]}
        with patch.object(server._pool,'submit',side_effect=lambda fn:fn()):
            response=self.post({'resume_from':'old'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(seen,[{'marker':17}])
    def test_active_jobs_survive_retention(self):
        self.store.create('active','test-life',{})
        self.store.set_checkpoint('active','{}')
        for i in range(205):
            jid=f'done{i}';self.store.create(jid,'test-life',{});self.store.set_done(jid,{})
        self.assertIsNotNone(self.store.get('active'))
        self.assertEqual(self.store.get_checkpoint('active'),'{}')
    def test_invalid_resume_type_rejected(self):
        with patch.object(server._pool,'submit') as submit:
            response=self.post({'resume_from':[]})
        self.assertEqual(response.status_code,400,response.text)
        submit.assert_not_called()

    def test_rejected_executor_is_recorded_as_error(self):
        with patch.object(server._pool,'submit',side_effect=RuntimeError('executor closed')):
            response=self.post({})
        self.assertEqual(response.status_code,503,response.text)
        jobs=self.store.recent()
        self.assertEqual(len(jobs),1)
        self.assertEqual(jobs[0]['status'],'error')

    def test_induction_bad_fingerprint_rejected_before_build(self):
        from web.modules import run_induction
        with patch('rw.induction.InductionQuenchSim',side_effect=AssertionError('built before fingerprint check')) as build:
            with self.assertRaisesRegex(ValueError,'指纹'):
                run_induction({'mode':'power'},lambda f:None,resume={'params_fingerprint':'wrong'})
            build.assert_not_called()

if __name__=='__main__':
    try:unittest.main()
    finally:server._pool.shutdown(wait=True);_tmp.cleanup()
