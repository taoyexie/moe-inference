import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_kite_e2e import same
from kite_controller import CostCalibration
from kite_e2e import new_requests
from kite_e2e_core import packed_layout
from kite_e2e_runtime import Runtime,execute_requests


class DriverTests(unittest.TestCase):
    def test_real_driver_lifecycle_and_startup_ttft_boundary(self):
        specs = [dict(request_id='background',input_ids=[1]*2048,release_step=0,output_tokens=32)]
        specs += [dict(request_id=f'{s}-{i}',input_ids=[2]*512,release_step=s,output_tokens=8)
                  for s in (9,17,25) for i in range(3)]
        requests = new_requests(specs)
        expired,roles,masked = [],[],[]
        class Logits:
            def __init__(self,n):
                self.n = n
            def __setitem__(self,index,value):
                masked.append(index)
            def argmax(self,dim):
                return SimpleNamespace(tolist=lambda:[3]*self.n)
        def forward(batch):
            layout = packed_layout(batch)
            p = sum(row['phase']=='prefill' for row in layout)
            roles.append((p,len(batch)-p))
            for req,row in zip(batch,layout):
                req.kv_length = row['after']
            return Logits(len(batch))
        engine = SimpleNamespace(synchronize=lambda:None,
            model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9])))
        runtime = SimpleNamespace(engine=engine,forward=forward,expire=lambda req:expired.append(req.request_id))
        start = time.perf_counter_ns()-1_000_000
        self.assertEqual(execute_requests(runtime,requests,start),33)
        self.assertEqual(len(expired),10)
        self.assertEqual(sum(len(r.generated) for r in requests),104)
        self.assertEqual(roles[9],(3,1))
        self.assertEqual(len(masked),33)
        self.assertEqual(requests[0].submitted_ns,start)
        self.assertGreaterEqual(requests[0].token_ticks[0]-start,1_000_000)
        self.assertTrue(all(r.done and r.kv_length==len(r.prompt_ids)+r.output_tokens-1 for r in requests))

    def test_calibration_warmup_is_retained_not_fitted(self):
        runtime = Runtime.__new__(Runtime)
        runtime.engine = SimpleNamespace(synchronize=lambda:None)
        runtime.controller = SimpleNamespace(calibration=CostCalibration())
        runtime.calibration_rows = []
        self.assertEqual(runtime.timed_sample('expert_cpu',1,lambda:'result',True),'result')
        self.assertFalse(runtime.controller.calibration.samples['expert_cpu'])
        runtime.timed_sample('expert_cpu',1,lambda:'result',False)
        self.assertEqual(len(runtime.controller.calibration.samples['expert_cpu']),1)
        self.assertEqual(len(runtime.calibration_rows),2)

    def test_audit_float_array_roundoff_not_false_alarm(self):
        same([1.,2.],[1.+1e-12,2.])
        with self.assertRaises(ValueError):
            same([1.,2.],[1.,3.])


if __name__=='__main__':
    unittest.main()
