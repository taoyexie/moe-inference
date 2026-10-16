"""Dispatch state checks of the actual engine method; fake kernels, not CUDA."""
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kite_e2e_core import FramePool
from kite_e2e_torch import QwenEngine


class DispatchTests(unittest.TestCase):
    def engine(self):
        engine = QwenEngine.__new__(QwenEngine)
        engine.pool = FramePool(21*1024**2)
        engine.pool.set_limits(1,1)
        engine.arena = SimpleNamespace(weights=lambda entry:entry['frames'])
        engine.events,engine.counts = [],Counter()
        engine.in_replay,engine.force_path = False,None
        engine.want_capture,engine.capture = False,None
        engine.validate_cpu_outputs = False
        engine.scratch_weights = ('scratch',)
        engine.costs = SimpleNamespace(expert=lambda q:(1.,.1*q))
        loaded = {}
        engine.upload = lambda key,weights:loaded.update({tuple(weights):key})
        engine.gpu_expert = lambda weights,x:('gpu',loaded[tuple(weights)])
        engine.cpu_expert = lambda key,x:('cpu',key)
        return engine

    def test_bypass_hit_and_admit_execute_on_shared_pool(self):
        engine = self.engine()
        x = lambda q:SimpleNamespace(shape=(q,2048))
        self.assertEqual(engine.expert((0,1),x(1)),('cpu',(0,1)))
        self.assertFalse(engine.pool.entries['expert'])
        self.assertEqual(engine.expert((0,2),x(20)),('gpu',(0,2)))
        self.assertEqual(engine.expert((0,3),x(1)),('cpu',(0,3)))
        self.assertEqual(engine.expert((0,2),x(1)),('gpu',(0,2)))
        self.assertEqual(engine.counts['expert_calls'],4)
        self.assertEqual(engine.counts['expert_hits'],1)
        self.assertEqual(engine.counts['expert_cpu'],2)
        self.assertEqual(engine.counts['expert_upload'],1)
        self.assertEqual(len(engine.events),4)

    def test_replay_children_not_reintroduced_into_normal_histogram(self):
        engine = self.engine()
        engine.in_replay = True
        engine.expert((1,2),SimpleNamespace(shape=(1,2048)))
        self.assertFalse(engine.events)
        self.assertEqual(engine.counts['replay_expert_calls'],1)
        self.assertEqual(engine.counts['replay_expert_cpu'],1)

    def test_partial_upload_failure_drops_invalid_admission(self):
        engine = self.engine()
        def fail(*args):
            raise RuntimeError('copy failure')
        engine.upload = fail
        with self.assertRaises(RuntimeError):
            engine.expert((1,2),SimpleNamespace(shape=(32,2048)))
        self.assertFalse(engine.pool.entries['expert'])
        self.assertEqual(len(engine.pool.free),engine.pool.frame_count)

    def test_cache_aware_cost_checks_real_fp32_presence_before_choosing_cpu(self):
        engine = self.engine()
        engine.cpu_cache = {(0,7):'actual FP32 copy'}
        observed = []
        def expert_for_cache(q,cpu_cached):
            observed.append((q,cpu_cached))
            return 1.,(.2 if cpu_cached else 2.)
        engine.costs = SimpleNamespace(expert_for_cache=expert_for_cache)
        x = SimpleNamespace(shape=(1,2048))
        self.assertEqual(engine.expert((0,1),x),('gpu',(0,1)))
        self.assertEqual(engine.expert((0,7),x),('cpu',(0,7)))
        self.assertEqual(observed,[(1,False),(1,True)])
        self.assertNotIn((0,7),engine.pool.entries['expert'])


if __name__=='__main__':
    unittest.main()
