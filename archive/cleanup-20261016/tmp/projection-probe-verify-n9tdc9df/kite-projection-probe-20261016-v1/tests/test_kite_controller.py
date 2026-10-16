"""Synthetic bookkeeping/property tests, not model or GPU performance evidence."""
import random
import sys
import unittest
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kite_controller import (Controller,CostCalibration,CostModel,History,RecoveryLedger,
                             ReuseStack,candidates,miss_count)


def add_costs(calibration):
    for path,x,y in [('expert_upload',1,1),('expert_cpu',1,.2),('expert_cpu',10,1.1),
                     ('kv_restore',128,2),('kv_replay',128,10),('kv_replay',512,20)]:
        calibration.add(path,x,y)


class ControllerTests(unittest.TestCase):
    def test_distances_and_boundary(self):
        stack = ReuseStack()
        distances = [stack.access(key) for key in 'ABACAB']
        self.assertEqual(distances,[None,None,1,None,1,2])
        self.assertEqual(miss_count(Counter(distances),2),4)
        self.assertEqual(miss_count(Counter(distances),0),6)

    def test_mrc_matches_independent_lru_many_capacities(self):
        rng = random.Random(17)
        stream = [rng.randrange(23) for _ in range(1500)]
        stack = ReuseStack()
        distances = [stack.access(key) for key in stream]
        for start,end in [(0,1500),(100,300),(950,1500)]:
            for capacity in range(17):
                cache,misses = OrderedDict(),0
                for i,key in enumerate(stream[:end]):
                    absent = key not in cache
                    if start <= i and absent:
                        misses += 1
                    if capacity:
                        cache[key] = None
                        cache.move_to_end(key)
                        if len(cache)>capacity:
                            cache.popitem(last=False)
                self.assertEqual(miss_count(Counter(distances[start:end]),capacity),misses)

    def test_window_reset_does_not_reset_recency(self):
        history = History()
        history.observe([('A',1),('B',1)],[])
        history.clear_window()
        history.observe([('A',1)],[])
        self.assertEqual(history.expert_bins,Counter({(1,1):1}))

    def test_kv_macro_charged_once_for_two_missing_blocks(self):
        history = History()
        history.observe([], [{'group_id':('req',0),'context':128,'objects':['A','B']}])
        result = history.score(1,1,CostModel(1,1,0,2,3,0))
        self.assertEqual(result['kv_block_misses'],2)
        self.assertEqual(result['kv_replay_tasks'],1)
        self.assertEqual(result['recovery_ms'],3)

    def test_restore_count_remains_per_block(self):
        history = History()
        history.observe([], [{'group_id':('req',0),'context':128,'objects':['A','B']}])
        result = history.score(1,1,CostModel(1,1,0,2,100,0))
        self.assertEqual(result['kv_restore_events'],2)
        self.assertEqual(result['recovery_ms'],4)

    def test_real_q_cost_correlation_retained(self):
        history = History()
        history.observe([('A',1),('B',10),('A',2)],[])
        result = history.score(2,1,CostModel(1,0,.1,1,100,0))
        self.assertEqual(result['expert_misses'],2)
        self.assertAlmostEqual(result['recovery_ms'],1.1)

    def test_calibration_once_per_workload(self):
        calibration = CostCalibration()
        with self.assertRaises(ValueError):
            calibration.freeze()
        add_costs(calibration)
        model = calibration.freeze()
        self.assertAlmostEqual(model.expert_cpu_slope_ms,.1)
        with self.assertRaises(RuntimeError):
            calibration.add('expert_cpu',2,1)
        with self.assertRaises(RuntimeError):
            calibration.freeze()

    def test_nonpositive_prediction_not_silently_used(self):
        model = CostModel(1,1,0,1,10,-1)
        with self.assertRaises(ValueError):
            model.kv(20)

    def test_candidate_bytes_feasible(self):
        for candidate in candidates(100,9,12,(.1,.25,.5,.75,.9)):
            self.assertLessEqual(candidate.expert_bytes+candidate.kv_bytes,100)
            self.assertEqual(candidate.expert_bytes%9,0)
            self.assertEqual(candidate.kv_bytes%12,0)

    def controller(self):
        controller = Controller(100,9,12,refresh_batches=3,current_weight=1)
        controller.before_batch(1,0)
        controller.after_batch([('A',1)],[])
        add_costs(controller.calibration)
        controller.finish_calibration()
        return controller

    def test_refresh_period_and_no_future_observation(self):
        controller = self.controller()
        for i in range(3):
            decision = controller.before_batch(0,1)
            self.assertEqual(decision['refresh_count'],1)
            controller.after_batch([('new-'+str(i),1)],[])
        self.assertEqual(controller.refresh_count,2)

    def test_batch_signal_moves_to_nearest_feasible_candidate(self):
        controller = self.controller()
        prefill = controller.before_batch(1,0)
        controller.after_batch([],[])
        decode = controller.before_batch(0,1)
        self.assertGreater(prefill['candidate']['expert_bytes'],decode['candidate']['expert_bytes'])
        self.assertFalse(prefill['physical_split_applied'])

    def test_batch_lifecycle_enforced(self):
        controller = Controller(100,9,12)
        with self.assertRaises(RuntimeError):
            controller.after_batch([],[])
        controller.before_batch(1,0)
        with self.assertRaises(RuntimeError):
            controller.before_batch(1,0)

    def test_macro_children_are_not_charged_again(self):
        ledger = RecoveryLedger()
        ledger.add('macro','kv_replay',10)
        ledger.add('internal','expert_upload',2,parent_id='macro')
        ledger.add('normal','expert_cpu',1)
        self.assertEqual(ledger.total_ms,11)
        with self.assertRaises(ValueError):
            ledger.add('macro','kv_replay',10)
        with self.assertRaises(ValueError):
            ledger.add('bad-child','expert_cpu',1,parent_id='missing')


if __name__ == '__main__':
    unittest.main()
