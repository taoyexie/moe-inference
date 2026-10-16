import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from expert_cache import ExpertCosts
from expert_cache_bench import case_plan, measure_pass, paired_summary
from test_expert_cache_smoke import DriverBackend


class BenchTests(unittest.TestCase):
    def test_predeclared_cases_equal_length_with_control(self):
        plans = case_plan(['A','B','C'],64)
        self.assertEqual({len(plan) for plan in plans.values()},{192})
        self.assertEqual(set(plans['mixed_q_pressure']),{('A',32),('B',1),('C',1)})
        self.assertEqual(set(plans['stable_hot_control']),{('A',32),('B',32)})

    def test_keep_negative_paired_results(self):
        records = [dict(case='case',trial=i,warmup=False,policy=p,execution_wall_ms=v)
                   for i,times in enumerate(((10,8),(10,12)))
                   for p,v in zip(('fixed','calibrated'),times)]
        result = paired_summary(records)['case']
        self.assertAlmostEqual(result['paired_savings_mean'],0)
        self.assertAlmostEqual(result['paired_savings_min'],-.2)
        self.assertEqual(result['affine_faster_pairs'],1)

    def test_incomplete_pairs_rejected(self):
        with self.assertRaises(ValueError):
            paired_summary([dict(case='case',trial=0,warmup=False,policy='fixed',execution_wall_ms=1)])

    def test_pair_excludes_warmup(self):
        records = [dict(case='case',trial=0,warmup=False,policy=p,execution_wall_ms=v)
                   for p,v in (('fixed',10),('calibrated',8))]
        records += [dict(case='case',trial=-1,warmup=True,policy='fixed',execution_wall_ms=1000)]
        self.assertAlmostEqual(paired_summary(records)['case']['paired_savings_mean'],.2)

    def test_measured_loop_has_no_reference_validation_and_releases_slots(self):
        backend = DriverBackend()
        backend.validate = lambda *args: (_ for _ in ()).throw(AssertionError('Validation in timed pass'))
        keys = sorted(backend.sources)
        plan = case_plan(keys,2)['mixed_q_pressure']
        inputs = {item:backend.activation(*item)[0] for item in set(plan)}
        row = measure_pass(backend,plan,inputs,2,'calibrated',ExpertCosts(1,.1,.1,1.5,1,32),True)
        self.assertEqual(row['counters']['calls'],6)
        self.assertEqual(row['paths'],{'upload':1,'cpu':4,'hit':1})
        self.assertFalse(backend.live)
        self.assertGreaterEqual(row['execution_wall_ms'],row['invocation_wall_ms_sum'])

    def test_stable_hot_control_has_same_physical_counts(self):
        backend = DriverBackend()
        plan = case_plan(sorted(backend.sources),3)['stable_hot_control']
        inputs = {item:backend.activation(*item)[0] for item in set(plan)}
        costs = ExpertCosts(1,.1,.1,1.5,1,32)
        fixed = measure_pass(backend,plan,inputs,2,'fixed',costs)
        affine = measure_pass(backend,plan,inputs,2,'calibrated',costs)
        self.assertEqual(fixed['counters'],affine['counters'])
        self.assertEqual(fixed['counters']['misses'],2)


if __name__=='__main__':
    unittest.main()
