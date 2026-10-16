import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_kite_e2e import expert_reference,predicted_cpu,reconstruct_experts
from kite_e2e_core import EXPERT_BYTES


class DiagnosisTests(unittest.TestCase):
    def calibration(self):
        return dict(cpu_fp32_cache_bytes=2*EXPERT_BYTES,capture=dict(key=[0,99]),
                    domains=dict(expert_cpu=[1,32]),fixed=dict(expert_upload=1.,expert_cpu=1.2),
                    affine=dict(expert_cpu_intercept_ms=.1,expert_cpu_slope_ms=.1))

    def batches(self):
        return [dict(step=0,prefill=1,decode=0,
            decision=dict(candidate=dict(expert_objects=1)),
            expert_accesses=[([0,1],1),([0,2],20)],
            expert_counts=dict(expert_calls=2,expert_misses=2,expert_cpu=1,
                               cpu_fp32_conversions=1,expert_upload=1)),
            dict(step=1,prefill=0,decode=1,
            decision=dict(candidate=dict(expert_objects=1)),
            expert_accesses=[([0,1],1),([0,2],1),([0,3],1)],
            expert_counts=dict(expert_calls=5,expert_misses=4,expert_cpu=3,
                cpu_fp32_conversions=2,expert_upload=1,expert_hits=1,
                cpu_fp32_cache_hits=1,cpu_fp32_evictions=1))]

    def test_exact_cold_warm_hit_and_eviction_reconstruction(self):
        result = reconstruct_experts(self.batches(),self.calibration(),'kite',4*EXPERT_BYTES)
        self.assertTrue(result['all_batch_counters_match'])
        self.assertEqual(result['counters']['cpu_fp32_conversions'],2)
        self.assertEqual(result['by_phase']['decode']['cpu_cold'],1)
        self.assertEqual(result['by_phase']['decode']['cpu_warm'],1)
        self.assertEqual(result['by_phase']['decode']['gpu_hits'],1)

    def test_mismatched_raw_counter_is_rejected(self):
        batches = self.batches()
        batches[1]['expert_counts']['cpu_fp32_conversions'] = 1
        with self.assertRaisesRegex(ValueError,'batch 1'):
            reconstruct_experts(batches,self.calibration(),'kite',4*EXPERT_BYTES)

    def test_prediction_matches_frozen_domain_and_invalid_guard(self):
        c = self.calibration()
        self.assertAlmostEqual(predicted_cpu(c,'kite',1),.2)
        self.assertAlmostEqual(predicted_cpu(c,'baseline',1),1.2)
        self.assertEqual(predicted_cpu(c,'kite',33),float('inf'))
        c['affine']['expert_cpu_intercept_ms'] = -10.
        self.assertEqual(predicted_cpu(c,'kite',1),float('inf'))

    def test_reference_distance_is_separate_from_actual_cpu_bypass(self):
        result = expert_reference([dict(expert_accesses=[([0,1],1),([0,2],1),([0,1],1)])])
        self.assertEqual(result['minimum_finite_reuse_distance'],1)
        self.assertEqual(result['first_possible_hit_capacity'],2)
        self.assertEqual(result['points'][0]['hits'],1)
        self.assertEqual(result['unique_experts'],2)


if __name__=='__main__':
    unittest.main()
