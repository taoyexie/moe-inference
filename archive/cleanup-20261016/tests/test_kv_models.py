"""Cost-model bookkeeping with synthetic fixtures; not GPU measurement evidence."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_kv_models import fit_and_evaluate


class KVModelTests(unittest.TestCase):
    def points(self, slope):
        return [{'context_tokens':length,'timings':{'restore':{'median_ms':.7},
                 'recompute':{'median_ms':10000+slope*length}}}
                for length in (128,256,512,1024,2048)]

    def test_exact_affine_fixture_and_holdout(self):
        result = fit_and_evaluate(self.points(2))
        self.assertEqual(result['calibration_lengths'],[128,512,2048])
        self.assertEqual(result['validation_lengths'],[256,1024])
        self.assertAlmostEqual(result['model']['recompute_affine']['slope_ms_per_token'],2)
        self.assertAlmostEqual(result['validation_mape']['recompute'],0)

    def test_negative_slope_not_hidden(self):
        result = fit_and_evaluate(self.points(-1))
        self.assertAlmostEqual(result['model']['recompute_affine']['slope_ms_per_token'],-1)
        self.assertFalse(result['positive_context_growth_supported'])
        self.assertFalse(result['runtime_extrapolation_allowed'])

    def test_validation_not_fitted(self):
        points = self.points(2)
        points[1]['timings']['recompute']['median_ms'] = 20000
        result = fit_and_evaluate(points)
        self.assertAlmostEqual(result['model']['recompute_affine']['slope_ms_per_token'],2)
        self.assertGreater(result['validation_mape']['recompute'],0)


if __name__ == '__main__':
    unittest.main()
