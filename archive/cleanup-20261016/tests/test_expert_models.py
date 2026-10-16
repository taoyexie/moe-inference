"""Model arithmetic tests: no torch, plotting dependencies or model downloads."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_expert_models import evaluate, linear_fit, predict_cpu


class ExpertModelTests(unittest.TestCase):
    def test_linear_fit_known_solution(self):
        fit=linear_fit([1,2,4,8],[.3,.4,.6,1.0])
        self.assertAlmostEqual(fit['intercept_ms'],.2)
        self.assertAlmostEqual(fit['slope_ms_per_token'],.1)

    def test_constant_q_rejected(self):
        with self.assertRaises(ValueError):
            linear_fit([1,1],[.2,.3])

    def test_choices_costed_with_measurements_not_predictions(self):
        model={'cpu_linear':{'intercept_ms':.2,'slope_ms_per_token':.02},
               'upload_constant_ms':.5,'cpu_fixed_ms':.3}
        points=[{'layer':0,'expert':1,'q':q,
                 'timings':{'upload_gpu':{'median_ms':.5},'cpu_return':{'median_ms':cpu}}}
                for q,cpu in [(1,.2),(128,2.0)]]
        result=evaluate(points,model,[1,128])
        self.assertEqual(result['decision_agreement_with_median_oracle'],1)
        self.assertAlmostEqual(result['mean_recovery_ms']['linear_costs'],.35)
        self.assertAlmostEqual(result['mean_recovery_ms']['fixed_costs'],1.1)
        self.assertNotAlmostEqual(predict_cpu(model,128),2.0)

    def test_bad_model_is_not_replaced_with_oracle(self):
        model={'cpu_linear':{'intercept_ms':1,'slope_ms_per_token':.1},
               'upload_constant_ms':.5,'cpu_fixed_ms':1.5}
        point={'layer':0,'expert':1,'q':1,
               'timings':{'upload_gpu':{'median_ms':.5},'cpu_return':{'median_ms':.2}}}
        result=evaluate([point],model,[1])
        self.assertEqual(result['decision_agreement_with_median_oracle'],0)
        self.assertEqual(result['saving_vs_fixed'],0)
        self.assertEqual(result['mean_recovery_ms']['linear_costs'],.5)
        self.assertEqual(result['mean_recovery_ms']['median_oracle'],.2)


if __name__=='__main__':
    unittest.main()
