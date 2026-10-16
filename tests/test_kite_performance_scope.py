"""Performance-run policy/control-flow tests, not additional precision tests."""
import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from kite_advantage import OPTIONS,PROTOCOL,performance_validation_record,run
from kite_e2e_runtime import Runtime


class PerformanceScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'tmp').mkdir(exist_ok=True)

    def runtime(self,finite=True):
        runtime = Runtime.__new__(Runtime)
        runtime.options = dict(numerical_policy='performance_only')
        runtime.calibration_checks = []
        runtime.torch = SimpleNamespace(isfinite=lambda x:SimpleNamespace(all=lambda:finite))
        return runtime

    def test_replay_cost_sample_no_longer_calls_precision_comparison(self):
        runtime = self.runtime()
        with patch('expert_paths.error_metrics',side_effect=AssertionError('Must not run precision comparison')):
            runtime.check_startup_replay('actual',None,128,False,0)
        check, = runtime.calibration_checks
        self.assertEqual(check['validation'],'finite_only')
        self.assertTrue(check['finite'])
        self.assertNotIn('error',check)

    def test_nan_inf_still_stops_and_retains_record(self):
        runtime = self.runtime(False)
        with self.assertRaisesRegex(RuntimeError,'NaN/Inf'):
            runtime.check_startup_replay('actual',None,512,False,0)
        self.assertFalse(runtime.calibration_checks[0]['finite'])

    def test_skipped_quality_check_is_not_mislabeled_passed(self):
        record = performance_validation_record()
        self.assertEqual(record['status'],'not_run_for_performance_trial')
        self.assertFalse(record['accuracy_certified'])
        self.assertFalse(record['checks'])
        self.assertEqual(OPTIONS['numerical_policy'],'performance_only')

    def test_no_allocator_or_workload_retuning(self):
        self.assertEqual(PROTOCOL['budgets_mib'],[8192])
        self.assertEqual((PROTOCOL['current_weight'],PROTOCOL['refresh_batches']),(.65,16))
        self.assertEqual(PROTOCOL['bursts']['release_steps'],[17,49,81])
        self.assertEqual(PROTOCOL['cpu_fp32_cache_gib'],8)

    def test_runner_goes_directly_to_two_performance_workloads(self):
        class Tensor:
            def __setitem__(self,index,value):
                pass
            def all(self):
                return True
        tensor = Tensor()
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1),
            version=SimpleNamespace(cuda='11.8'),set_num_threads=lambda n:None,
            set_num_interop_threads=lambda n:None,manual_seed=lambda n:None,device=lambda s:s,
            inference_mode=contextlib.nullcontext,bfloat16='bf16',zeros=lambda *a,**k:tensor,
            full=lambda *a,**k:tensor,finfo=lambda dtype:SimpleNamespace(min=-1e30),
            isfinite=lambda x:tensor,nn=SimpleNamespace(functional=SimpleNamespace(scaled_dot_product_attention=lambda *a,**k:tensor)))
        fake_attention = SimpleNamespace(SDPBackend=SimpleNamespace(EFFICIENT_ATTENTION='efficient'),
            sdpa_kernel=lambda _:contextlib.nullcontext())
        ready = dict(status='ready',checks=dict(software={},checkpoint={}),resources={},errors=[],blockers=[])
        monitor = Mock(error=None)
        monitor.record.return_value = {}
        calls = []
        def execute(engine,tokenizer,specs,budget,mode,output,mon,**kwargs):
            calls.append(mode)
            self.assertEqual(kwargs['runtime_options']['numerical_policy'],'performance_only')
            return dict(status='passed',budget_mib=budget,mode=mode,workload_total_ms=100 if mode=='baseline' else 90)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary, \
             patch('kite_advantage.new_output',return_value=Path(temporary)), \
             patch('kite_advantage.readiness',return_value=ready), \
             patch('kite_advantage.initialize_cuda_stats'), \
             patch('kite_advantage.os.sched_getaffinity',return_value=set(range(96)),create=True), \
             patch('kite_advantage.os.sched_setaffinity',create=True), \
             patch.dict(sys.modules,{'torch':fake_torch,'torch.nn.attention':fake_attention}), \
             patch('kite_advantage.ResourceMonitor',return_value=monitor), \
             patch('kite_advantage.load_engine',return_value=(Mock(),Mock(),{})), \
             patch('kite_advantage.workload',return_value=[]), \
             patch('kite_advantage.execute_task',side_effect=execute), \
             patch('kite_e2e.validate') as precision:
            args = SimpleNamespace(command='search',checkpoint='/unused',seed=17,selection=None,results_dir='/unused')
            run(args)
            precision.assert_not_called()
            self.assertEqual(set(calls),{'baseline','kite'})
            record = json.loads((Path(temporary)/'validation.json').read_text())
            self.assertEqual(record,performance_validation_record())


if __name__=='__main__':
    unittest.main()
