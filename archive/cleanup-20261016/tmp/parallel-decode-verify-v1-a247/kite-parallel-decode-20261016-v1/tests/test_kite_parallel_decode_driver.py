"""Mocked orchestration tests, not GPU precision or performance measurements."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from kite_parallel_decode import run


class Tokenizer:
    def apply_chat_template(self,items,**kwargs):
        return 'USER:'+items[0]['content']+':ASSISTANT'
    def encode(self,text,**kwargs):
        return list(map(ord,text))


class ParallelDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'tmp').mkdir(exist_ok=True)

    def run_mock(self,fail_at=None,monitor_error=None):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1),
            version=SimpleNamespace(cuda='11.8'),set_num_threads=lambda n:None,set_num_interop_threads=lambda n:None,
            manual_seed=lambda n:None,device=lambda s:s,inference_mode=contextlib.nullcontext)
        fake_attention = SimpleNamespace(SDPBackend=SimpleNamespace(EFFICIENT_ATTENTION='efficient'),
            sdpa_kernel=lambda _:contextlib.nullcontext())
        tokenizer = Tokenizer()
        fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:tokenizer))
        ready = dict(status='ready',checks=dict(checkpoint={}),resources={},errors=[],blockers=[])
        monitor = Mock(error=monitor_error)
        monitor.record.return_value = {}
        calls = []
        def execute(engine,tok,specs,budget,mode,output,mon,**kwargs):
            calls.append((budget,mode))
            if len(calls)==fail_at:
                raise RuntimeError('simulated resource failure')
            self.assertEqual(len(specs),4)
            self.assertEqual(kwargs['runtime_options']['numerical_policy'],'performance_only')
            return dict(status='passed',budget_mib=budget,mode=mode,workload_total_ms=100 if mode=='baseline' else 90,
                execution_inclusive_ms=70 if mode=='baseline' else 60,
                kv_counts={'kv_restore_pages':1000 if mode=='baseline' else 100})
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary,contextlib.ExitStack() as stack:
            path = Path(temporary)
            patches = [('new_output',dict(return_value=path)),('readiness',dict(return_value=ready)),
                ('initialize_cuda_stats',{}),('os.sched_setaffinity',dict(create=True)),
                ('ResourceMonitor',dict(return_value=monitor)),('load_engine',dict(return_value=(Mock(),tokenizer,{}))),
                ('execute_task',dict(side_effect=execute))]
            mocks = {name:stack.enter_context(patch('kite_parallel_decode.'+name,**kwargs)) for name,kwargs in patches}
            stack.enter_context(patch.dict(sys.modules,{'torch':fake_torch,'torch.nn.attention':fake_attention,'transformers':fake_transformers}))
            precision = stack.enter_context(patch('kite_e2e.validate',side_effect=AssertionError('No precision probes allowed.')))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            args = SimpleNamespace(command='search',checkpoint='/unused',seed=17,selection=None,results_dir='/unused')
            if fail_at is not None or monitor_error:
                with self.assertRaises(RuntimeError):
                    run(args)
            else:
                run(args)
            precision.assert_not_called()
            mocks['load_engine'].assert_called_once()
            monitor.close.assert_called_once()
            summary = json.loads((path/'summary.json').read_text())
            return calls,summary

    def test_load_once_and_exactly_four_actual_task_calls(self):
        calls,summary = self.run_mock()
        self.assertEqual(calls,[(6144,'baseline'),(6144,'kite'),(7168,'kite'),(7168,'baseline')])
        self.assertEqual(summary['status'],'passed')
        self.assertEqual(len(summary['selection']['all_cases']),2)
        self.assertIn('job_wall_ms_including_loading_and_all_comparisons',summary)

    def test_resource_failure_does_not_continue_search(self):
        calls,summary = self.run_mock(fail_at=2)
        self.assertEqual(len(calls),2)
        self.assertEqual(summary['status'],'failed')
        self.assertEqual(len(summary['tasks']),1)
        self.assertNotIn('selection',summary)

    def test_monitor_failure_cannot_report_success(self):
        _,summary = self.run_mock(monitor_error='another GPU process observed')
        self.assertEqual(summary['status'],'failed')
        self.assertEqual(summary['error_type'],'ResourceInterference')


if __name__=='__main__':
    unittest.main()
