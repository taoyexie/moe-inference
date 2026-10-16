"""No-CUDA resource/parser/launch-order regression tests."""
import contextlib
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from kite_trace import GIB,parse_gpu_pids
from kite_readiness import readiness,verify_source_bundle
from kite_e2e import ResourceMonitor
from kite_advantage import run


def query(stdout='',code=0):
    return dict(returncode=code,stdout=stdout,stderr='')


class ReadinessTests(unittest.TestCase):
    def test_pid_parser_does_not_silently_treat_unknown_as_idle(self):
        self.assertEqual(parse_gpu_pids(query()),set())
        self.assertEqual(parse_gpu_pids(query(' 23308\n23308\n')),{'23308'})
        for text in ('N/A','No permission','0','٢٣٣٠٨','123\nunknown'):
            with self.assertRaises(RuntimeError):
                parse_gpu_pids(query(text))
        with self.assertRaises(RuntimeError):
            parse_gpu_pids(query('123',code=1))

    def test_readiness_import_does_not_import_torch(self):
        command = "import sys;sys.path.insert(0,'scripts');import kite_readiness;assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable,'-c',command],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_bundle_hash_and_traversal_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/'source.py').write_bytes(b'pass\n')
            record = dict(path='source.py',bytes=5,sha256=hashlib.sha256(b'pass\n').hexdigest())
            manifest = dict(name='test',files=[record])
            (folder/'bundle_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            self.assertEqual(verify_source_bundle(folder)['files'],1)
            (folder/'source.py').write_bytes(b'fail\n')
            with self.assertRaisesRegex(RuntimeError,'changed'):
                verify_source_bundle(folder)
            record['path'] = '../outside.py'
            (folder/'bundle_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'Unsafe'):
                verify_source_bundle(folder)

    def test_ready_busy_and_unknown_gpu_are_separate_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/'model.safetensors.index.json').write_text(json.dumps(dict(metadata=dict(total_size=30*GIB))),encoding='utf-8')
            snapshot = dict(effective_available_bytes=80*GIB,disk=dict(free_bytes=20*GIB),
                allowed_logical_cpus=list(range(96)),gpus=[dict(used_bytes=0,util_percent=0,free_bytes=40*GIB)])
            manifest = dict(model_id='Qwen/Qwen3-30B-A3B',weights_ready=True,content_manifest_sha256='a'*64)
            with patch('kite_readiness.verify_source_bundle',return_value={'status':'passed'}), \
                 patch('kite_readiness.require_versions',return_value={}), \
                 patch('kite_readiness.verify_checkpoint',return_value=manifest), \
                 patch('kite_readiness.resource_snapshot',return_value=snapshot):
                for raw,status in [(query(),'ready'),(query('23308\n'),'blocked'),(query('N/A'),'failed')]:
                    with patch('kite_readiness.run_readonly',return_value=raw):
                        report = readiness(folder,folder)
                    self.assertEqual(report['status'],status)
                    self.assertTrue(report['no_cuda_context_created'])
                    self.assertTrue(report['no_model_loaded'])

    def test_monitor_startup_records_empty_vs_multiple(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)/'monitor.json'
            for stdout,fragment in [('', 'empty'),('123\n23308\n','23308'),('N/A','unrecognized')]:
                with patch('kite_e2e.run_readonly',return_value=query(stdout)), \
                     self.assertRaisesRegex(RuntimeError,fragment):
                    ResourceMonitor(evidence_path=evidence)
                record = json.loads(evidence.read_text(encoding='utf-8'))
                self.assertEqual(record['status'],'failed')
                self.assertEqual(record['pid_query']['stdout'],stdout)

    def test_monitor_rejects_new_or_missing_gpu_pid(self):
        memory = lambda path:'MemAvailable: 10485760 kB' if path=='/proc/meminfo' else 'NSpid:\t123'
        with patch('kite_e2e.run_readonly',return_value=query('123')), \
             patch('kite_e2e.threading.Thread'),patch('kite_e2e.read_optional',side_effect=memory):
            monitor = ResourceMonitor()
            monitor.observe_once()
            with patch('kite_e2e.run_readonly',return_value=query('123\n23308')):
                with self.assertRaisesRegex(RuntimeError,'23308'):
                    monitor.observe_once()
            with patch('kite_e2e.run_readonly',return_value=query('')):
                with self.assertRaisesRegex(RuntimeError,'no longer visible'):
                    monitor.observe_once()
            monitor.close()

    def args(self):
        return SimpleNamespace(command='search',checkpoint='/unused',seed=17,selection=None,results_dir='/unused')

    def test_busy_runner_does_not_initialize_cuda_or_load_model(self):
        ready = dict(status='blocked',checks=dict(software={},checkpoint={}),resources={},errors=[],
                     blockers=[dict(message='Existing GPU processes: 23308')])
        with tempfile.TemporaryDirectory() as temporary, \
             patch('kite_advantage.new_output',return_value=Path(temporary)), \
             patch('kite_advantage.readiness',return_value=ready), \
             patch('kite_advantage.initialize_cuda_stats') as initialize, \
             patch('kite_advantage.load_engine') as load:
            with self.assertRaisesRegex(RuntimeError,'23308'):
                run(self.args())
            initialize.assert_not_called()
            load.assert_not_called()
            self.assertEqual(json.loads((Path(temporary)/'summary.json').read_text())['status'],'failed')

    def test_monitor_starts_before_expensive_model_loading(self):
        ready = dict(status='ready',checks=dict(software={},checkpoint={}),resources={},errors=[],blockers=[])
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
                                        sdpa_kernel=lambda backend:contextlib.nullcontext())
        monitor = Mock(error=None)
        monitor.record.return_value = dict(error=None,samples=[],allowed_gpu_pids=['123'])
        with tempfile.TemporaryDirectory() as temporary, \
             patch('kite_advantage.new_output',return_value=Path(temporary)), \
             patch('kite_advantage.readiness',return_value=ready), \
             patch('kite_advantage.initialize_cuda_stats'), \
             patch('kite_advantage.os.sched_getaffinity',return_value=set(range(96)),create=True), \
             patch('kite_advantage.os.sched_setaffinity',create=True), \
             patch.dict(sys.modules,{'torch':fake_torch,'torch.nn.attention':fake_attention}), \
             patch('kite_advantage.ResourceMonitor',return_value=monitor) as construct:
            def load(*args,**kwargs):
                construct.assert_called_once()
                self.assertEqual(kwargs['health_check'],monitor.check)
                raise RuntimeError('intentional stop before model loading')
            with patch('kite_advantage.load_engine',side_effect=load), \
                 self.assertRaisesRegex(RuntimeError,'intentional stop'):
                run(self.args())
            monitor.close.assert_called_once()


if __name__=='__main__':
    unittest.main()
