"""Lightweight bookkeeping tests. No torch, network, GPU, or model weights."""
import json
import hashlib
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from kite_trace import (GIB, ExpertRecorder, cache_snapshot, check_gpu_budget,
                        mapped_device, parse_gpus, parse_meminfo, write_json)
from qwen_smoke import (EXPECTED, validate_config, verify_checkpoint,
                        normalize_modelscope_files, content_identity, verify_file_hashes,
                        initialize_cuda_stats)


class FakeTensor:
    def __init__(self, shape, element_size=2, device='cpu'):
        self.shape, self.device, self.dtype = shape, device, 'torch.bfloat16'
        self.size = element_size

    def numel(self):
        value = 1
        for dim in self.shape:
            value *= dim
        return value

    def element_size(self):
        return self.size


class FakeCache:
    def __init__(self, length):
        self.key_cache = [FakeTensor((1,4,length,128)) for _ in range(48)]
        self.value_cache = [FakeTensor((1,4,length,128)) for _ in range(48)]


class TraceTests(unittest.TestCase):
    def test_cuda_init_precedes_statistics_reset(self):
        calls=[]
        class FakeCuda:
            initialized=False
            def init(self):
                self.initialized=True
                calls.append('init')
            def synchronize(self, device):
                self.assert_initialized()
                calls.append(('synchronize',device))
            def reset_peak_memory_stats(self, device):
                self.assert_initialized()
                calls.append(('reset',device))
            def assert_initialized(self):
                if not self.initialized:
                    raise RuntimeError('Invalid device argument')
        fake=SimpleNamespace(cuda=FakeCuda())
        with self.assertRaisesRegex(RuntimeError,'Invalid device argument'):
            fake.cuda.reset_peak_memory_stats('cuda:0')
        initialize_cuda_stats(fake, 'cuda:0')
        self.assertEqual(calls,['init',('synchronize','cuda:0'),('reset','cuda:0')])

    def test_empty_experts_are_not_access_events(self):
        recorder = ExpertRecorder([0], 2)
        recorder.begin(0, 'prefill', 3, 0)
        recorder.observe(0, 0, FakeTensor((0,32)), 100, 'cpu')
        recorder.observe(0, 1, FakeTensor((3,32)), 100, 'cpu')
        recorder.observe(0, 2, FakeTensor((3,32)), 100, '0')
        events, checks = recorder.finish()
        self.assertTrue(checks['pass'])
        self.assertEqual(checks['ignored_empty_calls'], 1)
        self.assertEqual([e['expert'] for e in events], [1,2])

    def test_shared_history_order_across_phases(self):
        recorder = ExpertRecorder([0], 1)
        events = []
        for step, phase in [(0,'prefill'),(1,'decode')]:
            recorder.begin(step, phase, 1, step)
            recorder.observe(0, 1, FakeTensor((1,32)), 100, 'cpu')
            part, checks = recorder.finish()
            events.extend(part)
        self.assertEqual([e['sequence'] for e in events], [0,1])
        self.assertEqual([e['phase'] for e in events], ['prefill','decode'])

    def test_missing_layer_is_detected(self):
        recorder = ExpertRecorder([0,1], 1)
        recorder.begin(0, 'prefill', 1, 0)
        recorder.observe(0, 1, FakeTensor((1,32)), 100, 'cpu')
        _, checks = recorder.finish()
        self.assertFalse(checks['pass'])

    def test_wrong_routed_count_is_detected(self):
        recorder = ExpertRecorder([0], 8)
        recorder.begin(0, 'decode', 1, 10)
        recorder.observe(0, 1, FakeTensor((7,32)), 100, 'cpu')
        _, checks = recorder.finish()
        self.assertFalse(checks['pass'])

    def test_kv_bytes_and_shape(self):
        snapshot = cache_snapshot(FakeCache(128),48,128,4,128)
        self.assertEqual(snapshot['logical_tensor_bytes'], 12*1024**2)
        with self.assertRaises(RuntimeError):
            cache_snapshot(FakeCache(128),48,129,4,128)

    def test_memory_units(self):
        memory = parse_meminfo('MemTotal: 20 kB\nMemAvailable: 10 kB\nHugePages_Total: 0')
        self.assertEqual(memory['MemAvailable'], 10240)

    def test_placement_longest_prefix(self):
        mapping = {'':'cpu','model.layers.0':0,'model.layers.0.mlp.experts.3':'cpu'}
        self.assertEqual(mapped_device('model.layers.0.mlp.experts.3.gate_proj.weight',mapping),'cpu')
        self.assertEqual(mapped_device('model.layers.0.self_attn.q_proj.weight',mapping),'0')
        self.assertEqual(mapped_device('model.norm.weight',mapping),'cpu')

    def test_resource_guard_rejects_busy_or_small_gpu(self):
        gpus = parse_gpus('0, NVIDIA A100-PCIE-40GB, 40960, 0, 40960, 0')
        check_gpu_budget({'gpus':gpus},24)
        with self.assertRaises(RuntimeError):
            check_gpu_budget({'gpus':gpus},39)
        gpus[0]['used_bytes'] = 31*GIB
        with self.assertRaises(RuntimeError):
            check_gpu_budget({'gpus':gpus},24)

    def test_config_guard(self):
        validate_config(dict(EXPECTED))
        bad = {**EXPECTED,'num_experts':4}
        with self.assertRaises(RuntimeError):
            validate_config(bad)

    def test_checkpoint_must_be_complete(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            folder=Path(directory)
            write_json(folder/'config.json',dict(EXPECTED))
            manifest={'model_id':'Qwen/Qwen3-30B-A3B','revision':'a'*40,
                      'config':dict(EXPECTED),'weights_ready':True,
                      'shards':[{'name':'model-00001-of-00001.safetensors','bytes':10}]}
            write_json(folder/'kite_manifest.json',manifest)
            with self.assertRaises(RuntimeError):
                verify_checkpoint(folder)

            # A wrong-size file must not be accepted as a complete shard.
            (folder/manifest['shards'][0]['name']).write_bytes(b'abc')
            with self.assertRaises(RuntimeError):
                verify_checkpoint(folder)

    def test_modelscope_file_identity_is_stable(self):
        raw = [{'Path': name, 'Type':'blob', 'Size':10, 'Sha256':'a'*64} for name in
               ['config.json','tokenizer_config.json','model.safetensors.index.json','model-1.safetensors']]
        normal = normalize_modelscope_files(raw)
        self.assertEqual(content_identity(normal), content_identity(normalize_modelscope_files(list(reversed(raw)))))
        changed = [dict(x) for x in normal]
        changed[0]['sha256']='b'*64
        self.assertNotEqual(content_identity(normal), content_identity(changed))

    def test_modelscope_missing_hash_is_rejected(self):
        with self.assertRaises(RuntimeError):
            normalize_modelscope_files([{'Path':'config.json','Size':10,'Type':'blob'}])

    def test_modelscope_unsafe_path_is_rejected(self):
        with self.assertRaises(RuntimeError):
            normalize_modelscope_files([{'Path':'../model.safetensors','Size':10,'Sha256':'a'*64}])

    def test_modelscope_hash_and_modification_guard(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            folder = Path(directory)
            config_data = json.dumps(dict(EXPECTED)).encode('utf-8')
            (folder/'config.json').write_bytes(config_data)
            record={'name':'config.json','bytes':len(config_data),
                    'sha256':hashlib.sha256(config_data).hexdigest()}
            verified = verify_file_hashes(folder,[record])
            manifest={'model_id':'Qwen/Qwen3-30B-A3B','source':'modelscope','revision':'master',
                      'config':dict(EXPECTED),'weights_ready':True,'shards':[],
                      'content_files':[record],'content_manifest_sha256':content_identity([record]),
                      'verified_files':verified}
            write_json(folder/'kite_manifest.json',manifest)
            self.assertEqual(verify_checkpoint(folder)['source'],'modelscope')
            stat=(folder/'config.json').stat()
            os.utime(folder/'config.json', ns=(stat.st_atime_ns,stat.st_mtime_ns+2_000_000_000))
            with self.assertRaises(RuntimeError):
                verify_checkpoint(folder)
            (folder/'config.json').write_bytes(b'x'*len(config_data))
            with self.assertRaises(RuntimeError):
                verify_file_hashes(folder,[record])


if __name__ == '__main__':
    unittest.main()
