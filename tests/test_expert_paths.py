"""Protocol/bookkeeping tests only. These do not validate real CPU/CUDA kernels."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from expert_paths import expert_shards, input_source, parse_int_list, select_experts, summarize, timed
from kite_trace import ExpertRecorder, write_json
from qwen_smoke import attach_hooks


class ExpertPathTests(unittest.TestCase):
    def test_controlled_input_provenance(self):
        self.assertEqual(input_source(8, 8), 'native_captured_input')
        self.assertIn('first_q_rows', input_source(1, 8))
        self.assertIn('not_new_real_routing', input_source(128, 8))

    def test_q_validation(self):
        self.assertEqual(parse_int_list('8,1,2', 1, 128), [1,2,8])
        for value in ('', '0,1', '1,1', '129', 'a'):
            with self.assertRaises(ValueError):
                parse_int_list(value, 1, 128)

    def test_summary_preserves_negative_extra_penalties(self):
        stats = summarize([-2, 0, 2])
        self.assertEqual(stats['median_ms'], 0)
        self.assertAlmostEqual(stats['p10_ms'], -1.6)
        self.assertEqual(stats['n'], 3)

    def test_nonfinite_timing_rejected(self):
        with self.assertRaises(ValueError):
            summarize([float('nan')])

    def test_reference_selection_not_decode_or_unrelated_layer(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            folder = Path(temporary)
            write_json(folder/'summary.json', {'status':'passed', 'is_tiny_synthetic_test':False})
            write_json(folder/'request.json', {'request_id':'smoke-0001'})
            records = []
            for layer in (0,23,47):
                for expert, q in ((4, 5), (2, 5), (1, 3)):
                    records.append({'layer':layer, 'expert':expert, 'routed_tokens':q,
                                    'request_id':'smoke-0001', 'phase':'prefill', 'step':0})
            records.append({'layer':0, 'expert':9, 'routed_tokens':99, 'phase':'decode'})
            (folder/'expert_trace.jsonl').write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
            _, _, selected = select_experts(folder, [0,23,47])
            self.assertEqual([r['expert'] for r in selected], [2,2,2])
            with self.assertRaises(RuntimeError):
                select_experts(folder, [1])

    def test_only_selected_weight_names(self):
        prefix = 'model.layers.23.mlp.experts.2.'
        index = {'weight_map':{prefix+p+'.weight':'model-1.safetensors'
                              for p in ('gate_proj','up_proj','down_proj')}}
        mapped = expert_shards(index, {'layer':23, 'expert':2})
        self.assertEqual(len(mapped), 3)
        index['weight_map'][prefix+'gate_proj.weight'] = '../model-1.safetensors'
        with self.assertRaises(RuntimeError):
            expert_shards(index, {'layer':23, 'expert':2})

    def test_wall_timing_waits_for_completion(self):
        calls = []
        fake = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda d: calls.append('sync')))
        def operation():
            calls.append('operation')
            return 'ready'
        duration, value = timed(fake, 'cuda:0', operation)
        self.assertEqual(calls, ['sync', 'operation', 'sync'])
        self.assertGreaterEqual(duration, 0)
        self.assertEqual(value, 'ready')

    def test_optional_observer_sees_real_recording_context(self):
        hooks, observations = [], []
        module = SimpleNamespace(parameters=lambda: [],
                                 register_forward_pre_hook=lambda hook: hooks.append(hook))
        model = SimpleNamespace(named_modules=lambda: [('model.layers.0.mlp.experts.0', module)],
                                config=SimpleNamespace(num_hidden_layers=1, num_experts=1),
                                hf_device_map={'':'0'})
        recorder = ExpertRecorder([0], 1)
        recorder.begin(0, 'prefill', 3, 0)
        attach_hooks(model, recorder, lambda l, e, x, c: observations.append((l,e,c['phase'],x.shape)))
        hooks[0](module, [SimpleNamespace(shape=(3,2048), device='cuda:0')])
        events, checks = recorder.finish()
        self.assertTrue(checks['pass'])
        self.assertEqual(events[0]['routed_tokens'], 3)
        self.assertEqual(observations, [(0,0,'prefill',(3,2048))])


if __name__ == '__main__':
    unittest.main()
