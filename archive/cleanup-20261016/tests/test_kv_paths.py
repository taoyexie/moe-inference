"""Protocol-only tests. No local CPU/CUDA model inference is represented by these tests."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from kv_paths import object_spec, pack_tail, repeated_prefix, validate_args, validate_loading


class KVPathTests(unittest.TestCase):
    def test_fixed_object_not_entire_context(self):
        self.assertEqual(object_spec(128)['bytes'], 12*1024**2)
        self.assertEqual(object_spec(64)['bytes'], 6*1024**2)
        with self.assertRaises(ValueError):
            object_spec(129)

    def test_nested_controlled_inputs(self):
        self.assertEqual(repeated_prefix([3, 7, 2], 8), [3, 7, 2, 3, 7, 2, 3, 7])
        self.assertEqual(repeated_prefix([3, 7, 2], 5), repeated_prefix([3, 7, 2], 8)[:5])
        with self.assertRaises(ValueError):
            repeated_prefix([], 10)

    def test_incomplete_weight_loading_rejected(self):
        validate_loading({'unexpected_keys': ['lm_head.weight']})
        for info in ({'missing_keys':['layers.0.weight']}, {'unexpected_keys':['other.weight']},
                     {'mismatched_keys':['weight']}, {'error_msgs':['bad']}):
            with self.assertRaises(RuntimeError):
                validate_loading(info)

    def test_pilot_limits(self):
        args = SimpleNamespace(contexts='512,256', block_tokens=128, warmup=1, repeats=3,
                               cpu_threads=8, gpu_memory_gib=24, cpu_memory_gib=48)
        self.assertEqual(validate_args(args), [256,512])
        args.contexts = '64,256'
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_pack_uses_actual_same_tail_in_every_layer(self):
        calls = []
        class Tensor:
            shape = (1,4,512,128)
            dtype = 'bf16'
            device = SimpleNamespace(type='cuda')
            def __getitem__(self, key):
                calls.append(key)
                return self
        class Target:
            def __getitem__(self, key):
                return self
            def copy_(self, source):
                pass
        cache = SimpleNamespace(key_cache=[Tensor() for _ in range(48)],
                                value_cache=[Tensor() for _ in range(48)])
        pack_tail(SimpleNamespace(bfloat16='bf16'), cache, Target(), 512, 128)
        self.assertEqual(len(calls), 96)
        self.assertTrue(all(key[2] == slice(384,512) for key in calls))
        cache.key_cache.pop()
        with self.assertRaises(RuntimeError):
            pack_tail(SimpleNamespace(bfloat16='bf16'), cache, Target(), 512, 128)


if __name__ == '__main__':
    unittest.main()
