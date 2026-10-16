"""Metric arithmetic tests only; not Torch/CUDA execution."""
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from workload_baseline import distribution,token_metrics,measure_request


class BaselineMetricTests(unittest.TestCase):
    def test_first_token_and_intertoken_units(self):
        result = token_metrics(1_000_000,[11_000_000,15_000_000,21_000_000])
        self.assertEqual(result['ttft_ms'],10)
        self.assertEqual(result['completion_ms'],20)
        self.assertEqual(result['tpot_ms'],5)
        self.assertEqual(result['itl_ms'],[4,6])

    def test_one_token_has_no_decode_tpot(self):
        result = token_metrics(0,[1000])
        self.assertIsNone(result['tpot_ms'])
        self.assertEqual(result['itl_ms'],[])

    def test_invalid_timestamp_rejected(self):
        for ticks in ([],[1000,999],[-1]):
            with self.assertRaises(ValueError):
                token_metrics(0,ticks)

    def test_distribution_preserves_outlier(self):
        result = distribution([1,1,100])
        self.assertEqual(result['median'],1)
        self.assertEqual(result['max'],100)
        self.assertAlmostEqual(result['mean'],34)
        self.assertIsNone(distribution([]))

    def test_generation_stop_and_cache_semantics_with_fake_backend(self):
        # Explicitly a fake backend: verifies no extra forward after the EOS token.
        class Tensor:
            def __init__(self,shape):
                self.shape = shape
            def to(self,device):
                return self
        class Cache:
            def __init__(self):
                self.length = 0
            def get_seq_length(self):
                return self.length
        class Logits:
            def __init__(self,token):
                self.token = token
            def __getitem__(self,key):
                return self
            def argmax(self):
                return self
            def item(self):
                return self.token
        calls = []
        def model(**kwargs):
            cache = kwargs['past_key_values']
            calls.append(kwargs['input_ids'].shape[1])
            cache.length += calls[-1]
            return SimpleNamespace(past_key_values=cache,logits=Logits(7 if len(calls)==1 else 2))
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device:None),
            inference_mode=nullcontext,long='long',ones_like=lambda tensor:Tensor(tensor.shape),
            arange=lambda start,end,**kwargs:list(range(start,end)),
            tensor=lambda values,**kwargs:Tensor((len(values),len(values[0]))),
            ones=lambda shape,**kwargs:Tensor(shape))
        with patch.dict(sys.modules,{'transformers':SimpleNamespace(DynamicCache=Cache)}), \
                patch('workload_baseline.time.perf_counter_ns',side_effect=[0,10_000_000,15_000_000]):
            result = measure_request(torch,model,Tensor((1,3)),'cuda:0',{2},8)
        self.assertEqual(calls,[3,1])
        self.assertEqual(result['generated_token_ids'],[7,2])
        self.assertEqual(result['ttft_ms'],10)
        self.assertEqual(result['tpot_ms'],5)
        self.assertEqual(result['final_cache_length'],4)
        self.assertEqual(result['stop_reason'],'eos')


if __name__ == '__main__':
    unittest.main()
