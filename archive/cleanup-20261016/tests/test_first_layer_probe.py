"""Selective loading/flag lifecycle tests; no CUDA execution."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from first_layer_probe import bf16_reduction,fp32_projection_control,required_keys


class FirstLayerProbeTests(unittest.TestCase):
    def test_only_first_attention_and_selected_embedding_requested(self):
        index = dict(weight_map={'model.layers.0.self_attn.o_proj.weight':'a',
            'model.layers.0.self_attn.v_proj.weight':'a',
            'model.layers.0.mlp.experts.0.gate_proj.weight':'a',
            'model.layers.1.self_attn.o_proj.weight':'b','lm_head.weight':'z'})
        keys = required_keys(index,[151644,9370])
        self.assertEqual(len(keys),4)
        self.assertFalse(any('.mlp.' in k or 'layers.1.' in k or k=='lm_head.weight' for k in keys))
        with self.assertRaises(RuntimeError):
            required_keys(index,[-1])

    def test_reduction_flag_restored_on_error(self):
        flag = SimpleNamespace(allow_bf16_reduced_precision_reduction=True)
        torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=flag)))
        with self.assertRaises(RuntimeError):
            with bf16_reduction(torch,False):
                self.assertFalse(flag.allow_bf16_reduced_precision_reduction)
                raise RuntimeError('intentional')
        self.assertTrue(flag.allow_bf16_reduced_precision_reduction)

    def test_projection_control_restores_methods_and_tf32(self):
        class Weight:
            def detach(self):
                return self
            def float(self):
                return self
        original = lambda x:x
        modules = {name:SimpleNamespace(weight=Weight(),bias=None,forward=original)
                   for name in ('q_proj','k_proj','v_proj','o_proj')}
        flag = SimpleNamespace(allow_tf32=True)
        torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=flag)))
        with self.assertRaises(RuntimeError):
            with fp32_projection_control(torch,SimpleNamespace(**modules)):
                self.assertFalse(flag.allow_tf32)
                self.assertTrue(all(m.forward is not original for m in modules.values()))
                raise RuntimeError('intentional')
        self.assertTrue(flag.allow_tf32)
        self.assertTrue(all(m.forward is original for m in modules.values()))


if __name__=='__main__':
    unittest.main()
