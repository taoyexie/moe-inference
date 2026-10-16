"""Diagnostic bookkeeping/cleanup tests only, not Qwen or CUDA validation."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kite_numerical_diagnostic import Capture,TOLERANCES,accepted,query_indices,stock_row


class Module:
    def __init__(self):
        self.removed = 0
    def register_forward_hook(self,hook):
        return SimpleNamespace(remove=lambda:setattr(self,'removed',self.removed+1))


class NumericalDiagnosticTests(unittest.TestCase):
    def test_original_tolerance_not_relaxed_for_reported_failure(self):
        self.assertEqual(TOLERANCES,dict(relative_l2=.03,normalized_max_abs=.05))
        self.assertFalse(accepted(dict(relative_l2=.048165880143642426,
            normalized_max_abs=.04583333432674408)))
        self.assertTrue(accepted(dict(relative_l2=.03,normalized_max_abs=.05)))

    def test_decode_selects_same_token_in_full_and_incremental_forms(self):
        self.assertEqual(query_indices(stock_row('decode',9)),[8])
        self.assertEqual(query_indices(stock_row('decode',1)),[0])
        self.assertEqual(query_indices(dict(label='prefill',q_start=1,q_end=13)),list(range(1,13)))

    def engine(self):
        modules = [Module() for _ in range(10)]
        embed,norm,input_norm,q,k,v,attn,gate,moe,layer = modules
        layer.input_layernorm = input_norm
        layer.self_attn = attn
        attn.q_proj,attn.k_proj,attn.v_proj = q,k,v
        layer.mlp = moe
        moe.gate = gate
        functional = SimpleNamespace(scaled_dot_product_attention=lambda *a,**k:'unchanged-output')
        torch = SimpleNamespace(nn=SimpleNamespace(functional=functional))
        engine = SimpleNamespace(torch=torch,model=SimpleNamespace(model=SimpleNamespace(
            embed_tokens=embed,norm=norm,layers=[layer])))
        return engine,modules,functional

    def test_instrumentation_removed_when_forward_fails(self):
        engine,modules,functional = self.engine()
        original = functional.scaled_dot_product_attention
        capture = Capture(engine)
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with capture.record([stock_row('decode',9)]):
                self.assertIsNot(functional.scaled_dot_product_attention,original)
                raise RuntimeError('intentional')
        self.assertIs(functional.scaled_dot_product_attention,original)
        self.assertTrue(all(m.removed==1 for m in modules))
        self.assertFalse(capture.handles)

    def test_oracle_must_not_replace_efficient_attention_output(self):
        engine,modules,functional = self.engine()
        class Tensor:
            def float(self):
                return self
        functional.scaled_dot_product_attention = lambda *a,**k:'actual'
        capture = Capture(engine)
        fake_attention = SimpleNamespace(SDPBackend=SimpleNamespace(MATH='math'),
            sdpa_kernel=lambda _:__import__('contextlib').nullcontext())
        with patch.dict(sys.modules,{'torch.nn.attention':fake_attention}), \
             patch('kite_numerical_diagnostic.error_metrics',return_value={'relative_l2':0.}):
            with capture.record([]):
                self.assertEqual(functional.scaled_dot_product_attention(Tensor(),Tensor(),Tensor()),'actual')
        self.assertEqual(len(capture.data['attention_oracle']),1)


if __name__=='__main__':
    unittest.main()
