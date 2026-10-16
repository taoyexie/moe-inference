"""Burst protocol/capacity bookkeeping only; no extra GPU tests."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_kite_e2e import expected_batches
from kite_burst_search import CASES,OPTIONS,PROTOCOL,mode_order,selection,structural_screen,workload
from kite_e2e_core import FramePool


class Tokenizer:
    def apply_chat_template(self,items,tokenize=False,**kwargs):
        return [1]*20 if tokenize else 'USER:'+items[0]['content']+':ASSISTANT'
    def encode(self,text,**kwargs):
        return [ord(t) for t in text]


class BurstSearchTests(unittest.TestCase):
    def test_real_scheduling_shape_and_independent_requests(self):
        specs = workload(Tokenizer(),CASES[0])
        self.assertEqual(len(specs),16)
        self.assertEqual([s['release_step'] for s in specs],list(range(16)))
        self.assertEqual([len(s['input_ids']) for s in specs],[512]+[20]*15)
        self.assertTrue(all(s['output_tokens']==1 for s in specs))
        self.assertIsNot(specs[1]['input_ids'],specs[2]['input_ids'])
        rows = expected_batches(specs)
        self.assertEqual(len(rows),16)
        self.assertTrue(all((r['prefill'],r['decode'])==(1,0) for r in rows))

    def test_declared_capacity_window_not_a_claimed_trace(self):
        for case,a,b in [(CASES[0],1820,2730),(CASES[1],1592,2389)]:
            screen = structural_screen(case,workload(Tokenizer(),case))
            self.assertEqual(screen['if_history_half_baseline_experts'],a)
            self.assertEqual(screen['if_history_half_kite_prefill_experts'],b)
        self.assertLess(1820,2565)
        self.assertLess(2565,2730)

    def test_lru_repeated_working_set_capacity_cliff(self):
        # Synthetic capacity logic check, not a hardware performance prediction.
        for capacity,expected_hits in [(1820,0),(2730,2565)]:
            pool = FramePool(32768*1024**2)
            pool.set_limits(capacity,1)
            hits = []
            for _ in range(2):
                count = 0
                for item in range(2565):
                    if pool.get('expert',item) is not None:
                        count += 1
                    else:
                        pool.admit('expert',item)
                hits.append(count)
            self.assertEqual(hits,[0,expected_hits])

    def test_bounded_selection_retains_all_completed_pairs(self):
        tasks = [dict(case_name=c['name'],mode=m,status='passed',
            workload_total_ms=100 if m=='baseline' else t)
            for c,t in [(CASES[0],101),(CASES[1],80)] for m in ('baseline','kite')]
        result = selection(tasks)
        self.assertEqual(result['selected']['case_name'],'repeat-short')
        self.assertEqual(len(result['all_cases']),2)
        with self.assertRaises(RuntimeError):
            selection(tasks[:1])

    def test_same_allocator_and_explicit_execution_order(self):
        self.assertEqual((PROTOCOL['current_weight'],PROTOCOL['refresh_batches']),(.65,16))
        self.assertEqual(OPTIONS['numerical_policy'],'performance_only')
        self.assertFalse(OPTIONS['require_dynamic_split'])  # All-P need not resize across time.
        self.assertEqual(mode_order(0,17),['baseline','kite'])
        self.assertEqual(mode_order(1,17),['kite','baseline'])
        self.assertEqual(mode_order(0,23),['kite','baseline'])


if __name__=='__main__':
    unittest.main()
