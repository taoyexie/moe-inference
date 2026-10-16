"""Local scheduling, capacity and evidence rules; never loads Torch or a model."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_kite_e2e import expected_batches
from kite_controller import Controller
from kite_e2e_core import FRAME_BYTES
from kite_parallel_decode import CASES,OPTIONS,PROTOCOL,mode_order,selection,structural_screen,workload
from screen_kite_parallel_decode import replay_segment


class Tokenizer:
    def apply_chat_template(self,items,**kwargs):
        return 'USER:'+items[0]['content']+':ASSISTANT'
    def encode(self,text,**kwargs):
        return list(map(ord,text))


class ParallelDecodeTests(unittest.TestCase):
    def test_exact_shape_and_all_complete_requests(self):
        specs = workload(Tokenizer())
        self.assertEqual([len(r['input_ids']) for r in specs],[8192,11264,11264,11264])
        rows = expected_batches(specs)
        self.assertEqual(len(rows),16)
        self.assertEqual((rows[0]['prefill'],rows[0]['decode']),(4,0))
        self.assertTrue(all((r['prefill'],r['decode'])==(0,4) for r in rows[1:]))
        shape = structural_screen()
        self.assertEqual((shape['output_tokens'],shape['max_kv_pages'],shape['input_tokens']),(64,332,41984))
        self.assertLess(shape['max_kv_bytes'],OPTIONS['host_kv_limit_bytes'])
        self.assertLess(shape['max_padded_mask_bytes'],4*1024**3)

    def test_conditional_window_not_forced_on_runtime(self):
        for case,base_k,kite_k in [(CASES[0],256,384),(CASES[1],298,448)]:
            choices = {}
            budget = case['budget_mib']*1024**2//FRAME_BYTES*FRAME_BYTES
            for mode,weight in [('baseline',0.),('kite',.65)]:
                c = Controller(budget,current_weight=weight)
                c.costs = object()
                choices[mode] = c.before_batch(0,4)['candidate']
            self.assertEqual(choices['baseline']['kv_objects'],base_k)
            self.assertEqual(choices['kite']['kv_objects'],kite_k)
            self.assertLess(base_k,332)
            self.assertGreaterEqual(kite_k,332)
        self.assertEqual(PROTOCOL['refresh_batches'],16)

    def test_selection_does_not_reward_startup_only_gain(self):
        def pair(kite_e2e,kite_main,kite_restores):
            return [dict(status='passed',case_name='parallel-6g',budget_mib=6144,mode=m,
                workload_total_ms=e,execution_inclusive_ms=t,kv_counts={'kv_restore_pages':n})
                for m,e,t,n in [('baseline',100,70,1000),('kite',kite_e2e,kite_main,kite_restores)]]
        for args in [(90,71,1),(101,60,1),(90,60,1000)]:
            self.assertEqual(selection(pair(*args))['status'],'no_supported_positive_result')
        tasks = pair(90,60,1)
        self.assertEqual(selection(tasks)['status'],'selected')
        with self.assertRaises(RuntimeError):
            selection(tasks+[tasks[-1]])

    def test_order_and_held_out_topics(self):
        search,confirm = workload(Tokenizer()),workload(Tokenizer(),True)
        self.assertFalse({s['source_text'] for s in search}&{s['source_text'] for s in confirm})
        self.assertEqual(mode_order(0,17),['baseline','kite'])
        self.assertEqual(mode_order(1,17),['kite','baseline'])
        self.assertEqual(mode_order(0,23),['kite','baseline'])
        self.assertEqual(OPTIONS['numerical_policy'],'performance_only')

    def test_reference_screen_read_and_write_touches(self):
        rows = [dict(expert_accesses=[([0,1],1),([0,2],1)],kv_read_groups=[
            dict(group_id=['r',i],context=128+i,objects=[['r',0],['r',1]])]) for i in range(3)]
        large = replay_segment(rows,2,2)
        small = replay_segment(rows,1,1)
        self.assertEqual(large['accesses'],dict(expert=6,kv=6))
        self.assertEqual(large['misses'],dict(expert=2,kv=2))
        self.assertEqual(small['misses'],dict(expert=6,kv=6))


if __name__=='__main__':
    unittest.main()
