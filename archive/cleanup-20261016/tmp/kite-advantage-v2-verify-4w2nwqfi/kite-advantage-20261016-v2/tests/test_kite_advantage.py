"""Structural feasibility and policy logic; not measured CUDA performance."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_kite_e2e import expected_batches
from kite_advantage import OPTIONS,PROTOCOL,choose_selection,specifications,structural_screen,workload
from kite_controller import Controller,CostCalibration
from kite_e2e_core import RuntimeCosts


class AdvantageTests(unittest.TestCase):
    def test_exact_schedule_and_no_unbounded_batch(self):
        result = structural_screen()
        self.assertEqual(result['batch_count'],104)
        self.assertEqual(result['output_tokens'],248)
        self.assertEqual(result['input_tokens'],109568)
        self.assertEqual(result['query_tokens'],109806)
        self.assertEqual(result['max_active'],4)
        self.assertEqual(result['max_kv_pages'],332)
        self.assertLess(result['max_kv_bytes'],OPTIONS['host_kv_limit_bytes'])
        self.assertLess(result['max_mask_bytes'],3*1024**3)
        self.assertEqual([r['step'] for r in result['batches'] if r['prefill'] and r['requests']>1],[17,49,81])
        specs = [{**s,'input_ids':[1]*s['input_tokens']} for s in specifications()]
        rows = expected_batches(specs)
        self.assertEqual(len(rows),104)
        self.assertEqual((rows[17]['prefill'],rows[17]['decode']),(3,1))
        self.assertEqual(rows[18]['query_tokens'],4)

    def test_conditional_history_ninety_percent_signal_can_cover_the_shock(self):
        peak = structural_screen()['max_kv_pages']
        decisions = {}
        for mode,weight in [('baseline',0.),('kite',.65)]:
            controller = Controller(8190*1024**2,current_weight=weight)
            controller.proposal = max(controller.choices,key=lambda c:c.expert_bytes)
            controller.costs = object()  # Structural condition, not a forced runtime choice.
            decisions[mode] = controller.before_batch(0,4)['candidate']
        self.assertLess(decisions['baseline']['kv_objects'],peak)
        self.assertGreaterEqual(decisions['kite']['kv_objects'],peak)
        self.assertEqual(decisions['kite']['kv_objects'],341)

    def test_real_prompts_are_distinct_without_forced_experts(self):
        class Tokenizer:
            def apply_chat_template(self,items,**kwargs):
                return 'USER:'+items[0]['content']+':ASSISTANT'
            def encode(self,text,**kwargs):
                return [ord(ch) for ch in text]
        search,confirm = workload(Tokenizer(),'search'),workload(Tokenizer(),'confirmation')
        self.assertTrue(all(len(s['input_ids'])==s['input_tokens'] for s in search+confirm))
        for step in (17,49,81):
            burst = [s for s in search if s['release_step']==step]
            self.assertEqual(len({s['source_text'] for s in burst}),3)
        self.assertFalse(set(s['source_text'] for s in search)&set(s['source_text'] for s in confirm))

    def test_no_positive_selection_from_negative_measured_times(self):
        rows = [dict(status='passed',budget_mib=8192,mode=m,workload_total_ms=t)
                for m,t in [('baseline',100),('kite',101)]]
        self.assertEqual(choose_selection(rows)['status'],'no_positive_result')
        rows[1]['workload_total_ms'] = 90
        self.assertEqual(choose_selection(rows)['status'],'selected')
        self.assertEqual(PROTOCOL['budgets_mib'],[8192])

    def test_cold_state_and_robust_estimator_do_not_drop_raw_evidence(self):
        cal = CostCalibration()
        for q in (1,8,32):
            for trial in range(5):
                cal.add('expert_upload',q,50. if trial==0 else 1.)
                cal.add('expert_cpu',q,.1+.05*q)
        for context in (128,512):
            for _ in range(2):
                cal.add('kv_restore',context,.5)
                cal.add('kv_replay',context,10+.01*context)
        fit = cal.freeze()
        kite = RuntimeCosts(cal,fit,'kite',robust=True,cpu_cold_setup_ms=2.)
        base = RuntimeCosts(cal,fit,'baseline',robust=True,cpu_cold_setup_ms=2.)
        self.assertEqual(kite.fixed['expert_upload'],1.)
        self.assertEqual(kite.fixed,base.fixed)
        self.assertGreater(kite.expert_for_cache(1,False)[1],1.)
        self.assertLess(kite.expert_for_cache(1,True)[1],1.)
        self.assertEqual(kite.expert(1),kite.expert_for_cache(1,False))
        self.assertEqual(len(cal.samples['expert_upload']),15)
        self.assertEqual(kite.expert_for_cache(33,True)[1],float('inf'))
        with self.assertRaises(ValueError):
            RuntimeCosts(cal,fit,'kite',cpu_cold_setup_ms=-1)

    def test_robust_freeze_is_not_preempted_by_invalid_raw_ols(self):
        cal = CostCalibration()
        for q in (1,8,32):
            for trial in range(5):
                cal.add('expert_upload',q,1.)
                cal.add('expert_cpu',q,10000. if q==32 and trial==0 else .1+.05*q)
        for context in (128,512):
            cal.add('kv_restore',context,.5)
            cal.add('kv_replay',context,10+.01*context)
        with self.assertRaises(ValueError):
            cal.freeze()
        controller = Controller(8190*1024**2)
        controller.before_batch(1,0)
        controller.after_batch([((0,1),1)],[])
        controller.calibration = cal
        controller.finish_calibration(robust=True)
        self.assertAlmostEqual(controller.costs.expert_cpu_slope_ms,.05)
        self.assertEqual(len(cal.samples['expert_cpu']),15)


if __name__=='__main__':
    unittest.main()
