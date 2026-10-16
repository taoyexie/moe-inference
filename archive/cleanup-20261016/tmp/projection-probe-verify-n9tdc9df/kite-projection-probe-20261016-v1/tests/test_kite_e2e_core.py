"""Logical runtime tests, not GPU performance evidence."""
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kite_controller import CostCalibration,CostModel,Controller
from kite_e2e import PROTOCOL,choose_selection,new_requests,protocol_hash
from kite_e2e_core import (FRAME_BYTES,FramePool,Request,RuntimeCosts,allowed_attention,
                          packed_layout,ready_batch,validate_requests)


class FrameTests(unittest.TestCase):
    def pool(self):
        pool = FramePool(30*FRAME_BYTES)
        pool.set_limits(6,3)
        return pool

    def check(self,pool):
        frames = list(pool.free)
        for kind,entries in pool.entries.items():
            self.assertLessEqual(len(entries),pool.limits[kind])
            for entry in entries.values():
                self.assertEqual(len(entry['frames']),3 if kind=='expert' else 4)
                frames.extend(entry['frames'])
        self.assertEqual(sorted(frames),list(range(pool.frame_count)))

    def test_real_ownership_reuse_and_quotas(self):
        pool = self.pool()
        for key in range(8):
            pool.admit('expert',key)
        self.assertEqual(list(pool.entries['expert']),list(range(2,8)))
        pool.get('expert',2)
        self.assertEqual(pool.admit('expert',8)[1],3)
        for key in range(4):
            pool.admit('kv',key)
        self.assertEqual(list(pool.entries['kv']),[1,2,3])
        self.check(pool)
        self.assertEqual(len(pool.set_limits(2,6)),4)
        self.check(pool)

    def test_random_resizes_never_overlap_frames(self):
        pool = self.pool()
        rng = random.Random(17)
        for _ in range(1500):
            e,k = rng.randrange(8),rng.randrange(5)
            if 3*e+4*k<=30:
                pool.set_limits(e,k)
            kind = rng.choice(['expert','kv'])
            if rng.random()<.7:
                pool.admit(kind,rng.randrange(20))
            else:
                pool.remove(kind,rng.randrange(20))
            self.check(pool)
        pool.clear()
        self.assertEqual(len(pool.free),30)

    def test_infeasible_quota_does_not_mutate(self):
        pool = self.pool()
        pool.admit('expert','A')
        before = pool.snapshot()
        with self.assertRaises(ValueError):
            pool.set_limits(10,1)
        self.assertEqual(before,pool.snapshot())


class SchedulingTests(unittest.TestCase):
    def test_exact_main_schedule(self):
        specs = [dict(request_id='background',input_ids=list(range(2048)),release_step=0,output_tokens=32)]
        specs += [dict(request_id=f'{step}-{i}',input_ids=list(range(512)),release_step=step,output_tokens=8)
                  for step in (9,17,25) for i in range(3)]
        requests = new_requests(specs)
        validate_requests(requests)
        mixed = []
        for step in range(33):
            batch = ready_batch(requests,step)
            layout = packed_layout(batch)
            p = sum(row['phase']=='prefill' for row in layout)
            if p and p<len(batch):
                mixed.append((step,p,len(batch)-p))
            for req,row in zip(batch,layout):
                req.kv_length = row['after']
                req.generated.append(1)
        self.assertEqual(mixed,[(9,3,1),(17,3,1),(25,3,1)])
        self.assertTrue(all(r.done for r in requests))
        self.assertEqual(sum(len(r.generated) for r in requests),104)
        self.assertEqual(sum(len(r.prompt_ids) for r in requests),6656)
        self.assertFalse(ready_batch(requests,33))

    def test_packed_positions_and_isolation(self):
        d = Request('D',[1,2,3],0,2,generated=[4],kv_length=3)
        p = Request('P',[5,6],0,2)
        layout = packed_layout([d,p])
        self.assertEqual(layout[0]['after'],4)
        self.assertEqual(layout[1]['k_start'],4)
        self.assertEqual([any(allowed_attention(row,0,k) for row in layout) for k in range(6)],
                         [True,True,True,True,False,False])
        self.assertFalse(any(allowed_attention(row,1,3) for row in layout))
        self.assertTrue(allowed_attention(layout[1],1,4))
        self.assertFalse(allowed_attention(layout[1],1,5))
        self.assertEqual(d.prefix,[1,2,3])

    def test_invalid_state_or_overfull_batch_fails(self):
        with self.assertRaises(ValueError):
            packed_layout([Request('bad',[1,2],0,2,generated=[3],kv_length=1)])
        with self.assertRaises(RuntimeError):
            ready_batch([Request(str(i),[1],0,1) for i in range(5)],0)


class CostsTests(unittest.TestCase):
    def calibration(self):
        cal = CostCalibration()
        for q in (1,8,32):
            cal.add('expert_upload',q,1)
            cal.add('expert_cpu',q,.1+.1*q)
        for length in (128,512):
            cal.add('kv_restore',length,.5)
            cal.add('kv_replay',length,10+.01*length)
        return cal

    def test_same_samples_fixed_vs_affine_and_domain(self):
        cal = self.calibration()
        fit = cal.freeze()
        baseline = RuntimeCosts(cal,fit,'baseline')
        kite = RuntimeCosts(cal,fit,'kite')
        self.assertEqual(baseline.expert(1)[0],kite.expert(1)[0])
        self.assertGreater(baseline.expert(1)[1],1)
        self.assertLess(kite.expert(1)[1],1)
        self.assertEqual(kite.expert(64)[1],float('inf'))
        self.assertEqual(kite.kv(2048)[1],float('inf'))

    def test_runtime_controller_transform_and_phase(self):
        for mode in ('baseline','kite'):
            controller = Controller(510*1024**2,current_weight=.65 if mode=='kite' else 0)
            controller.before_batch(1,0)
            controller.after_batch([((0,1),1)],[])
            controller.calibration = self.calibration()
            controller.finish_calibration(lambda cal,fit:RuntimeCosts(cal,fit,mode))
            old = controller.proposal
            decision = controller.before_batch(0,1)
            if mode=='baseline':
                self.assertEqual(decision['candidate']['expert_bytes'],old.expert_bytes)
            self.assertFalse(decision['physical_split_applied'])

    def test_selection_requires_measured_total_and_keeps_all(self):
        rows = [dict(status='passed',budget_mib=b,mode=m,workload_total_ms=v)
                for b,times in [(512,(100,110)),(1024,(100,90)),(2048,(100,80))]
                for m,v in zip(('baseline','kite'),times)]
        selected = choose_selection(rows)
        self.assertEqual(selected['selected']['budget_mib'],2048)
        self.assertEqual(len(selected['all_budgets']),3)
        for row in rows:
            row['workload_total_ms']=100
        self.assertEqual(choose_selection(rows)['status'],'no_positive_result')
        self.assertEqual(len(protocol_hash()),64)
        self.assertEqual(PROTOCOL['cpu_fp32_cache_gib'],8)


if __name__=='__main__':
    unittest.main()
