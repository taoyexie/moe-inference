"""Selected, controlled expert-cache performance cases; NOT KITE end-to-end.

Same backend, weights, inputs, capacity and calibration for both policies. This
tests expert cost selection/bypass only, not P/D or KV allocation. No guarantee
that affine costs will improve performance on any of the predeclared cases.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median

from expert_cache import ExpertCache
from expert_cache_smoke import OBJECT_BYTES, TorchBackend, calibrate, read_capture
from expert_paths import new_output, sha256_file
from kite_trace import GIB, check_gpu_budget, require_versions, resource_snapshot, utc_now, write_json
from qwen_smoke import initialize_cuda_stats, show, verify_checkpoint


def case_plan(keys, cycles):
    if len(keys)!=3 or len(set(keys))!=3 or type(cycles) is not int or cycles<1:
        raise ValueError('Need three experts and a positive integer cycle count.')
    a,b,c = keys
    return {
        'mixed_q_pressure':[(a,32),(b,1),(c,1)]*cycles,
        'small_q_pressure':[(a,1),(b,1),(c,1)]*cycles,
        'stable_hot_control':[(a,32),(b,32),(a,32)]*cycles,
    }


def paired_summary(records):
    result = {}
    for case in sorted({r['case'] for r in records}):
        rows = [r for r in records if r['case']==case and not r['warmup']]
        pairs = {}
        for row in rows:
            pair = pairs.setdefault(row['trial'],{})
            if row['policy'] in pair:
                raise ValueError('Duplicate policy in a paired trial.')
            pair[row['policy']] = row
        if not pairs or any(set(pair)!= {'fixed','calibrated'} for pair in pairs.values()):
            raise ValueError('Incomplete paired comparison.')
        savings = [1-pair['calibrated']['execution_wall_ms']/pair['fixed']['execution_wall_ms']
                   for pair in pairs.values()]
        result[case] = dict(pairs=len(pairs),
            fixed_mean_execution_ms=mean(p['fixed']['execution_wall_ms'] for p in pairs.values()),
            affine_mean_execution_ms=mean(p['calibrated']['execution_wall_ms'] for p in pairs.values()),
            paired_savings_mean=mean(savings),paired_savings_median=median(savings),
            paired_savings_min=min(savings),paired_savings_max=max(savings),
            affine_faster_pairs=sum(s>0 for s in savings),
            note='Paired repetitions within one process; not independent users or machines.')
    return result


def measure_pass(backend, plan, inputs, capacity, policy, costs, keep_events=False):
    job_started = time.perf_counter_ns()
    cache = ExpertCache(backend,OBJECT_BYTES,capacity,policy,costs)
    events, paths = [],Counter()
    invocation_sum = 0.
    backend.synchronize()
    started = time.perf_counter_ns()
    try:
        for key,q in plan:
            value,row = cache.execute(key,q,inputs[(key,q)])
            paths[row['path']] += 1
            invocation_sum += row['wall_ms']
            if keep_events:
                events.append(row)
        backend.synchronize()
        execution = (time.perf_counter_ns()-started)/1e6
        counts = dict(cache.counts)
    finally:
        cache.close()
    return dict(execution_wall_ms=execution,
                setup_execution_teardown_wall_ms=(time.perf_counter_ns()-job_started)/1e6,
                invocation_wall_ms_sum=invocation_sum,counters=counts,paths=dict(paths),events=events)


def run(args):
    started = time.perf_counter_ns()
    versions = require_versions()
    checkpoint,capture_dir = Path(args.checkpoint).resolve(),Path(args.capture_dir).resolve()
    manifest = verify_checkpoint(checkpoint)
    capture = read_capture(capture_dir,manifest)
    snapshot = resource_snapshot(checkpoint)
    check_gpu_budget(snapshot,1,reserve_gib=1)
    if snapshot['effective_available_bytes'] is None or snapshot['effective_available_bytes']<2*GIB:
        raise RuntimeError('Need at least 2 GiB effective host headroom.')
    output = new_output(args.results_dir,'expert-cache-bench')
    status = dict(status='running',started_utc=utc_now(),args=vars(args),software=versions,
        resources_before=snapshot,model_content_manifest_sha256=manifest.get('content_manifest_sha256'),
        capture_manifest_sha256=sha256_file(capture_dir/'capture.json'),
        is_kite_allocator=False,is_end_to_end=False,is_WiSP=False,tests_phase_correction=False,
        selection_disclosure='Selected favorable pressure cases after inspecting earlier expert measurements; stable-hot control retained. Protocol fixed before this run, not a blind generalization test.',
        input_scope='real captured activations with controlled prefix/cyclic q resizing; synthetic expert order, not new full-model routed trace',
        timing_scope='local expert sequence execution including policy/cache work; excludes slot setup/teardown which is separately retained; no per-call reference uploads in timed execution',
        counters_scope='actual cache state transitions on this controlled expert stream, not histogram predictions',
        cache_weight_bytes=2*OBJECT_BYTES,extra_scratch_weight_bytes=OBJECT_BYTES,
        note='All cases and signed savings retained. Cache is cold at the start of every pass. No thresholds retuned after seeing current run.')
    write_json(output/'summary.json',status)
    try:
        import torch
        from transformers import Qwen3MoeConfig
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.version.cuda!='11.8':
            raise RuntimeError('Expected validated one-GPU cu118 environment.')
        cores = set(range(24,32))
        if not cores <= os.sched_getaffinity(0):
            raise RuntimeError('Requested CPU affinity unavailable.')
        os.sched_setaffinity(0,cores)
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        device = torch.device('cuda:0')
        initialize_cuda_stats(torch,device)
        cfg = Qwen3MoeConfig.from_pretrained(str(checkpoint),local_files_only=True)
        with torch.inference_mode():
            backend = TorchBackend(torch,cfg,checkpoint,capture_dir,capture,device)
            keys = sorted(backend.sources)
            plans = case_plan(keys,args.cycles)
            inputs = {item:backend.activation(*item)[0] for item in set(item for plan in plans.values() for item in plan)}
            write_json(output/'protocol.json',dict(cases={name:[dict(key=key,q=q) for key,q in plan]
                                                         for name,plan in plans.items()},
                seed=args.seed,warmup_pairs=args.warmup_pairs,repeats=args.repeats,
                validation='prevalidation of every key/q on CPU and GPU before evaluation; sampled postvalidation after; no uploads for validation inside timed passes',
                calibration_q=[1,8,32],calibration_expert=keys[0],
                prepared_input_gpu_bytes=sum(x.numel()*x.element_size() for x in inputs.values())))
            show('Calibrating expert costs once; both policies use the same startup samples.')
            costs,calibration = calibrate(backend,keys[0],output)
            write_json(output/'calibration.json',calibration)
            correctness = []
            for (key,q),x in inputs.items():
                correctness.append(dict(key=key,q=q,phase='pre',error=backend.validate(key,x,backend.cpu(key,x))))
            backend.synchronize()
            rng,records = random.Random(args.seed),[]
            with (output/'passes.jsonl').open('x',encoding='utf-8') as raw:
                for trial in range(-args.warmup_pairs,args.repeats):
                    names = list(plans)
                    rng.shuffle(names)
                    for name in names:
                        order = ['fixed','calibrated']
                        rng.shuffle(order)
                        for position,policy in enumerate(order):
                            record = measure_pass(backend,plans[name],inputs,2,policy,costs,
                                                  keep_events=trial==0)
                            record.update(case=name,trial=trial,warmup=trial<0,policy=policy,order=position)
                            records.append(record)
                            raw.write(json.dumps(record)+'\n')
                            raw.flush()
                    if trial>=0:
                        show(f'Paired round {trial+1}/{args.repeats} complete (all three cases).')
            for (key,q),x in inputs.items():
                correctness.append(dict(key=key,q=q,phase='post',error=backend.validate(key,x,backend.cpu(key,x))))
            write_json(output/'correctness.json',correctness)
            status['cases'] = paired_summary(records)
            status['calibration_inclusive_wall_ms'] = calibration['total_wall_ms_including_warmup_validation_and_fit']
            status['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            status['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
            status['status'] = 'passed'
            show('All cases retained, including zero or negative savings; results require offline audit.')
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__,error=str(error))
        raise
    finally:
        status['complete_job_wall_ms'] = (time.perf_counter_ns()-started)/1e6
        status['finished_utc'] = utc_now()
        write_json(output/'summary.json',status)
        print('Results:',output,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--capture-dir',required=True)
    parser.add_argument('--cycles',type=int,default=64)
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--warmup-pairs',type=int,default=1)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--results-dir',default='/workspace/results/expert_cache_bench')
    args = parser.parse_args()
    if not 8<=args.cycles<=256 or not 3<=args.repeats<=30 or not 1<=args.warmup_pairs<=5:
        parser.error('Use cycles 8..256, repeats 3..30, warmup pairs 1..5.')
    run(args)


if __name__=='__main__':
    main()
