#!/usr/bin/env python3
"""At most two actual E2E pairs: short repeated-prompt startup bursts, no probes."""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from expert_paths import new_output
from kite_advantage import OPTIONS as SHARED_OPTIONS,performance_validation_record
from kite_e2e import ResourceMonitor,execute_task,implementation_hash as backend_hash
from kite_e2e_core import EXPERT_BYTES,KV_BYTES,FRAME_BYTES
from kite_controller import candidates
from kite_readiness import readiness,require_ready
from kite_trace import GIB,write_json,utc_now
from multi_trace import MARKER,build_ids
from qwen_smoke import initialize_cuda_stats,show
from kite_e2e_torch import load_engine


CASES = [dict(name='repeat-20',budget_mib=32768,search_prompt='请用一句话解释什么是缓存。',
              confirm_prompt='请用一句话解释什么是内存。'),
         dict(name='repeat-short',budget_mib=28672,search_prompt='解释缓存。',confirm_prompt='解释内存。')]
PROTOCOL = dict(version='kite-short-prefill-burst-v1',model='Qwen/Qwen3-30B-A3B',cases=CASES,main_rounds=16,
    initial_request=dict(input_tokens=512,output_tokens=1,release_step=0),
    repeat_requests=dict(count=15,output_tokens=1,release_steps=list(range(1,16))),
    phase_scope='prefill-only one-token completion bursts; not decode-heavy or natural online traffic',
    current_weight=.65,refresh_batches=16,fractions=[.1,.25,.5,.75,.9],
    prefill_expert_share=.8,decode_expert_share=.2,cpu_fp32_cache_gib=8,
    generation='genuine model/router, greedy EOS-suppressed single output per request; no response/prefix cache or forced experts',
    timing='full actual workload including startup calibration, pool init, online decisions, transfers and all requests; loading separate',
    numerical_policy='performance_only; no new precision probes',
    stopping='at most two predeclared pairs; stop after first pair if actual E2E speedup >= 1.10',
    order='seed17/41 baseline-first, seed23 KITE-first; alternate for second case',
    hypothesis='repeated prefill expert working set may fit current-signal split but not stale initial split before first 16-batch history refresh',
    previous_evidence='official Qwen real 20-token prefill trace observed 2565 unique experts; not guaranteed identical under new executor')
OPTIONS = dict(SHARED_OPTIONS,require_dynamic_split=False)


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def implementation_hash():
    folder = Path(__file__).resolve().parent
    records = dict(backend=backend_hash(),**{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
        for name in ('kite_burst_search.py','kite_advantage.py','kite_readiness.py')})
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()


def mode_order(case_index,seed):
    return ['kite','baseline'] if (case_index+(seed==23))%2 else ['baseline','kite']


def workload(tokenizer,case,confirmation=False):
    text = case['confirm_prompt'] if confirmation else case['search_prompt']
    hot = tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
        add_generation_prompt=True,enable_thinking=False)
    if not 1<=len(hot)<=64:
        raise RuntimeError('Unexpected short prompt length.')
    if not confirmation and case['name']=='repeat-20' and len(hot)!=20:
        raise RuntimeError('The known real prompt no longer has 20 tokens; do not silently replace it.')
    rendered = tokenizer.apply_chat_template([dict(role='user',content=MARKER)],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    if rendered.count(MARKER)!=1:
        raise RuntimeError('Unknown chat template boundary.')
    prefix,suffix = rendered.split(MARKER)
    encode = lambda value:tokenizer.encode(value,add_special_tokens=False)
    initial = build_ids(encode(prefix),encode(text),encode(suffix),512)
    specs = [dict(request_id='initial-512',release_step=0,input_tokens=512,input_ids=initial,
        output_tokens=1,source_text=text,construction='repeated body between genuine chat boundaries')]
    specs += [dict(request_id=f'repeat-{i:02d}',release_step=i,input_tokens=len(hot),input_ids=list(hot),
        output_tokens=1,source_text=text,construction='same genuine short prompt; independent request and KV')
        for i in range(1,16)]
    return specs


def structural_screen(case,specs):
    budget = case['budget_mib']*1024**2//FRAME_BYTES*FRAME_BYTES
    choices = candidates(budget,EXPERT_BYTES,KV_BYTES,(.1,.25,.5,.75,.9))
    initial = min(choices,key=lambda c:abs(c.expert_bytes-.5*budget))
    corrected = min(choices,key=lambda c:abs(c.expert_bytes-(.35*initial.expert_bytes+.65*.8*budget)))
    return dict(scope='capacity arithmetic only, not a new trace or measured speedup',
        request_count=len(specs),batch_count=16,output_tokens=16,
        input_tokens=sum(s['input_tokens'] for s in specs),max_active=1,
        max_host_kv_bytes=4*KV_BYTES,if_history_half_baseline_experts=initial.expert_objects,
        if_history_half_kite_prefill_experts=corrected.expert_objects,usable_bytes=budget)


def selection(tasks):
    rows = []
    for case in CASES:
        pair = {t['mode']:t for t in tasks if t['case_name']==case['name']}
        if not pair:
            continue
        if set(pair)!={'baseline','kite'} or any(t['status']!='passed' for t in pair.values()):
            raise RuntimeError('Both actual workloads must finish before selecting a scene.')
        a,b = pair['baseline']['workload_total_ms'],pair['kite']['workload_total_ms']
        rows.append(dict(case_name=case['name'],budget_mib=case['budget_mib'],baseline_ms=a,kite_ms=b,speedup=a/b))
    best = max(rows,key=lambda r:r['speedup']) if rows else None
    return dict(status='selected' if best and best['speedup']>1 else 'no_positive_result',
        selected=best if best and best['speedup']>1 else None,all_cases=rows,
        metric='actual E2E including each task startup, not modeled miss cost')


def run(args):
    started = time.perf_counter_ns()
    output = new_output(args.results_dir,'prefill-burst-'+args.command)
    checkpoint = Path(args.checkpoint).resolve()
    ready = readiness(checkpoint,Path(__file__).resolve().parents[1])
    state = dict(status='running',stage='readiness',command=args.command,args=vars(args),
        protocol=PROTOCOL,protocol_sha256=protocol_hash(),implementation_sha256=implementation_hash(),
        runtime_options=OPTIONS,model_manifest_sha256=ready['checks'].get('checkpoint',{}).get('content_manifest_sha256'),
        resources_before=ready.get('resources'),tasks=[],is_WiSP=False)
    monitor = None
    write_json(output/'readiness.json',ready)
    try:
        require_ready(ready)
        active_cases = list(enumerate(CASES))
        if args.command=='confirm':
            frozen = json.loads(Path(args.selection).read_text(encoding='utf-8'))
            if (frozen['status']!='selected' or frozen['implementation_sha256']!=implementation_hash()
                    or frozen['protocol_sha256']!=protocol_hash()
                    or frozen['model_manifest_sha256']!=state['model_manifest_sha256']):
                raise RuntimeError('Confirmation requires a positive selection for these exact sources and model.')
            active_cases = [(i,c) for i,c in active_cases if c['name']==frozen['selected']['case_name']]
            if len(active_cases)!=1:
                raise RuntimeError('Unknown frozen case.')
            state['frozen_selection'] = frozen
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
        specifications = {case['name']:workload(tokenizer,case,args.command=='confirm') for _,case in active_cases}
        write_json(output/'workload.json',dict(protocol=PROTOCOL,split=args.command,cases=specifications))
        state['workload_sha256'] = hashlib.sha256((output/'workload.json').read_bytes()).hexdigest()
        write_json(output/'structural_screen.json',{case['name']:structural_screen(case,specifications[case['name']])
            for _,case in active_cases})
        write_json(output/'validation.json',performance_validation_record())
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.version.cuda!='11.8':
            raise RuntimeError('Expected the validated one-GPU cu118 installation.')
        os.sched_setaffinity(0,set(range(24,32)))
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        device = torch.device('cuda:0')
        initialize_cuda_stats(torch,device)
        monitor = ResourceMonitor(evidence_path=output/'gpu_monitor_startup.json')
        with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            state['stage'] = 'model_loading'
            write_json(output/'summary.json',state)
            show('Loading real Qwen once; next: actual short-prefill burst pairs, no precision probes.')
            engine,tokenizer,state['loading'] = load_engine(torch,checkpoint,device,8*GIB,
                ready['resources'],health_check=monitor.check)
            engine.pad_attention_keys = True
            engine.health_check = monitor.check
            for case_index,case in active_cases:
                for mode in mode_order(case_index,args.seed):
                    state['stage'] = case['name']+'-'+mode
                    write_json(output/'summary.json',state)
                    task = execute_task(engine,tokenizer,specifications[case['name']],case['budget_mib'],mode,
                        output/f'{case["name"]}-{case["budget_mib"]}-{mode}',monitor,
                        runtime_options=OPTIONS,timing_protocol=PROTOCOL)
                    # execute_task's own immutable performance.json remains byte-for-byte unchanged.
                    state['tasks'].append(dict(task,case_name=case['name']))
                    write_json(output/'summary.json',state)
                    show(f'{case["name"]} {mode}: actual E2E={task["workload_total_ms"]/1000:.3f}s (startup included).')
                chosen = selection(state['tasks'])
                row = chosen['all_cases'][-1]
                show(f'{case["name"]}: measured complete-strategy speedup={row["speedup"]:.3f}x.')
                if args.command=='search' and row['speedup']>=1.10:
                    state['early_stop'] = 'predeclared speedup >= 1.10 after a complete pair'
                    break
            if args.command=='search':
                chosen = selection(state['tasks'])
                chosen.update(protocol_sha256=protocol_hash(),implementation_sha256=implementation_hash(),
                    model_manifest_sha256=state['model_manifest_sha256'],search_run=str(output),exploratory=True)
                write_json(output/'selection.json',chosen)
                state['selection'] = chosen
            monitor.check()
            state.update(status='passed',stage='complete')
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
            error_type=type(error).__name__,error=str(error))
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        if monitor is not None:
            monitor.close()
            write_json(output/'resource_monitor.json',monitor.record())
            if monitor.error:
                state.update(status='failed',error_type='ResourceInterference',error=monitor.error)
        state['finished_utc'] = utc_now()
        state['job_wall_ms_including_loading_and_all_comparisons'] = (time.perf_counter_ns()-started)/1e6
        write_json(output/'summary.json',state)
        print('Results:',output,flush=True)
    if state['status']!='passed':
        raise RuntimeError(state.get('error','Run failed.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['search','confirm'])
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--selection')
    parser.add_argument('--seed',type=int,choices=[17,23,41],default=17)
    parser.add_argument('--results-dir',default='/workspace/results/kite_burst_search')
    args = parser.parse_args()
    if (args.command=='confirm')!=bool(args.selection):
        parser.error('--selection is required only for confirmation.')
    run(args)


if __name__=='__main__':
    main()
