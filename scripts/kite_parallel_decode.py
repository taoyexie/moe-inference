#!/usr/bin/env python3
"""Two bounded real E2E pairs at a trace-screened parallel-decode capacity cliff."""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from expert_paths import new_output
from kite_advantage import OPTIONS,SEARCH_TOPICS,CONFIRM_TOPICS,performance_validation_record
from kite_e2e import ResourceMonitor,execute_task,implementation_hash as backend_hash,new_requests
from kite_e2e_core import KV_BYTES,packed_layout,ready_batch
from kite_readiness import readiness,require_ready
from kite_trace import GIB,utc_now,write_json
from multi_trace import MARKER,build_ids
from qwen_smoke import initialize_cuda_stats,show
from kite_e2e_torch import load_engine


CASES = [dict(name='parallel-6g',budget_mib=6144),dict(name='parallel-7g',budget_mib=7168)]
PROTOCOL = dict(version='kite-parallel-decode-cliff-v1',model='Qwen/Qwen3-30B-A3B',cases=CASES,
    requests=dict(input_tokens=[8192,11264,11264,11264],release_steps=[0,0,0,0],output_tokens_each=16),
    current_weight=.65,refresh_batches=16,fractions=[.1,.25,.5,.75,.9],
    prefill_expert_share=.8,decode_expert_share=.2,cpu_fp32_cache_gib=8,
    timing='actual complete workload including pool initialization, workload-specific startup calibration, all prefill/decode, transfers, writes, expiry and online decisions; loading separate',
    scope='controlled simultaneous four-request short-output workload; one packed all-prefill batch followed by 15 four-request decode batches; not steady-state long generation or natural traffic; not original WiSP',
    stopping='exactly two complete pairs during search; no early stop or hidden additional budget scan',
    order='seed17/41 baseline-first for 6g and KITE-first for 7g; seed23 reversed',
    generation='real Qwen/router greedy autoregression, EOS suppressed, same fixed output counts, no teacher forcing or forced experts',
    selection='positive actual startup-inclusive E2E and main execution, plus fewer actual KV restore pages; rank eligible cases by actual E2E speedup',
    hypothesis='when history proposes half the pool, current decode correction raises KV capacity across the observed working-set cliff without useful Expert residency being lost; history proposal is never forced',
    numerical_policy='performance_only; no new precision probes',
    schedule='four requests arrive simultaneously, independent KV/positions and isolated attention, no chunked prefill')


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def implementation_hash():
    folder = Path(__file__).resolve().parent
    records = dict(backend=backend_hash(),**{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
        for name in ('kite_parallel_decode.py','kite_advantage.py','kite_readiness.py')})
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()


def mode_order(case_index,seed):
    return ['kite','baseline'] if (case_index+(seed==23))%2 else ['baseline','kite']


def workload(tokenizer,confirmation=False):
    template = tokenizer.apply_chat_template([dict(role='user',content=MARKER)],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    if template.count(MARKER)!=1:
        raise RuntimeError('Unknown chat template boundaries.')
    prefix,suffix = template.split(MARKER)
    encode = lambda text:tokenizer.encode(text,add_special_tokens=False)
    prefix,suffix = encode(prefix),encode(suffix)
    topics = CONFIRM_TOPICS if confirmation else SEARCH_TOPICS
    return [dict(request_id=f'parallel-{i}',release_step=0,input_tokens=length,output_tokens=16,
        topic_index=i,source_text=topics[i],construction='repeated body inside genuine chat boundaries',
        input_ids=build_ids(prefix,encode(topics[i]),suffix,length))
        for i,length in enumerate(PROTOCOL['requests']['input_tokens'])]


def structural_screen():
    specs = [dict(request_id=f'parallel-{i}',input_ids=[1]*length,release_step=0,output_tokens=16)
        for i,length in enumerate(PROTOCOL['requests']['input_tokens'])]
    requests = new_requests(specs)
    rows = []
    for step in range(16):
        active = ready_batch(requests,step)
        layout = packed_layout(active)
        q,k = sum(len(r.query) for r in active),sum(r['after'] for r in layout)
        padded = k+(-k)%8
        pages = sum((r['after']+127)//128 for r in layout)
        rows.append(dict(step=step,prefill=sum(r['phase']=='prefill' for r in layout),decode=sum(r['phase']=='decode' for r in layout),
            query_tokens=q,kv_pages=pages,kv_bytes=pages*KV_BYTES,padded_mask_bytes=2*q*padded,
            dense_kv_bytes=48*2*4*128*2*k))
        for req,row in zip(active,layout):
            req.kv_length = row['after']
            req.generated.append(0)
    return dict(scope='shape/page arithmetic, not GPU performance or a routing forecast',batch_count=16,
        request_count=4,output_tokens=64,input_tokens=sum(map(len,[s['input_ids'] for s in specs])),
        max_active=4,max_kv_pages=max(r['kv_pages'] for r in rows),max_kv_bytes=max(r['kv_bytes'] for r in rows),
        max_dense_kv_bytes=max(r['dense_kv_bytes'] for r in rows),max_padded_mask_bytes=max(r['padded_mask_bytes'] for r in rows),
        batches=rows)


def selection(tasks):
    rows = []
    for case in CASES:
        records = [t for t in tasks if t['case_name']==case['name']]
        pair = {t['mode']:t for t in records}
        if not pair:
            continue
        if (len(records)!=2 or set(pair)!={'baseline','kite'}
                or any(t['status']!='passed' or t['budget_mib']!=case['budget_mib'] for t in records)):
            raise RuntimeError('Selection requires both completed actual policies.')
        a,b = pair['baseline'],pair['kite']
        e2e = a['workload_total_ms']/b['workload_total_ms']
        main = a['execution_inclusive_ms']/b['execution_inclusive_ms']
        avoided = a['kv_counts'].get('kv_restore_pages',0)-b['kv_counts'].get('kv_restore_pages',0)
        rows.append(dict(case_name=case['name'],budget_mib=case['budget_mib'],baseline_ms=a['workload_total_ms'],
            kite_ms=b['workload_total_ms'],speedup=e2e,main_speedup=main,actual_kv_restore_pages_avoided=avoided,
            eligible=e2e>1 and main>1 and avoided>0))
    eligible = [r for r in rows if r['eligible']]
    return dict(status='selected' if eligible else 'no_supported_positive_result',
        selected=max(eligible,key=lambda r:r['speedup']) if eligible else None,all_cases=rows,
        metric=PROTOCOL['selection'])


def run(args):
    started = time.perf_counter_ns()
    output = new_output(args.results_dir,'parallel-decode-'+args.command)
    checkpoint = Path(args.checkpoint).resolve()
    ready = readiness(checkpoint,Path(__file__).resolve().parents[1])
    state = dict(status='running',stage='readiness',command=args.command,args=vars(args),protocol=PROTOCOL,
        protocol_sha256=protocol_hash(),implementation_sha256=implementation_hash(),runtime_options=OPTIONS,
        model_manifest_sha256=ready['checks'].get('checkpoint',{}).get('content_manifest_sha256'),
        resources_before=ready.get('resources'),tasks=[],is_WiSP=False)
    monitor = None
    write_json(output/'readiness.json',ready)
    try:
        require_ready(ready)
        active_cases = list(enumerate(CASES))
        if args.command=='confirm':
            frozen = json.loads(Path(args.selection).read_text(encoding='utf-8'))
            if (frozen['status']!='selected' or frozen['protocol_sha256']!=protocol_hash()
                    or frozen['implementation_sha256']!=implementation_hash()
                    or frozen['model_manifest_sha256']!=state['model_manifest_sha256']):
                raise RuntimeError('Confirmation requires a supported frozen selection for this version/model.')
            prior = json.loads((Path(frozen['search_run'])/'summary.json').read_text(encoding='utf-8'))
            if prior['status']!='passed' or prior.get('selection')!=frozen:
                raise RuntimeError('Frozen search did not finish successfully, including its resource monitor.')
            active_cases = [(i,c) for i,c in active_cases if c['name']==frozen['selected']['case_name']]
            if len(active_cases)!=1:
                raise RuntimeError('Unknown selected case.')
            state['frozen_selection'] = frozen
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
        specs = workload(tokenizer,args.command=='confirm')
        shape = structural_screen()
        if shape['max_kv_bytes']>OPTIONS['host_kv_limit_bytes'] or shape['max_padded_mask_bytes']>4*GIB:
            raise RuntimeError('Workload shape exceeds the declared host/mask guard.')
        write_json(output/'workload.json',dict(protocol=PROTOCOL,split=args.command,requests=specs))
        state['workload_sha256'] = hashlib.sha256((output/'workload.json').read_bytes()).hexdigest()
        write_json(output/'structural_screen.json',shape)
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
            show('Loading Qwen once for exactly two parallel-decode E2E pairs, 6/7 GiB; no precision probes.')
            engine,tokenizer,state['loading'] = load_engine(torch,checkpoint,device,8*GIB,ready['resources'],health_check=monitor.check)
            engine.pad_attention_keys = True
            engine.health_check = monitor.check
            for case_index,case in active_cases:
                for mode in mode_order(case_index,args.seed):
                    state['stage'] = case['name']+'-'+mode
                    write_json(output/'summary.json',state)
                    task = execute_task(engine,tokenizer,specs,case['budget_mib'],mode,
                        output/f'{case["name"]}-{case["budget_mib"]}-{mode}',monitor,runtime_options=OPTIONS,timing_protocol=PROTOCOL)
                    state['tasks'].append(dict(task,case_name=case['name']))
                    write_json(output/'summary.json',state)
                    show(f'{case["name"]} {mode}: E2E={task["workload_total_ms"]/1000:.3f}s; main={task["execution_inclusive_ms"]/1000:.3f}s; KV restores={task["kv_counts"].get("kv_restore_pages",0)}.')
                row = selection(state['tasks'])['all_cases'][-1]
                show(f'{case["name"]}: actual E2E={row["speedup"]:.3f}x, main={row["main_speedup"]:.3f}x; mechanism-supported positive={row["eligible"]}.')
            if args.command=='search':
                chosen = selection(state['tasks'])
                chosen.update(protocol_sha256=protocol_hash(),implementation_sha256=implementation_hash(),
                    model_manifest_sha256=state['model_manifest_sha256'],search_run=str(output),exploratory=True)
                write_json(output/'selection.json',chosen)
                state['selection'] = chosen
            monitor.check()
            state.update(status='passed',stage='complete')
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error_type=type(error).__name__,error=str(error))
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
    parser.add_argument('command',choices=['check','search','confirm'])
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--selection')
    parser.add_argument('--seed',type=int,choices=[17,23,41],default=17)
    parser.add_argument('--results-dir',default='/workspace/results/kite_parallel_decode')
    args = parser.parse_args()
    if (args.command=='confirm')!=bool(args.selection):
        parser.error('--selection is required only for confirmation.')
    if args.command=='check':
        report = readiness(Path(args.checkpoint),Path(__file__).resolve().parents[1])
        print(json.dumps(report,ensure_ascii=False,indent=2))
        raise SystemExit(0 if report['status']=='ready' else 2 if report['status']=='blocked' else 1)
    run(args)


if __name__=='__main__':
    main()
