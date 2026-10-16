#!/usr/bin/env python3
"""Full Qwen controlled mixed-request E2E: fixed/history baseline vs KITE.

validate: one necessary full-model integration gate, not a performance claim.
search: exactly three predeclared budgets and two policies, preserve all results.
confirm: frozen selected budget, held-out handwritten topics, independent session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from expert_paths import error_metrics, new_output
from kite_e2e_core import Request, packed_layout
from kite_e2e_runtime import Runtime, execute_requests
from kite_e2e_torch import load_engine
from kite_trace import GIB, check_gpu_budget, read_optional, parse_meminfo, parse_gpu_pids, require_versions, resource_snapshot, run_readonly, utc_now, write_json
from multi_trace import MARKER, TOPICS, build_ids
from qwen_smoke import initialize_cuda_stats, show, verify_checkpoint
from workload_baseline import latency_summary, token_metrics

PROTOCOL = dict(version='kite-e2e-v1',model='Qwen/Qwen3-30B-A3B',
    background=dict(prompt_tokens=2048,output_tokens=32,release_step=0),
    bursts=dict(release_steps=[9,17,25],requests_each=3,prompt_tokens=512,output_tokens=8),
    budgets_mib=[512,1024,2048],current_weight=.65,refresh_batches=16,
    fractions=[.1,.25,.5,.75,.9],prefill_expert_share=.8,decode_expert_share=.2,
    expert_q=[1,8,32],kv_context=[128,512],cpu_fp32_cache_gib=8,
    generation='greedy; EOS suppressed for fixed prescribed length, no teacher forcing',
    timing='model-ready workload submission before pool init/calibration to completion including KV writes, expiry and online decisions',
    schedule='logical-round arrivals, maximum four active requests, no chunked prefill, real packed mixed forward')


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def implementation_hash():
    names = ['kite_e2e.py','kite_e2e_core.py','kite_e2e_torch.py','kite_e2e_runtime.py',
             'kite_controller.py','kite_trace.py','qwen_smoke.py','expert_paths.py',
             'multi_trace.py','workload_baseline.py']
    folder = Path(__file__).resolve().parent
    return hashlib.sha256(json.dumps({name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                                     for name in names},sort_keys=True).encode()).hexdigest()


def workload(tokenizer, split):
    template = tokenizer.apply_chat_template([{'role':'user','content':MARKER}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if template.count(MARKER)!=1:
        raise RuntimeError('Unknown chat template boundaries.')
    prefix,suffix = template.split(MARKER)
    encode = lambda text:tokenizer.encode(text,add_special_tokens=False)
    prefix,suffix = encode(prefix),encode(suffix)
    topics = [text for _,text in TOPICS] if split=='search' else [
        '科学研究通过提出问题、建立假设、设计实验和分析证据探索自然规律。测量需要重复和记录。请概括材料主题。',
        '数学学习包括代数运算、几何证明与概率分析。解决问题时要检查条件和推导过程。请概括材料主题。',
        '烹饪需要准备食材，控制火候，安排调味与烹煮顺序，并注意卫生。请概括材料主题。',
        '体育训练包括热身、耐力、力量和技术练习，合理安排休息有助于恢复。请概括材料主题。']
    specs = [dict(request_id='background',release_step=0,input_tokens=2048,output_tokens=32,topic_index=0)]
    specs += [dict(request_id=f'burst-{step}-{i}',release_step=step,input_tokens=512,output_tokens=8,
                   topic_index=1+(i+batch)%max(1,len(topics)-1))
              for batch,step in enumerate((9,17,25)) for i in range(3)]
    for spec in specs:
        text = topics[spec['topic_index']]
        spec['source_text'] = text
        spec['input_ids'] = build_ids(prefix,encode(text),suffix,spec['input_tokens'])
    return specs


def new_requests(specs):
    return [Request(s['request_id'],list(s['input_ids']),s['release_step'],s['output_tokens']) for s in specs]


def choose_selection(results):
    choices = []
    for budget in PROTOCOL['budgets_mib']:
        pair = {r['mode']:r for r in results if r['budget_mib']==budget and r['status']=='passed'}
        if set(pair)!={'baseline','kite'}:
            continue
        ratio = pair['baseline']['workload_total_ms']/pair['kite']['workload_total_ms']
        choices.append(dict(budget_mib=budget,speedup=ratio,
                            baseline_ms=pair['baseline']['workload_total_ms'],kite_ms=pair['kite']['workload_total_ms']))
    positives = [row for row in choices if row['speedup']>1]
    return dict(status='selected' if positives else 'no_positive_result',
                selected=max(positives,key=lambda r:r['speedup']) if positives else None,
                all_budgets=choices,selection_metric='measured workload wall time including startup calibration and initialization')


class ResourceMonitor:
    def __init__(self, evidence_path=None):
        self.stop_event = threading.Event()
        self.error,self.samples = None,[]
        self.last_pid_query = None
        self.startup = dict(time_utc=utc_now(),python_pid=os.getpid(),
            nspid_line=next((s for s in (read_optional('/proc/self/status') or '').splitlines()
                             if s.startswith('NSpid:')),None),
            identity_scope='Single visible GPU PID inferred to be self. Requires separate clean pre-CUDA evidence; Docker PID may differ. Not a reservation.')
        try:
            initial = self.query_pids()
            self.startup['gpu_pids'] = sorted(initial)
            if not initial:
                raise RuntimeError('GPU PID query is empty while our CUDA context is active; cannot verify isolation.')
            if len(initial)!=1:
                raise RuntimeError(f'Multiple GPU PIDs after our CUDA initialization: {sorted(initial)}. '
                                   'Another process may have entered; do not stop theirs.')
            self.startup['status'] = 'passed'
        except Exception as error:
            self.startup.update(status='failed',error=str(error))
            raise
        finally:
            self.startup['pid_query'] = self.last_pid_query
            if evidence_path is not None:
                write_json(evidence_path,self.startup)
        self.allowed = initial
        self.thread = threading.Thread(target=self.loop,daemon=True)
        self.thread.start()

    def query_pids(self):
        self.last_pid_query = run_readonly(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
        return parse_gpu_pids(self.last_pid_query)

    def observe_once(self):
        pids = self.query_pids()
        available = parse_meminfo(read_optional('/proc/meminfo')).get('MemAvailable')
        self.samples.append(dict(time_utc=utc_now(),gpu_pids=sorted(pids),host_available_bytes=available))
        if pids-self.allowed:
            raise RuntimeError(f'Another GPU process entered: {sorted(pids-self.allowed)}. '
                               'Stop our run, never stop theirs.')
        if pids!=self.allowed:
            raise RuntimeError('Our expected GPU PID is no longer visible; cannot continue verifying isolation.')
        if available is not None and available<2*GIB:
            raise RuntimeError('Host RAM is below 2 GiB available. Stop to protect the shared server.')

    def loop(self):
        while not self.stop_event.wait(1):
            try:
                self.observe_once()
            except Exception as error:
                self.error = str(error)
                return

    def check(self):
        if self.error:
            raise RuntimeError(self.error)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=12)

    def record(self):
        return dict(samples=self.samples,error=self.error,allowed_gpu_pids=sorted(self.allowed),
                    poll_seconds=1,startup=self.startup,last_pid_query=self.last_pid_query)


def validate(engine, specs):
    """Compare genuine packed P+D against serial full-prefix reference forwards."""
    from transformers import DynamicCache
    torch = engine.torch
    runtime = Runtime(engine,32*1024**2,'baseline')
    checks = []
    try:
        for path,kv in [('upload','restore'),('upload','replay'),('cpu','restore')]:
            engine.pool.clear()
            engine.force_path = path
            engine.validate_cpu_outputs = path=='cpu'
            d = Request(f'validation-D-{path}-{kv}',specs[0]['input_ids'][:8],0,3)
            p = Request(f'validation-P-{path}-{kv}',specs[1]['input_ids'][:12],0,2)
            first = runtime.forward([d],observe=False)
            d.generated.append(int(first[0].argmax().item()))
            # Reference uses standard DynamicCache and no packed attention adapter.
            expected = []
            for req in (d,p):
                ids = torch.tensor([req.prompt_ids+req.generated],dtype=torch.long,device=engine.device)
                reference = engine.model(input_ids=ids,past_key_values=DynamicCache(),use_cache=True,
                    return_dict=True,output_router_logits=False,logits_to_keep=1)
                expected.append(reference.logits[0,-1].clone())
                del reference
            # Force actual recovery, even when a page happens to be resident.
            for key in list(engine.pool.entries['kv']):
                engine.pool.remove('kv',key)
            observed = runtime.forward([d,p],observe=False,force_kv=kv)
            error = error_metrics(torch,observed,torch.stack(expected))
            if error['relative_l2']>.03 or error['normalized_max_abs']>.05:
                raise RuntimeError(f'Packed/serial logits check failed ({path},{kv}): {error}')
            checks.append(dict(expert_path=path,kv_path=kv,logits_error=error,
                               argmax_equal=bool(torch.equal(observed.argmax(-1),torch.stack(expected).argmax(-1)))))
            runtime.expire(d)
            runtime.expire(p)
        return dict(status='passed',checks=checks,scope='short complete-model packed P+D vs serial reference; not performance',
                    counters=dict(runtime.counters),cpu_experts=dict(engine.cpu_validation))
    finally:
        engine.force_path = None
        engine.validate_cpu_outputs = False
        engine.synchronize()
        engine.pool.clear()
        engine.capture = None


def execute_task(engine, tokenizer, specs, budget, mode, output, monitor, runtime_options=None, timing_protocol=None):
    output.mkdir(exist_ok=False)
    status = dict(status='running',mode=mode,budget_mib=budget,is_end_to_end=True,
                  is_full_model=True,actual_kite_enabled=mode=='kite',started_utc=utc_now())
    write_json(output/'performance.json',status)
    requests = new_requests(specs)
    runtime = None
    engine.synchronize()
    reset_started = time.perf_counter_ns()
    engine.clear_between_tasks()
    status['harness_reset_ms_excluded'] = (time.perf_counter_ns()-reset_started)/1e6
    task_started = time.perf_counter_ns()
    try:
        runtime = Runtime(engine,budget*1024**2,mode,options=runtime_options)
        initialized = time.perf_counter_ns()
        engine.torch.cuda.reset_peak_memory_stats(engine.device)
        show(f'{mode} budget={budget} MiB: workload-start calibration (included in E2E).')
        calibration = runtime.calibrate(specs[0]['input_ids'])
        calibrated = time.perf_counter_ns()
        monitor.check()
        show(f'{mode} budget={budget} MiB: full-model mixed-request generation.')
        rounds = execute_requests(runtime,requests,task_started)
        engine.synchronize()
        ended = time.perf_counter_ns()
        monitor.check()
        if engine.torch.cuda.max_memory_allocated(engine.device)>engine.torch.cuda.get_device_properties(engine.device).total_memory-4*GIB:
            raise RuntimeError('GPU peak violated the declared 4 GiB safety reserve.')
        rows = []
        for req in requests:
            row = dict(request_id=req.request_id,input_tokens=len(req.prompt_ids),release_step=req.release_step,
                       final_kv_length=req.kv_length,
                       submitted_task_offset_ms=(req.submitted_ns-task_started)/1e6,
                       token_task_offsets_ms=[(t-task_started)/1e6 for t in req.token_ticks],
                       generated_ids=req.generated,generated_text=tokenizer.decode(req.generated,skip_special_tokens=True),
                       **token_metrics(req.submitted_ns,req.token_ticks))
            rows.append(row)
        status.update(status='passed',workload_total_ms=(ended-task_started)/1e6,
            task_clock_ns=dict(start=task_started,initialized=initialized,calibrated=calibrated,finished=ended),
            initialization_ms=(initialized-task_started)/1e6,
            startup_inclusive_ms=(calibrated-initialized)/1e6,
            execution_inclusive_ms=(ended-calibrated)/1e6,calibration_ms=calibration['wall_ms'],
            batch_count=rounds,request_count=len(rows),output_tokens=sum(r['output_tokens'] for r in rows),
            request_metrics=latency_summary(rows),expert_counts=dict(engine.counts),kv_counts=dict(runtime.counters),
            host_kv_peak_bytes=runtime.host_kv_peak,cpu_fp32_peak_bytes=engine.cpu_cache_peak,
            packed_workspace_peak_bytes=runtime.workspace_peak,
            replay_workspace_payload_peak_bytes=engine.replay_workspace_peak,
            cuda_peak_allocated_bytes=engine.torch.cuda.max_memory_allocated(engine.device),
            cuda_peak_reserved_bytes=engine.torch.cuda.max_memory_reserved(engine.device),
            pool=engine.pool.snapshot(),history_reference_only=True,
            timing_scope=(timing_protocol or PROTOCOL)['timing'])
        write_json(output/'calibration.json',calibration)
        write_json(output/'requests.json',rows)
        with (output/'batches.jsonl').open('x',encoding='utf-8') as stream:
            for row in runtime.batch_records:
                stream.write(json.dumps(row)+'\n')
        write_json(output/'recoveries.json',runtime.recoveries)
        return status
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__,error=str(error))
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        if status['status']!='passed' and runtime is not None:
            write_json(output/'partial_calibration.json',dict(rows=runtime.calibration_rows,
                checks=getattr(runtime,'calibration_checks',[]),cpu_cold_setup_rows=runtime.cpu_cold_setup_rows))
            write_json(output/'partial_batches.json',runtime.batch_records)
            write_json(output/'partial_recoveries.json',runtime.recoveries)
            write_json(output/'partial_requests.json',[dict(request_id=r.request_id,generated_ids=r.generated,
                kv_length=r.kv_length,token_ticks_ns=r.token_ticks,submitted_ns=r.submitted_ns) for r in requests])
        status['finished_utc'] = utc_now()
        write_json(output/'performance.json',status)


def run(args):
    job_started = time.perf_counter_ns()
    versions = require_versions()
    checkpoint = Path(args.checkpoint).resolve()
    manifest = verify_checkpoint(checkpoint)
    snapshot = resource_snapshot(checkpoint)
    check_gpu_budget(snapshot,8,reserve_gib=4)
    selection = None
    if args.command=='confirm':
        selection = json.loads(Path(args.selection).read_text(encoding='utf-8'))
        if (selection['status']!='selected' or selection['protocol_sha256']!=protocol_hash()
                or selection['implementation_sha256']!=implementation_hash()
                or selection['model_manifest_sha256']!=manifest.get('content_manifest_sha256')):
            raise RuntimeError('Confirmation requires a positive frozen search selection for this exact model/protocol.')
        if selection['selected']['budget_mib'] not in PROTOCOL['budgets_mib']:
            raise RuntimeError('Frozen selection has an undeclared budget.')
    output = new_output(args.results_dir,args.command)
    status = dict(status='running',command=args.command,args=vars(args),software=versions,
        model_manifest_sha256=manifest.get('content_manifest_sha256'),resources_before=snapshot,
        protocol=PROTOCOL,protocol_sha256=protocol_hash(),is_WiSP=False,
        implementation_sha256=implementation_hash(),
        selection_disclosure='Controlled advantage-oriented workload; exploratory budget selection separated from held-out-topic confirmation.',
        scope='full Qwen autoregressive generation with genuine packed P+D and physical Expert/KV pool; no HTTP or natural-arrival claim',tasks=[])
    if selection is not None:
        status['frozen_selection'] = selection
    write_json(output/'summary.json',status)
    monitor = None
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.version.cuda!='11.8':
            raise RuntimeError('Expected the validated one-GPU cu118 installation.')
        cores = set(range(24,32))
        if not cores<=os.sched_getaffinity(0):
            raise RuntimeError('Required GPU-near CPU affinity unavailable.')
        os.sched_setaffinity(0,cores)
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        device = torch.device('cuda:0')
        initialize_cuda_stats(torch,device)
        show('Loading complete Qwen on CPU, materializing expert masters; fixed weights move to GPU. No old layer offload.')
        with torch.inference_mode():
            engine,tokenizer,loading = load_engine(torch,checkpoint,device,8*GIB,snapshot)
            status['loading'] = loading
            monitor = ResourceMonitor()
            engine.health_check = monitor.check
            specs = workload(tokenizer,'confirmation' if args.command=='confirm' else 'search')
            write_json(output/'workload.json',dict(requests=specs,protocol=PROTOCOL,
                       split='confirmation' if args.command=='confirm' else 'search'))
            status['workload_sha256'] = hashlib.sha256((output/'workload.json').read_bytes()).hexdigest()
            if args.command=='confirm':
                search_root = Path(selection['search_run'])
                prior = json.loads((search_root/'summary.json').read_text(encoding='utf-8'))
                validation = json.loads((search_root/'validation.json').read_text(encoding='utf-8'))
                if (prior['status']!='passed' or prior['implementation_sha256']!=implementation_hash()
                        or validation['status']!='passed'):
                    raise RuntimeError('Frozen search has no passed integration check for these exact sources.')
                validation = {**validation,'reused_from':str(search_root)}
            else:
                show('Running necessary short full-model packed/serial and recovery checks.')
                validation = validate(engine,specs)
            write_json(output/'validation.json',validation)
            status['validation_status'] = validation['status']
            if args.command!='validate':
                budgets = list(PROTOCOL['budgets_mib']) if args.command=='search' else [selection['selected']['budget_mib']]
                rng = random.Random(args.seed)
                rng.shuffle(budgets)
                for budget in budgets:
                    modes = ['baseline','kite']
                    rng.shuffle(modes)
                    for mode in modes:
                        task = execute_task(engine,tokenizer,specs,budget,mode,output/f'{budget}-{mode}',monitor)
                        status['tasks'].append(task)
                        write_json(output/'summary.json',status)
                        show(f'{mode} {budget} MiB: E2E={task["workload_total_ms"]/1000:.3f}s (startup included).')
                if args.command=='search':
                    selected = choose_selection(status['tasks'])
                    selected.update(protocol_sha256=protocol_hash(),model_manifest_sha256=manifest.get('content_manifest_sha256'),
                                    implementation_sha256=implementation_hash(),
                                    search_run=str(output),exploratory=True)
                    write_json(output/'selection.json',selected)
                    status['selection'] = selected
            monitor.check()
            status['status'] = 'passed'
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__,error=str(error))
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        if monitor is not None:
            monitor.close()
            write_json(output/'resource_monitor.json',dict(samples=monitor.samples,error=monitor.error,
                       allowed_gpu_pids=sorted(monitor.allowed),poll_seconds=1))
            if monitor.error and status['status']=='passed':
                status.update(status='failed',error_type='ResourceInterference',error=monitor.error)
        status['finished_utc'] = utc_now()
        status['job_wall_ms_including_loading_validation_and_all_comparisons'] = (time.perf_counter_ns()-job_started)/1e6
        write_json(output/'summary.json',status)
        print('Results:',output,flush=True)
    if status['status']!='passed':
        raise RuntimeError(status.get('error','E2E run failed. See retained records.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['validate','search','confirm'])
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--selection')
    parser.add_argument('--seed',type=int,choices=[17,23,41],default=17)
    parser.add_argument('--results-dir',default='/workspace/results/kite_e2e')
    args = parser.parse_args()
    if (args.command=='confirm') != bool(args.selection):
        parser.error('--selection is required only for confirmation.')
    run(args)


if __name__=='__main__':
    main()
