#!/usr/bin/env python3
"""Bounded second exploration: long-context KV shocks, 8 GiB, two actual policies.

No measured speedup is assumed. A fixed schedule places workload changes just
after a 16-batch history refresh. Both policies see the same declared arrivals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path

from expert_paths import new_output
from kite_e2e import ResourceMonitor,execute_task,implementation_hash as backend_hash,new_requests,validate
from kite_e2e_core import KV_BYTES,packed_layout,ready_batch
from kite_e2e_torch import load_engine
from kite_readiness import readiness,require_ready
from kite_trace import GIB,utc_now,write_json
from multi_trace import MARKER,build_ids
from qwen_smoke import initialize_cuda_stats,show


PROTOCOL = dict(version='kite-kv-shock-v1',model='Qwen/Qwen3-30B-A3B',budgets_mib=[8192],
    background=dict(prompt_tokens=8192,output_tokens=104,release_step=0),
    bursts=dict(release_steps=[17,49,81],requests_each=3,prompt_tokens=11264,output_tokens=16),
    current_weight=.65,refresh_batches=16,fractions=[.1,.25,.5,.75,.9],
    prefill_expert_share=.8,decode_expert_share=.2,cpu_fp32_cache_gib=8,host_kv_limit_gib=5,
    expert_q=[1,8,32],kv_context=[128,512],expert_repeats=5,cpu_cold_setup_repeats=5,
    cost_estimation='per-x medians followed by equal-weight group mean or affine OLS; all raw samples retained',
    cpu_cold_setup='max of five distinct routed-expert formal materialization times, after one warmup; add unless FP32 copy actually exists',
    attention='EFFICIENT_ATTENTION only for both policies and references; fail instead of math fallback; masked K/V storage padded to 8 columns',
    generation='greedy; EOS suppressed, fixed output lengths, no teacher forcing',
    timing='model-ready workload submission before pool init/calibration to completion including KV writes, expiry and online decisions',
    scope='controlled advantage-oriented second exploration, not natural traffic; not original WiSP',
    hypothesis='KV working set jumps after history refresh; current batch signal may adapt sooner, but reduced Expert capacity may offset the benefit')

OPTIONS = dict(host_kv_limit_bytes=5*GIB,progress_every=16,expert_repeats=5,
               robust_costs=True,measure_cpu_cold_setup=True,cpu_cold_setup_repeats=5)

SEARCH_TOPICS = [
    '计算机系统需要安排请求、访问内存和完成计算。缓存保存可重复使用的数据，减少重复搬运。请用若干条目详细解释。',
    '旅行规划需要比较城市交通、天气和景点开放时间。请按每天的上午、下午和晚上安排活动并说明理由。',
    '植物生长需要光照、水分和适合的土壤。请列出常见养护问题，逐条解释原因与观察方法。',
    '编写程序时需要拆分函数、设计输入输出和测试边界情况。请逐步解释如何调试一个数据处理程序。']
CONFIRM_TOPICS = [
    '图书馆管理涉及图书分类、借阅记录和读者服务。请逐项介绍这些工作的步骤以及注意事项。',
    '烹饪需要准备食材、控制火候并安排烹煮顺序。请介绍制作一顿家常饭的步骤和注意事项。',
    '天文观测需要记录天空、时间、方向和天气。请解释初学者如何整理观察记录与比较现象。',
    '体育训练包括热身、力量、耐力与休息。请安排一份循序渐进的训练计划并解释各部分作用。']


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def implementation_hash():
    folder = Path(__file__).resolve().parent
    return hashlib.sha256((backend_hash()+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                           +hashlib.sha256((folder/'kite_readiness.py').read_bytes()).hexdigest()).encode()).hexdigest()


def specifications():
    specs = [dict(request_id='background',release_step=0,input_tokens=8192,output_tokens=104,topic_index=0)]
    specs += [dict(request_id=f'burst-{step}-{i}',release_step=step,input_tokens=11264,output_tokens=16,
                   topic_index=1+(i+wave)%3)
              for wave,step in enumerate((17,49,81)) for i in range(3)]
    return specs


def workload(tokenizer, split):
    template = tokenizer.apply_chat_template([{'role':'user','content':MARKER}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if template.count(MARKER)!=1:
        raise RuntimeError('Unknown chat template boundaries.')
    prefix,suffix = template.split(MARKER)
    encode = lambda s:tokenizer.encode(s,add_special_tokens=False)
    prefix,suffix = encode(prefix),encode(suffix)
    topics = SEARCH_TOPICS if split=='search' else CONFIRM_TOPICS
    specs = specifications()
    for spec in specs:
        spec['source_text'] = topics[spec['topic_index']]
        spec['input_ids'] = build_ids(prefix,encode(spec['source_text']),suffix,spec['input_tokens'])
    return specs


def structural_screen():
    """Exact shape/page arithmetic only. Does not generate or predict router traces."""
    specs = [{**s,'input_ids':[1]*s['input_tokens']} for s in specifications()]
    requests = new_requests(specs)
    rows = []
    step = 0
    while not all(r.done for r in requests):
        active = ready_batch(requests,step)
        layout = packed_layout(active)
        live = sum((row['after']+127)//128 for row in layout)
        q,k = sum(len(r.query) for r in active),sum(row['after'] for row in layout)
        padded_k = k+(-k)%8
        rows.append(dict(step=step,prefill=sum(row['phase']=='prefill' for row in layout),
            requests=len(active),kv_pages=live,kv_bytes=live*KV_BYTES,mask_bytes=2*q*k,
            padded_mask_bytes=2*q*padded_k,
            dense_kv_payload_bytes=48*2*4*128*2*k))
        for req,row in zip(active,layout):
            req.kv_length = row['after']
            req.generated.append(1)
        step += 1
    return dict(scope='shape/page feasibility, not GPU measurement or speedup prediction',
        request_count=len(specs),batch_count=len(rows),output_tokens=sum(s['output_tokens'] for s in specs),
        input_tokens=sum(s['input_tokens'] for s in specs),
        query_tokens=sum(s['input_tokens']+s['output_tokens']-1 for s in specs),
        max_active=max(r['requests'] for r in rows),max_kv_pages=max(r['kv_pages'] for r in rows),
        max_kv_bytes=max(r['kv_bytes'] for r in rows),max_mask_bytes=max(r['mask_bytes'] for r in rows),
        max_padded_mask_bytes=max(r['padded_mask_bytes'] for r in rows),
        max_dense_kv_payload_bytes=max(r['dense_kv_payload_bytes'] for r in rows),batches=rows)


def choose_selection(tasks):
    pair = {r['mode']:r for r in tasks if r['status']=='passed' and r['budget_mib']==8192}
    if set(pair)!={'baseline','kite'}:
        raise RuntimeError('Both measured workloads must finish before any selection.')
    a,b = pair['baseline']['workload_total_ms'],pair['kite']['workload_total_ms']
    row = dict(budget_mib=8192,baseline_ms=a,kite_ms=b,speedup=a/b)
    return dict(status='selected' if a>b else 'no_positive_result',selected=row if a>b else None,
                all_budgets=[row],selection_metric='actual E2E including startup, not replayed cost')


def run(args):
    started = time.perf_counter_ns()
    checkpoint = Path(args.checkpoint).resolve()
    ready = readiness(checkpoint,Path(__file__).resolve().parents[1])
    versions = ready['checks'].get('software')
    snapshot = ready.get('resources')
    manifest = ready['checks'].get('checkpoint',{})
    output = new_output(args.results_dir,'kv-shock-'+args.command)
    status = dict(status='running',stage='readiness',command=args.command,args=vars(args),software=versions,
        protocol=PROTOCOL,protocol_sha256=protocol_hash(),runtime_options=OPTIONS,
        model_manifest_sha256=manifest.get('content_manifest_sha256'),implementation_sha256=implementation_hash(),
        resources_before=snapshot,tasks=[],is_WiSP=False,scope=PROTOCOL['scope'])
    write_json(output/'readiness.json',ready)
    write_json(output/'summary.json',status)
    write_json(output/'structural_screen.json',structural_screen())
    monitor = None
    try:
        require_ready(ready)
        status['stage'] = 'cuda_preflight'
        frozen = None
        if args.command=='confirm':
            frozen = json.loads(Path(args.selection).read_text(encoding='utf-8'))
            if (frozen['status']!='selected' or frozen['protocol_sha256']!=protocol_hash()
                    or frozen['implementation_sha256']!=implementation_hash()
                    or frozen['model_manifest_sha256']!=manifest.get('content_manifest_sha256')):
                raise RuntimeError('Confirmation requires a positive frozen selection for this exact version.')
            status['frozen_selection'] = frozen
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
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
        show('Checking memory-efficient attention availability before loading the model.')
        with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            # Capability gate only, not a benchmark. No long-context math fallback.
            q = torch.zeros((1,32,3,128),dtype=torch.bfloat16,device=device)
            k = torch.zeros((1,32,8,128),dtype=torch.bfloat16,device=device)
            mask = torch.full((1,1,3,8),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16,device=device)
            mask[:,:,:,:5] = 0
            probe = torch.nn.functional.scaled_dot_product_attention(q,k,k,attn_mask=mask)
            if not bool(torch.isfinite(probe).all()):
                raise RuntimeError('Attention capability gate returned nonfinite values.')
            status['attention_gate'] = dict(status='passed',backend='EFFICIENT_ATTENTION',mask_columns=8)
            monitor = ResourceMonitor(evidence_path=output/'gpu_monitor_startup.json')
            del q,k,mask,probe
            status['stage'] = 'model_loading'
            write_json(output/'summary.json',status)
            show('Loading real Qwen once; next: 8 GiB pool, two complete measured workloads.')
            engine,tokenizer,status['loading'] = load_engine(torch,checkpoint,device,8*GIB,snapshot,
                                                          health_check=monitor.check)
            engine.pad_attention_keys = True
            monitor.check()
            engine.health_check = monitor.check
            status['stage'] = 'integration_validation'
            write_json(output/'summary.json',status)
            specs = workload(tokenizer,'search' if args.command=='search' else 'confirmation')
            write_json(output/'workload.json',dict(requests=specs,protocol=PROTOCOL,
                split='search' if args.command=='search' else 'confirmation'))
            status['workload_sha256'] = hashlib.sha256((output/'workload.json').read_bytes()).hexdigest()
            if frozen is not None:
                prior_root = Path(frozen['search_run'])
                prior = json.loads((prior_root/'summary.json').read_text(encoding='utf-8'))
                validation = json.loads((prior_root/'validation.json').read_text(encoding='utf-8'))
                if prior['status']!='passed' or prior['implementation_sha256']!=implementation_hash() or validation['status']!='passed':
                    raise RuntimeError('Frozen search did not pass integration under these exact sources.')
                validation = {**validation,'reused_from':str(prior_root)}
            else:
                show('Checking short genuine packed/serial and fallback correctness, without loosening tolerances.')
                validation = validate(engine,specs)
            write_json(output/'validation.json',validation)
            status['validation_status'] = validation['status']
            modes = ['baseline','kite']
            random.Random(args.seed).shuffle(modes)
            for mode in modes:
                status['stage'] = 'workload_'+mode
                write_json(output/'summary.json',status)
                task = execute_task(engine,tokenizer,specs,8192,mode,output/f'8192-{mode}',monitor,
                                    runtime_options=OPTIONS,timing_protocol=PROTOCOL)
                status['tasks'].append(task)
                write_json(output/'summary.json',status)
                show(f'{mode} 8192 MiB: E2E={task["workload_total_ms"]/1000:.3f}s (startup included).')
            if args.command=='search':
                selection = choose_selection(status['tasks'])
                selection.update(protocol_sha256=protocol_hash(),implementation_sha256=implementation_hash(),
                    model_manifest_sha256=manifest.get('content_manifest_sha256'),search_run=str(output),exploratory=True)
                write_json(output/'selection.json',selection)
                status['selection'] = selection
            monitor.check()
            status.update(status='passed',stage='complete')
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__,error=str(error))
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        if monitor is not None:
            monitor.close()
            write_json(output/'resource_monitor.json',monitor.record())
            if monitor.error and status['status']=='passed':
                status.update(status='failed',error_type='ResourceInterference',error=monitor.error)
        status['finished_utc'] = utc_now()
        status['job_wall_ms_including_loading_validation_and_all_comparisons'] = (time.perf_counter_ns()-started)/1e6
        write_json(output/'summary.json',status)
        print('Results:',output,flush=True)
    if status['status']!='passed':
        raise RuntimeError(status.get('error','Workload did not pass.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','search','confirm'])
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--selection')
    parser.add_argument('--seed',type=int,choices=[17,23,41],default=17)
    parser.add_argument('--results-dir',default='/workspace/results/kite_advantage')
    args = parser.parse_args()
    if (args.command=='confirm')!=bool(args.selection):
        parser.error('--selection is required only for confirmation.')
    if args.command=='check':
        report = readiness(Path(args.checkpoint),Path(__file__).resolve().parents[1])
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        raise SystemExit(0 if report['status']=='ready' else 2 if report['status']=='blocked' else 1)
    run(args)


if __name__=='__main__':
    main()
