#!/usr/bin/env python3
"""Uninstrumented local sequential inference timing on the existing offload backend.

No KITE cache or allocator is active. This script must not be used to claim KITE
speedup. It attaches no expert hooks and writes no per-layer records while timing.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

from kite_trace import utc_now, write_json
from multi_trace import prepare_requests
from qwen_smoke import run as smoke_run, show


def token_metrics(started_ns, finished_ns):
    if not finished_ns or any(b<a for a,b in zip([started_ns]+finished_ns[:-1],finished_ns)):
        raise ValueError('Expected nondecreasing completion timestamps and at least one token.')
    offsets = [(tick-started_ns)/1e6 for tick in finished_ns]
    gaps = [b-a for a,b in zip(offsets,offsets[1:])]
    return {'ttft_ms':offsets[0], 'completion_ms':offsets[-1],
            'tpot_ms':mean(gaps) if gaps else None, 'itl_ms':gaps,
            'token_completion_offsets_ms':offsets, 'output_tokens':len(offsets)}


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    if not all(math.isfinite(v) and v>=0 for v in values):
        raise ValueError('Invalid latency sample.')
    def quantile(p):
        i = (len(values)-1)*p
        lower,upper = math.floor(i),math.ceil(i)
        return values[lower]+(values[upper]-values[lower])*(i-lower)
    return {'n':len(values),'mean':mean(values),'median':median(values),
            'p10':quantile(.1),'p90':quantile(.9),'min':values[0],'max':values[-1]}


def latency_summary(rows):
    return {'ttft_ms':distribution([row['ttft_ms'] for row in rows]),
            'completion_ms':distribution([row['completion_ms'] for row in rows]),
            'tpot_ms':distribution([row['tpot_ms'] for row in rows if row['tpot_ms'] is not None]),
            'itl_ms':distribution([gap for row in rows for gap in row['itl_ms']])}


def measure_request(torch,model,input_ids,device,eos_ids,new_tokens):
    from transformers import DynamicCache
    generated,ticks = [],[]
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    with torch.inference_mode():
        ids = input_ids.to(device)
        mask = torch.ones_like(ids)
        cache = DynamicCache()
        for step in range(new_tokens):
            before = int(cache.get_seq_length())
            output = model(input_ids=ids, attention_mask=mask, past_key_values=cache,
                           cache_position=torch.arange(before,before+ids.shape[1],device=device),
                           use_cache=True,output_router_logits=False,return_dict=True,logits_to_keep=1)
            token = int(output.logits[0,-1].argmax().item())
            torch.cuda.synchronize(device)
            ticks.append(time.perf_counter_ns())
            generated.append(token)
            cache = output.past_key_values
            del output
            if token in eos_ids or step+1 == new_tokens:
                break
            ids = torch.tensor([[token]],dtype=torch.long,device=device)
            mask = torch.ones((1,int(cache.get_seq_length())+1),dtype=torch.long,device=device)
        cache_length = int(cache.get_seq_length())
    return {**token_metrics(started,ticks),'generated_token_ids':generated,
            'stop_reason':'eos' if generated[-1] in eos_ids else 'max_new_tokens',
            'forward_count':len(generated),'decode_forward_count':max(0,len(generated)-1),
            'final_cache_length':cache_length,
            'timing_scope':'pretokenized_CPU_input_to_first_or_last_token_id_ready_on_CPU; local batch=1',
            'excludes':['tokenization','model_loading','server_queue','network','text_detokenization','post_generation_cache_cleanup'],
            'includes':['input_H2D','all_model_execution_and_weight_offload','greedy_argmax','completion_synchronization']}


def collect(torch,model,tokenizer,root,device,eos_ids,args,job_started_ns):
    native_hooks = sum(len(module._forward_hooks)+len(module._forward_pre_hooks)
                       for module in model.modules())
    if native_hooks:
        raise RuntimeError('Registered module hooks found; refusing to label this an uninstrumented baseline.')
    templates = prepare_requests(tokenizer,args.lengths)
    plan = [{**request,'request_id':f'repeat-{repeat:02d}-'+request['request_id']}
            for repeat in range(args.repeats) for request in templates]
    if not 0 <= args.warmup_requests < len(plan):
        raise ValueError('Warmup prefix must leave at least one evaluation request.')
    write_json(root/'workload.json',{'requests':plan, 'warmup_requests':args.warmup_requests,
              'schedule':'sequential_requests_batch_1', 'new_tokens_cap':args.new_tokens,
              'backend':'hf_accelerate_cpu_weight_offload', 'allocator':'none',
              'cost_calibration_performed':False, 'trace_hooks_attached':False,
              'note':'Baseline warmup is NOT four-path cost calibration.'})
    rows = []
    workload_started = time.perf_counter_ns()
    evaluation_started = prefix_finished = workload_started
    with (root/'request_metrics.jsonl').open('x',encoding='utf-8') as raw:
        for index,request in enumerate(plan):
            is_warmup = index < args.warmup_requests
            if index == args.warmup_requests:
                evaluation_started = time.perf_counter_ns()
                prefix_finished = evaluation_started
            ids = torch.tensor([request['input_token_ids']],dtype=torch.long)
            show(f"Baseline {index+1}/{len(plan)} {'warmup' if is_warmup else 'evaluation'}: input={request['input_tokens']} cap={args.new_tokens}")
            result = measure_request(torch,model,ids,device,eos_ids,args.new_tokens)
            row = {**result,'request_id':request['request_id'],'input_tokens':request['input_tokens'],
                   'topic':request['topic'],'warmup':is_warmup,'time_utc':utc_now(),
                   'generated_text':tokenizer.decode(result['generated_token_ids'],skip_special_tokens=True)}
            rows.append(row)
            raw.write(json.dumps(row,ensure_ascii=False)+'\n')
            raw.flush()
            show(f"TTFT={row['ttft_ms']:.2f}ms completion={row['completion_ms']:.2f}ms output={row['output_tokens']}")
            del ids
            gc.collect()  # Outside request latency; included in driver wall time.
    ended = time.perf_counter_ns()
    evaluation = [row for row in rows if not row['warmup']]
    aggregate = latency_summary(evaluation)
    write_json(root/'performance.json',{
        'status':'passed','backend':'hf_accelerate_cpu_weight_offload','is_kite_allocator':False,
        'trace_hooks_attached':False,'cost_calibration_performed':False,'native_module_hook_count':native_hooks,
        'request_count':len(rows),'warmup_count':args.warmup_requests,'evaluation_count':len(evaluation),
        'evaluation_output_tokens':sum(row['output_tokens'] for row in evaluation),
        'loaded_workload_wall_ms_including_warmup_and_driver_gaps':(ended-workload_started)/1e6,
        'warmup_prefix_wall_ms':(prefix_finished-workload_started)/1e6 if args.warmup_requests else 0.,
        'evaluation_wall_ms_including_driver_gaps':(ended-evaluation_started)/1e6,
        'driver_job_wall_ms_including_model_load':(ended-job_started_ns)/1e6,
        'evaluation_processing_tokens_per_second_excluding_driver_gaps':
            sum(row['output_tokens'] for row in evaluation)/(sum(row['completion_ms'] for row in evaluation)/1000),
        'evaluation_wall_tokens_per_second_including_driver_gaps':
            sum(row['output_tokens'] for row in evaluation)/((ended-evaluation_started)/1e9),
        'evaluation_metrics':aggregate,
        'evaluation_by_input_tokens':{str(length):latency_summary([row for row in evaluation if row['input_tokens']==length])
                                      for length in sorted({row['input_tokens'] for row in evaluation})},
        'population':'controlled six-request sequence, not a natural serving workload; pooled distributions have unequal length weights',
        'timing':'per-request input already tokenized on CPU; no service queue or network; model load and warmup separately retained',
        'workload_sha256':hashlib.sha256((root/'workload.json').read_bytes()).hexdigest()})
    return {'instrumented':False,'is_multi_request_trace':False,'is_kite_allocator':False,
            'is_fallback_benchmark':False,'request_count':len(rows),
            'require_decode':False,
            'forward_count':sum(row['forward_count'] for row in rows),
            'decode_forward_count':sum(row['decode_forward_count'] for row in rows),
            'evidence_scope':'existing_backend_uninstrumented_local_latency_baseline_not_KITE',
            'timing_scope':'per_request_TTFT_TPOT_completion; model_load_and_warmup_reported_separately',
            'performance_file':'performance.json'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--lengths',default='128,512,1024')
    parser.add_argument('--new-tokens',type=int,default=8)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--warmup-requests',type=int,default=1)
    parser.add_argument('--results-dir',default='/workspace/results/workload_baseline')
    args = parser.parse_args()
    args.lengths = [int(value) for value in args.lengths.split(',')]
    if (not args.lengths or args.lengths != sorted(set(args.lengths))
            or not all(64<=value<=2048 for value in args.lengths)
            or not 2<=args.new_tokens<=32 or not 1<=args.repeats<=3
            or not 0<=args.warmup_requests<len(args.lengths)*2*args.repeats):
        parser.error('Invalid workload size, token cap or warmup prefix.')
    job_started_ns = time.perf_counter_ns()
    load_args = SimpleNamespace(command='run',checkpoint=args.checkpoint,results_dir=args.results_dir,
        prompt='请用一句话解释什么是缓存。',new_tokens=args.new_tokens,max_prompt_tokens=128,
        gpu_memory_gib=24,cpu_memory_gib=48,cpu_threads=8,
        cpu_affinity='24,25,26,27,28,29,30,31',allow_shared_gpu=False,tiny=False)
    smoke_run(load_args,request_runner=lambda torch,model,tokenizer,root,device,eos:
        collect(torch,model,tokenizer,root,device,eos,args,job_started_ns))


if __name__ == '__main__':
    main()
