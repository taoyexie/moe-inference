#!/usr/bin/env python3
"""Six controlled real-model requests; one model load, independent per-request KV."""
import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from kite_trace import utc_now, write_json
from qwen_smoke import generate_trace, run as smoke_run, show

TOPICS = (
    ('systems', '计算机推理系统需要管理显存、专家权重和键值缓存。预填充处理提示词，解码逐步生成。'
                '数据可以在CPU和GPU间搬运，也可以通过计算重新产生。请概括这段材料的主题。'),
    ('travel', '旅行计划包括城市交通、天气安排、博物馆和当地饮食。上午参观展览，下午步行游览，'
               '晚上选择合适的餐馆并整理第二天的路线。请概括这段材料的主题。'))
MARKER = 'KITE_TRACE_CONTENT_7F9C'


def build_ids(prefix, body, suffix, target):
    count = target-len(prefix)-len(suffix)
    if not body or count < 1:
        raise ValueError('Target cannot accommodate the chat wrapper and nonempty content.')
    return prefix+(body*((count+len(body)-1)//len(body)))[:count]+suffix


def request_plan(lengths):
    return [{'request_id':f'trace-{length}-{topic}', 'target_tokens':length, 'topic':topic,
             'source_text':text} for length in lengths for topic,text in TOPICS]


def prepare_requests(tokenizer, lengths):
    template = tokenizer.apply_chat_template([{'role':'user','content':MARKER}],
                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if template.count(MARKER) != 1:
        raise RuntimeError('Unexpected chat template; refusing to guess token boundaries.')
    prefix_text, suffix_text = template.split(MARKER)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix, suffix = encode(prefix_text), encode(suffix_text)
    requests = []
    for item in request_plan(lengths):
        tokens = build_ids(prefix, encode(item['source_text']), suffix, item['target_tokens'])
        request = {**item, 'input_tokens':len(tokens), 'input_token_ids':tokens,
                   'source':'handwritten_topic_repeated_and_truncated_at_token_level',
                   'chat_prefix_text':prefix_text, 'chat_suffix_text':suffix_text,
                   'decoded_input_for_inspection':tokenizer.decode(tokens,skip_special_tokens=False),
                   'enable_thinking':False, 'sampling':'greedy', 'seed':17,
                   'length_is_controlled_not_a_natural_workload':True}
        requests.append(request)
    return requests


def collect(torch, model, tokenizer, root, device, eos_ids, lengths, new_tokens):
    base = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    completed = []
    for request in prepare_requests(tokenizer,lengths):
        item = request
        tokens = request['input_token_ids']
        child = root/item['request_id']
        child.mkdir(exist_ok=False)
        write_json(child/'request.json', request)
        for name in ('device_map.json','model_config.json','resources_before.json','resources_after_load.json'):
            shutil.copyfile(root/name,child/name)
        status = {**base, 'status':'running', 'is_multi_request_trace':False,
                  'model_reused_from_parent':True, 'request_id':item['request_id'], 'started_utc':utc_now()}
        write_json(child/'summary.json', status)
        show(f"Request {len(completed)+1}/{len(lengths)*len(TOPICS)}: {item['request_id']}, input={len(tokens)}, output cap={new_tokens}.")
        try:
            ids = torch.tensor([tokens],dtype=torch.long)
            result = generate_trace(torch,model,ids,new_tokens,eos_ids,child,device,
                                    request_id=item['request_id'])
            status.update(result, status='passed',
                          generated_text=tokenizer.decode(result['generated_token_ids'],skip_special_tokens=True),
                          evidence_scope='real_expert_invocations_and_KV_tensor_metadata',
                          timing_scope='instrumented_diagnostic_only_not_TTFT_TPOT_or_speedup')
            show(f"{item['request_id']}: {result['forward_count']} forwards, {result['decode_forward_count']} decode; {status['generated_text']}")
            completed.append({'request_id':item['request_id'], 'folder':item['request_id'],
                              'input_tokens':len(tokens), 'forward_count':result['forward_count'],
                              'decode_forward_count':result['decode_forward_count'], 'stop_reason':result['stop_reason'],
                              'request_sha256':hashlib.sha256((child/'request.json').read_bytes()).hexdigest()})
        except BaseException as error:
            status.update(status='failed',error=str(error),error_type=type(error).__name__)
            raise
        finally:
            status['finished_utc'] = utc_now()
            write_json(child/'summary.json',status)
            write_json(root/'requests.json',{'completed':completed, 'planned':request_plan(lengths),
                       'schedule':'sequential_requests_not_mixed_batches',
                       'cache':'fresh_DynamicCache_per_request; expert weight placement reused',
                       'not_physical_cache_miss_trace':True})
        del ids
        gc.collect()
    return {'request_count':len(completed), 'forward_count':sum(r['forward_count'] for r in completed),
            'decode_forward_count':sum(r['decode_forward_count'] for r in completed),
            'requests_with_decode':sum(r['decode_forward_count']>0 for r in completed),
            'schedule':'sequential_requests_not_mixed_batches',
            'is_kite_allocator':False, 'is_fallback_benchmark':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--lengths',default='128,512,1024')
    parser.add_argument('--new-tokens',type=int,default=8)
    parser.add_argument('--results-dir',default='/workspace/results/multi_trace')
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(',')]
    if (lengths != sorted(set(lengths)) or not lengths or not all(64<=x<=2048 for x in lengths)
            or not 2<=args.new_tokens<=32):
        parser.error('Expected sorted distinct lengths 64..2048 and new-tokens 2..32.')
    load_args = SimpleNamespace(command='run',checkpoint=args.checkpoint,results_dir=args.results_dir,
                  prompt='请用一句话解释什么是缓存。',new_tokens=args.new_tokens,max_prompt_tokens=128,
                  gpu_memory_gib=24,cpu_memory_gib=48,cpu_threads=8,
                  cpu_affinity='24,25,26,27,28,29,30,31',allow_shared_gpu=False,tiny=False)
    smoke_run(load_args,request_runner=lambda torch,model,tokenizer,root,device,eos:
              collect(torch,model,tokenizer,root,device,eos,lengths,args.new_tokens))


if __name__ == '__main__':
    main()
