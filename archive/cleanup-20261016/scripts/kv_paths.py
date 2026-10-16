#!/usr/bin/env python3
"""Real Qwen KV recovery microbenchmark; not an allocator or serving benchmark."""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
import traceback
from pathlib import Path

from expert_paths import error_metrics, new_output, parse_int_list, sha256_file, summarize, timed
from kite_trace import (GIB, check_gpu_budget, mapped_device,
                        require_versions, resource_snapshot, utc_now, write_json)
from qwen_smoke import initialize_cuda_stats, show, verify_checkpoint

PATHS = ('restore', 'recompute')
TEXT = ('推理系统需要在有限显存中管理专家权重和键值缓存。'
        '请求包含提示词处理与逐步生成，恢复缺失数据需要传输或重新计算。\n')


def object_spec(block_tokens):
    if not isinstance(block_tokens, int) or not 1 <= block_tokens <= 128:
        raise ValueError('Block size must be 1..128 tokens for this pilot.')
    shape = (48, 2, 1, 4, block_tokens, 128)
    size = 2
    for dimension in shape:
        size *= dimension
    return {'shape': list(shape), 'dtype': 'torch.bfloat16', 'bytes': size,
            'layer_count': 48, 'kv_heads': 4, 'head_dim': 128,
            'block_tokens': block_tokens, 'layout': 'layer,K_or_V,batch,kv_head,token,head_dim',
            'semantics': 'one fixed all-layer KV block, NOT one whole context'}


def validate_args(args):
    lengths = parse_int_list(args.contexts, args.block_tokens, 4096)
    object_spec(args.block_tokens)
    if not 0 <= args.warmup <= 10 or not 1 <= args.repeats <= 30:
        raise ValueError('Expected warmup 0..10 and repeats 1..30.')
    if not 1 <= args.cpu_threads <= 16:
        raise ValueError('Expected CPU threads 1..16.')
    if not 1 <= args.gpu_memory_gib <= 28 or not 1 <= args.cpu_memory_gib <= 64:
        raise ValueError('Weight budgets outside the tested safety range.')
    return lengths


def repeated_prefix(tokens, length):
    if not tokens or length < 1:
        raise ValueError('A nonempty source token sequence and positive length are required.')
    return (tokens * ((length + len(tokens)-1)//len(tokens)))[:length]


def validate_loading(info):
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError('Incomplete/mismatched base-model weights; no random substitutes permitted.')
    if set(info.get('unexpected_keys', [])) - {'lm_head.weight'}:
        raise RuntimeError('Unexpected checkpoint tensors beyond the deliberately unused LM head.')


def load_model(torch, folder, args, output, snapshot, summary):
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    cfg = AutoConfig.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
    with init_empty_weights():
        empty = AutoModel.from_config(cfg, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    mapping = infer_auto_device_map(empty,
                  max_memory={0: f'{args.gpu_memory_gib}GiB', 'cpu': f'{args.cpu_memory_gib}GiB'},
                  no_split_module_classes=empty._no_split_modules, dtype=torch.bfloat16)
    write_json(output/'device_map.json', {key: str(value) for key, value in mapping.items()})
    if any(str(value) == 'disk' for value in mapping.values()):
        raise RuntimeError('Disk offload is forbidden. No checkpoint or model was changed.')
    cpu_bytes = sum(p.numel()*2 for name, p in empty.named_parameters()
                    if mapped_device(name, mapping) == 'cpu')
    available = snapshot['effective_available_bytes']
    if available is None or available < cpu_bytes + 8*GIB:
        raise RuntimeError('Insufficient host/cgroup headroom for CPU weights plus 8 GiB reserve.')
    summary['cpu_weight_bytes_planned'] = cpu_bytes
    del empty
    gc.collect()
    show('Loading the real BF16 Qwen transformer once; CPU weight offload, no LM head or disk offload.')
    started = time.perf_counter()
    model, info = AutoModel.from_pretrained(str(folder), config=cfg, local_files_only=True,
                  trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
                  attn_implementation='sdpa', device_map=mapping, low_cpu_mem_usage=True,
                  offload_state_dict=False, output_loading_info=True)
    write_json(output/'loading_info.json', info)
    validate_loading(info)
    model.eval()
    torch.cuda.synchronize()
    summary.update(load_seconds=time.perf_counter()-started, model_class=type(model).__name__)
    write_json(output/'model_config.json', cfg.to_dict())
    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
    return model, tokenizer


def pack_tail(torch, cache, destination, length, block_tokens):
    # Both recovery paths must end in this exact physical layout and absolute token range.
    if len(cache.key_cache) != 48 or len(cache.value_cache) != 48:
        raise RuntimeError('Expected all 48 layers of real KV.')
    for layer, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
        for kind, tensor in enumerate((key, value)):
            if (tuple(tensor.shape) != (1, 4, length, 128)
                    or tensor.dtype != torch.bfloat16 or tensor.device.type != 'cuda'):
                raise RuntimeError('KV shape, precision or execution device mismatch.')
            destination[layer, kind].copy_(tensor[:, :, length-block_tokens:length, :])


def check_output(torch, actual, reference, path):
    metrics = error_metrics(torch, actual, reference)
    passed = (metrics['max_abs'] == 0 if path == 'restore' else
              metrics['relative_l2'] <= .03 and metrics['normalized_max_abs'] <= .05)
    if not passed:
        raise RuntimeError(f'KV recovery numerical check failed: {path}: {metrics}')
    return {**metrics, 'pass': True}


def benchmark_point(torch, model, token_ids, length, args, output, raw, rng, device):
    from safetensors.torch import save_file
    from transformers import DynamicCache
    spec = object_spec(args.block_tokens)
    ids = torch.tensor([token_ids[:length]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    positions = torch.arange(length, device=device, dtype=torch.long).unsqueeze(0)
    destination = torch.empty(spec['shape'], dtype=torch.bfloat16, device=device)
    # GPU destination, token IDs, masks and weights are provisioned outside both path timings.
    def recompute():
        cache = DynamicCache()  # No prefix KV or hidden states are reused, even across warmups.
        result = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                       past_key_values=cache, use_cache=True, return_dict=True,
                       output_attentions=False, output_hidden_states=False, output_router_logits=False)
        pack_tail(torch, result.past_key_values, destination, length, args.block_tokens)
        # Releasing incidental prefix KV/hidden states is part of recovery, not retained as free reuse.
        del result, cache
        return destination

    show(f'context={length}: creating actual reference KV; this full forward may take minutes.')
    reference_ms, _ = timed(torch, device, recompute)
    reference = destination.clone()
    source = torch.empty(spec['shape'], dtype=torch.bfloat16, pin_memory=True)
    source.copy_(reference)
    torch.cuda.synchronize(device)
    source_name = f'context-{length}-source.safetensors'
    save_file({'kv': source}, str(output/source_name))
    reference_meta = {'context_tokens': length, 'block_range': [length-args.block_tokens, length],
                      'source_file': source_name, 'source_sha256': sha256_file(output/source_name),
                      'reference_creation_ms_not_a_trial': reference_ms, 'object': spec}
    write_json(output/f'context-{length}-reference.json', reference_meta)

    def restore():
        destination.copy_(source, non_blocking=True)
        return destination

    operations = {'restore': restore, 'recompute': recompute}
    values = {path: [] for path in PATHS}
    for warmup in (True, False):
        count = args.warmup if warmup else args.repeats
        for trial in range(count):
            order = list(PATHS)
            rng.shuffle(order)
            for position, path in enumerate(order):
                # Erase the target outside timing, so checking cannot pass on a stale reference.
                destination.fill_(float('nan'))
                show(f'context={length} {"warmup" if warmup else "trial"} {trial+1}/{count}: {path}')
                elapsed, recovered = timed(torch, device, operations[path])
                metrics = check_output(torch, recovered, reference, path)
                row = {'context_tokens': length, 'path': path, 'warmup': warmup, 'trial': trial,
                       'order': order, 'order_position': position, 'elapsed_ms': elapsed,
                       'correctness': metrics, 'object_bytes': spec['bytes'], 'time_utc': utc_now()}
                raw.write(json.dumps(row)+'\n')
                raw.flush()
                if not warmup:
                    values[path].append(elapsed)
                show(f'{path}: {elapsed:.3f} ms; correctness PASS')
    return {**reference_meta, 'timings': {path: summarize(values[path]) for path in PATHS}}


def benchmark(args):
    lengths = validate_args(args)
    versions = require_versions()
    folder = Path(args.checkpoint).resolve()
    manifest = verify_checkpoint(folder)
    snapshot = resource_snapshot(folder)
    check_gpu_budget(snapshot, args.gpu_memory_gib, reserve_gib=8)
    output = new_output(args.results_dir, 'kv-paths')
    summary = {'status': 'running', 'started_utc': utc_now(), 'args': vars(args), 'software': versions,
               'model_source': manifest.get('source'), 'model_revision': manifest['revision'],
               'model_content_manifest_sha256': manifest.get('content_manifest_sha256'),
               'is_allocator_benchmark': False, 'is_end_to_end_serving_benchmark': False,
               'object': object_spec(args.block_tokens), 'points': [],
               'protocol': 'pinned_CPU_restore_vs_full_token_prefix_replay_to_same_tail_block',
               'recompute_prerequisite': 'original tokens; no reusable prefix KV or hidden states',
               'restore_prerequisite': 'a valid BF16 CPU copy of the same block already exists',
               'limitations': ['Controlled nested token prefixes, not a natural request workload.',
                   'Includes CPU-offloaded expert weight traffic inside recompute; do not charge it twice.',
                   'Prefix replay may serve multiple missing blocks; this is NOT additive per-block cost.',
                   'Restoring one block alone does not reconstruct other absent blocks.',
                   'Packed all-layer block microbenchmark, not a production paged cache implementation.',
                   'Small pilot: no fit quality/generalization claim; resource snapshots do not reserve the GPU.']}
    write_json(output/'summary.json', summary)
    write_json(output/'resources_before.json', snapshot)
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.version.cuda != '11.8':
            raise RuntimeError('Expected the previously validated one-GPU PyTorch cu118 environment.')
        cores = set(parse_int_list(args.cpu_affinity, 0, 4095))
        if not hasattr(os, 'sched_getaffinity') or not cores <= os.sched_getaffinity(0):
            raise RuntimeError('CPU affinity is outside the allowed mask.')
        os.sched_setaffinity(0, cores)
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        device = torch.device('cuda:0')
        initialize_cuda_stats(torch, device)
        model, tokenizer = load_model(torch, folder, args, output, snapshot, summary)
        source_tokens = tokenizer.encode(TEXT, add_special_tokens=False)
        ids = repeated_prefix(source_tokens, max(lengths))
        input_record = {'source_text': TEXT, 'construction': 'repeat tokenized text then exact prefix truncation',
                        'source_token_ids': source_tokens, 'longest_input_ids': ids,
                        'contexts': lengths, 'absolute_positions': '0..context_tokens-1',
                        'not_real_chat_requests': True}
        write_json(output/'inputs.json', input_record)
        summary['inputs_sha256'] = sha256_file(output/'inputs.json')
        write_json(output/'resources_after_load.json', resource_snapshot(folder))
        write_json(output/'summary.json', summary)
        rng = random.Random(args.seed)
        order = list(lengths)
        rng.shuffle(order)
        summary['context_execution_order'] = order
        with (output/'raw.jsonl').open('x', encoding='utf-8') as raw, torch.inference_mode():
            for length in order:
                point = benchmark_point(torch, model, ids, length, args, output, raw, rng, device)
                summary['points'].append(point)
                write_json(output/'summary.json', summary)
        summary.update(status='passed', cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    except BaseException as error:
        summary.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       error_type=type(error).__name__, error=str(error))
        (output/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        summary['finished_utc'] = utc_now()
        write_json(output/'summary.json', summary)
        print('Benchmark directory:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--contexts', default='256,512')
    parser.add_argument('--block-tokens', type=int, default=128)
    parser.add_argument('--gpu-memory-gib', type=int, default=24)
    parser.add_argument('--cpu-memory-gib', type=int, default=48)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--cpu-affinity', default='24,25,26,27,28,29,30,31')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--results-dir', default='/workspace/results/kv_paths')
    benchmark(parser.parse_args())


if __name__ == '__main__':
    main()
