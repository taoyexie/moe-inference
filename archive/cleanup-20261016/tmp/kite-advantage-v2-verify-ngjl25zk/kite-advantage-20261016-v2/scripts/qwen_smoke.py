#!/usr/bin/env python3
"""Pinned Qwen3 MoE smoke trace. No KITE allocator or fallback benchmark yet."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from kite_trace import (GIB, ExpertRecorder, cache_snapshot, check_gpu_budget,
                        mapped_device, require_versions, resource_snapshot,
                        software_versions, utc_now, write_json)

MODEL_ID = 'Qwen/Qwen3-30B-A3B'
META_FILES = ['config.json', 'generation_config.json', 'model.safetensors.index.json',
              'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
              'added_tokens.json', 'special_tokens_map.json', 'chat_template.jinja']
EXPECTED = {'model_type': 'qwen3_moe', 'hidden_size': 2048, 'num_hidden_layers': 48,
            'num_experts': 128, 'num_experts_per_tok': 8, 'moe_intermediate_size': 768,
            'num_key_value_heads': 4, 'head_dim': 128}


def show(message):
    print(f'[{utc_now()}] {message}', flush=True)


def initialize_cuda_stats(torch, device):
    """Statistics reset does not itself initialize CUDA in pinned torch 2.7.1."""
    torch.cuda.init()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def cuda_check(args):
    """Exercise the exact initialization path before loading any checkpoint."""
    versions = require_versions()
    snapshot = resource_snapshot(args.path)
    check_gpu_budget(snapshot, 0, reserve_gib=1)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one accessible CUDA GPU.')
    device = torch.device('cuda:0')
    initialize_cuda_stats(torch, device)
    probe = torch.ones(1, dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    print(json.dumps({'status': 'passed', 'torch': versions['torch'],
                      'cuda_initialized': torch.cuda.is_initialized(),
                      'gpu': torch.cuda.get_device_name(device), 'probe_value': probe.item(),
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                      'note': 'CUDA initialization and one-value allocation only; no model loaded.'}, indent=2))


def validate_config(config):
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            raise RuntimeError(f'Unexpected model config {key}={config.get(key)}; expected {expected}')
    if config.get('quantization_config'):
        raise RuntimeError('Quantized checkpoints are outside this BF16 smoke protocol.')


def prepare(args):
    # Metadata and weight preparation do not import torch or allocate a GPU.
    if args.source == 'modelscope':
        return prepare_modelscope(args)
    for name, expected in [('huggingface-hub', '0.30.2')]:
        if software_versions()[name] != expected:
            raise RuntimeError(f'Install the pinned dependencies first: {name}=={expected}')
    from huggingface_hub import HfApi, snapshot_download
    show('Resolving the official model revision; no model loading.')
    info = HfApi().model_info(MODEL_ID, revision=args.revision or 'main', files_metadata=True, timeout=20)
    if not re.fullmatch(r'[0-9a-f]{40}', info.sha or ''):
        raise RuntimeError('Cannot resolve an immutable checkpoint revision.')
    folder = Path(args.models_dir).resolve() / 'Qwen3-30B-A3B' / info.sha
    folder.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, revision=info.sha, local_dir=str(folder),
                      allow_patterns=META_FILES, max_workers=2)
    cfg = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    validate_config(cfg)
    index = json.loads((folder/'model.safetensors.index.json').read_text(encoding='utf-8'))
    shard_names = sorted(set(index['weight_map'].values()))
    sizes = {s.rfilename: s.size for s in info.siblings}
    shards = []
    for name in shard_names:
        if Path(name).name != name or not name.endswith('.safetensors'):
            raise RuntimeError('Unexpected checkpoint shard path.')
        if not isinstance(sizes.get(name), int) or sizes[name] <= 0:
            raise RuntimeError(f'Missing authoritative shard size: {name}')
        shards.append({'name': name, 'bytes': sizes[name]})
    total = sum(s['bytes'] for s in shards)
    manifest = {'model_id': MODEL_ID, 'source': 'huggingface', 'revision': info.sha, 'config': cfg,
                'checkpoint_path': str(folder), 'shards': shards, 'download_bytes': total,
                'state_dict_bytes': index['metadata']['total_size'], 'prepared_utc': utc_now(),
                'weights_ready': False}
    write_json(folder/'kite_manifest.json', manifest)
    show(f'Metadata ready: {folder}; weights require about {total/GIB:.1f} GiB.')
    if args.weights:
        missing = sum(s['bytes'] for s in shards
                      if not (folder/s['name']).is_file() or (folder/s['name']).stat().st_size != s['bytes'])
        free = resource_snapshot(folder)['disk']['free_bytes']
        if free < missing + 10*GIB:
            raise RuntimeError('Insufficient disk headroom for checkpoint download plus 10 GiB reserve.')
        show('Downloading checkpoint weights with two workers; GPU remains unused.')
        snapshot_download(MODEL_ID, revision=info.sha, local_dir=str(folder),
                          allow_patterns=shard_names, max_workers=2)
        for shard in shards:
            if (folder/shard['name']).stat().st_size != shard['bytes']:
                raise RuntimeError(f'Incomplete shard: {shard["name"]}')
        manifest['weights_ready'] = True
        manifest['weights_ready_utc'] = utc_now()
        write_json(folder/'kite_manifest.json', manifest)
        show('Weights ready. No inference has been started.')
    print(json.dumps({'checkpoint': str(folder), 'revision': info.sha,
                      'weights_ready': manifest['weights_ready']}, ensure_ascii=False, indent=2))


def normalize_modelscope_files(records):
    selected = []
    for record in records:
        name = record.get('Path', '')
        if record.get('Type') == 'tree' or not (name in META_FILES or name.endswith('.safetensors')):
            continue
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise RuntimeError('Unexpected ModelScope file path.')
        digest = str(record.get('Sha256', '')).lower()
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise RuntimeError(f'Missing ModelScope SHA256 for {name}; cannot freeze a file identity.')
        size = record.get('Size')
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError(f'Missing ModelScope size for {name}.')
        selected.append({'name': name, 'bytes': size, 'sha256': digest})
    selected.sort(key=lambda x: x['name'])
    names = [x['name'] for x in selected]
    if len(names) != len(set(names)):
        raise RuntimeError('Duplicate ModelScope file paths.')
    for required in ['config.json', 'tokenizer_config.json', 'model.safetensors.index.json']:
        if required not in names:
            raise RuntimeError(f'ModelScope did not list required file {required}.')
    return selected


def content_identity(records):
    serialized = json.dumps(records, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


def verify_file_hashes(folder, records):
    verified = {}
    for record in records:
        path = folder / record['name']
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise RuntimeError(f'Missing/incomplete downloaded file: {record["name"]}')
        show(f'Checking SHA256: {record["name"]}')
        stat_before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for chunk in iter(lambda: file.read(8*1024**2), b''):
                digest.update(chunk)
        stat_after = path.stat()
        if digest.hexdigest() != record['sha256']:
            raise RuntimeError(f'Downloaded content does not match frozen ModelScope hash: {record["name"]}')
        if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
            raise RuntimeError('File changed during verification.')
        verified[record['name']] = {'sha256': digest.hexdigest(), 'mtime_ns': stat_after.st_mtime_ns,
                                     'bytes': stat_after.st_size}
    return verified


def prepare_modelscope(args):
    if software_versions()['modelscope'] != '1.25.0':
        raise RuntimeError('Install the optional downloader modelscope==1.25.0 first.')
    root = Path(args.models_dir).resolve()
    # ModelScope writes a session identifier even for anonymous public downloads.
    # Redirect SDK-only state to the data disk, without changing HOME or system config.
    sdk_state = root / '.modelscope-sdk'
    os.environ['MODELSCOPE_CACHE'] = str(sdk_state / 'cache')
    os.environ['MODELSCOPE_DOWNLOAD_PARALLELS'] = '1'
    from modelscope.hub.api import HubApi, ModelScopeConfig
    ModelScopeConfig.path_credential = str(sdk_state / 'credentials')
    from modelscope.hub.snapshot_download import snapshot_download
    api = HubApi(endpoint='https://www.modelscope.cn', timeout=(10, 30), max_retries=1)
    requested_revision = args.revision or 'master'
    show('Querying the official Qwen ModelScope repository; no model/GPU loading.')
    detail = api.get_valid_revision_detail(MODEL_ID, revision=requested_revision)
    if not isinstance(detail, dict) or not detail.get('Revision'):
        raise RuntimeError('Cannot resolve a ModelScope branch/tag.')
    revision = detail['Revision']
    records = normalize_modelscope_files(api.get_model_files(MODEL_ID, revision=revision,
                                                              recursive=True, use_cookies=False))
    identity = content_identity(records)
    folder = root / 'Qwen3-30B-A3B' / ('modelscope-' + identity[:16])
    folder.mkdir(parents=True, exist_ok=True)
    # This SDK accepts branches/tags rather than guaranteeing immutable commit reads.
    # File hashes, not the mutable branch name, freeze this experiment's identity.
    snapshot_download(MODEL_ID, revision=revision, local_dir=str(folder),
                      allow_patterns=META_FILES, max_workers=2)
    metadata = [r for r in records if r['name'] in META_FILES]
    verified = verify_file_hashes(folder, metadata)
    cfg = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    validate_config(cfg)
    index = json.loads((folder/'model.safetensors.index.json').read_text(encoding='utf-8'))
    shard_names = sorted(set(index['weight_map'].values()))
    shards = [r for r in records if r['name'].endswith('.safetensors')]
    if shard_names != [r['name'] for r in shards]:
        raise RuntimeError('Weight index and ModelScope shard list disagree.')
    total = sum(r['bytes'] for r in shards)
    manifest = {'model_id': MODEL_ID, 'source': 'modelscope', 'revision': revision,
                'source_revision_detail': detail, 'content_manifest_sha256': identity,
                'content_files': records, 'verified_files': verified, 'config': cfg,
                'checkpoint_path': str(folder), 'shards': shards, 'download_bytes': total,
                'state_dict_bytes': index['metadata']['total_size'],
                'prepared_utc': utc_now(), 'weights_ready': False,
                'revision_semantics': 'branch/tag plus verified file hashes; NOT a Hugging Face commit SHA'}
    write_json(folder/'kite_manifest.json', manifest)
    show(f'ModelScope metadata ready: {folder}; weights require about {total/GIB:.1f} GiB.')
    if args.weights:
        missing = sum(r['bytes'] for r in shards if not (folder/r['name']).is_file()
                      or (folder/r['name']).stat().st_size != r['bytes'])
        if resource_snapshot(folder)['disk']['free_bytes'] < missing + 10*GIB:
            raise RuntimeError('Insufficient disk headroom for download plus 10 GiB reserve.')
        show('Downloading ModelScope BF16 weights with two workers. GPU is not used.')
        snapshot_download(MODEL_ID, revision=revision, local_dir=str(folder),
                          allow_patterns=shard_names, max_workers=2)
        # Verify once after download; no second full checkpoint copy is allocated.
        verified.update(verify_file_hashes(folder, shards))
        latest = normalize_modelscope_files(api.get_model_files(MODEL_ID, revision=revision,
                                                                 recursive=True, use_cookies=False))
        if content_identity(latest) != identity:
            raise RuntimeError('Repository changed during preparation. Files are retained, but not marked ready.')
        manifest.update({'verified_files': verified, 'weights_ready': True,
                         'weights_ready_utc': utc_now()})
        write_json(folder/'kite_manifest.json', manifest)
        show('ModelScope weights and hashes verified. No inference has been started.')
    print(json.dumps({'checkpoint': str(folder), 'source': 'modelscope', 'revision': revision,
                      'content_manifest_sha256': identity, 'weights_ready': manifest['weights_ready']},
                     ensure_ascii=False, indent=2))


def verify_checkpoint(folder):
    manifest = json.loads((folder/'kite_manifest.json').read_text(encoding='utf-8'))
    if manifest['model_id'] != MODEL_ID:
        raise RuntimeError('Checkpoint manifest is not the expected Qwen model.')
    source = manifest.get('source', 'huggingface')
    if source == 'huggingface':
        if not re.fullmatch(r'[0-9a-f]{40}', manifest['revision']):
            raise RuntimeError('Missing Hugging Face commit SHA.')
    elif source == 'modelscope':
        records = manifest.get('content_files', [])
        if not records or content_identity(records) != manifest.get('content_manifest_sha256'):
            raise RuntimeError('Invalid ModelScope content manifest.')
        for record in records:
            path = folder/record['name']
            stamp = manifest.get('verified_files', {}).get(record['name'], {})
            if not path.is_file() or path.stat().st_size != record['bytes'] or stamp.get('sha256') != record['sha256']:
                raise RuntimeError(f'Missing ModelScope hash verification for {record["name"]}.')
            if path.stat().st_mtime_ns != stamp.get('mtime_ns'):
                raise RuntimeError(f'File modified since verification: {record["name"]}; re-prepare with --weights.')
    else:
        raise RuntimeError('Unknown checkpoint source.')
    cfg = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    validate_config(cfg)
    if cfg != manifest['config']:
        raise RuntimeError('Config changed since model preparation.')
    if not manifest['weights_ready']:
        raise RuntimeError('Weights are not marked ready; run prepare --weights first.')
    for shard in manifest['shards']:
        path = folder/shard['name']
        if not path.is_file() or path.stat().st_size != shard['bytes']:
            raise RuntimeError(f'Missing/incomplete checkpoint shard: {shard["name"]}')
    return manifest


def attach_hooks(model, recorder, expert_input_observer=None):
    pattern = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)$')
    handles = []
    for name, module in model.named_modules():
        match = pattern.fullmatch(name)
        if match is None:
            continue
        layer, expert = map(int, match.groups())
        weight_bytes = sum(p.numel()*p.element_size() for p in module.parameters())
        placement = mapped_device(name, getattr(model, 'hf_device_map', {'': 'cpu'}))
        def hook(m, inputs, layer=layer, expert=expert, weight_bytes=weight_bytes, placement=placement):
            recorder.observe(layer, expert, inputs[0], weight_bytes, placement)
            if expert_input_observer is not None:
                expert_input_observer(layer, expert, inputs[0], recorder.context)
        handles.append(module.register_forward_pre_hook(hook))
    expected = model.config.num_hidden_layers * model.config.num_experts
    if len(handles) != expected:
        for handle in handles:
            handle.remove()
        raise RuntimeError(f'Expected {expected} expert hooks, found {len(handles)}.')
    return handles


def append_jsonl(file, values):
    for value in values:
        file.write(json.dumps(value, ensure_ascii=False) + '\n')
    file.flush()


def traced_forward(torch, model, ids, mask, cache, recorder, step, phase, files, device):
    before = int(cache.get_seq_length())
    query = int(ids.shape[1])
    recorder.begin(step, phase, query, before)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = model(input_ids=ids, attention_mask=mask, past_key_values=cache,
                   cache_position=torch.arange(before, before+query, device=device),
                   use_cache=True, output_router_logits=False, return_dict=True, logits_to_keep=1)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-start
    events, counts = recorder.finish()
    append_jsonl(files['experts'], events)
    append_jsonl(files['checks'], [counts])
    if not counts['pass']:
        raise RuntimeError(f'Routed-token conservation failed at step {step}.')
    cache = output.past_key_values
    kv = cache_snapshot(cache, model.config.num_hidden_layers, before+query,
                        model.config.num_key_value_heads, model.config.head_dim)
    append_jsonl(files['kv'], [{'step': step, 'phase': phase, **kv}])
    next_token = int(output.logits[0,-1].argmax().item())
    metrics = {'step': step, 'phase': phase, 'query_tokens': query, 'context_before': before,
               'context_after': before+query, 'next_token_id': next_token,
               'diagnostic_forward_seconds_with_hooks_and_offload': elapsed,
               'not_a_benchmark': True}
    if device.type == 'cuda':
        metrics.update({'cuda_allocated_bytes': torch.cuda.memory_allocated(device),
                        'cuda_reserved_bytes': torch.cuda.memory_reserved(device),
                        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
    append_jsonl(files['steps'], [metrics])
    show(f'{phase} step {step}: q={query}, context={before+query}, expert checks PASS; {elapsed:.2f}s diagnostic.')
    return cache, next_token


def generate_trace(torch, model, input_ids, new_tokens, eos_ids, outdir, device,
                   expert_input_observer=None, request_id='smoke-0001'):
    from transformers import DynamicCache
    recorder = ExpertRecorder(range(model.config.num_hidden_layers), model.config.num_experts_per_tok, request_id)
    handles = attach_hooks(model, recorder, expert_input_observer)
    files = {key: (outdir/name).open('x', encoding='utf-8') for key, name in
             [('experts','expert_trace.jsonl'),('checks','routing_checks.jsonl'),
              ('kv','kv_snapshots.jsonl'),('steps','steps.jsonl')]}
    cache = DynamicCache()
    ids = input_ids.to(device)
    mask = torch.ones_like(ids)
    generated = []
    reason = 'max_new_tokens'
    try:
        with torch.inference_mode():
            for step in range(new_tokens):
                phase = 'prefill' if step == 0 else 'decode'
                cache, token = traced_forward(torch, model, ids, mask, cache,
                                              recorder, step, phase, files, device)
                generated.append(token)
                if token in eos_ids:
                    reason = 'eos'
                    break
                ids = torch.tensor([[token]], dtype=torch.long, device=device)
                mask = torch.ones((1, int(cache.get_seq_length())+1), dtype=torch.long, device=device)
        return {'generated_token_ids': generated, 'stop_reason': reason,
                'forward_count': len(generated), 'decode_forward_count': max(0,len(generated)-1),
                'final_cache_length': int(cache.get_seq_length()),
                'note': 'The last sampled token is returned but not fed through a further forward.'}
    finally:
        for handle in handles:
            handle.remove()
        for file in files.values():
            file.close()


def run(args, expert_input_observer=None, request_runner=None):
    versions = require_versions()
    outroot = Path(args.results_dir).resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    outdir = outroot / (('tiny-' if args.tiny else 'qwen-') + stamp)
    outdir.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'started_utc': utc_now(), 'software': versions,
               'instrumented': True, 'is_tiny_synthetic_test': args.tiny,
               'is_kite_allocator': False, 'is_fallback_benchmark': False,
               'captures_expert_inputs': expert_input_observer is not None, 'args': vars(args)}
    # Do not store callables or credentials in logs.
    summary['args'].pop('handler', None)
    write_json(outdir/'summary.json', summary)
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3MoeConfig
        torch.manual_seed(17)
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
        if args.tiny:
            cfg = Qwen3MoeConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                moe_intermediate_size=16, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                num_experts=4, num_experts_per_tok=2, max_position_embeddings=128,
                                decoder_sparse_step=1, mlp_only_layers=[], use_cache=True)
            model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32, attn_implementation='sdpa').eval()
            ids = torch.tensor([[3,7,11,5]], dtype=torch.long)
            device = torch.device('cpu')
            eos_ids = set()
            summary['warning'] = 'Random tiny weights: code validation ONLY; not model evidence.'
            tokenizer = None
        else:
            if not args.checkpoint:
                raise RuntimeError('Supply the immutable local checkpoint path printed by prepare.')
            folder = Path(args.checkpoint).resolve()
            manifest = verify_checkpoint(folder)
            snapshot = resource_snapshot(folder)
            write_json(outdir/'resources_before.json', snapshot)
            check_gpu_budget(snapshot, args.gpu_memory_gib, allow_shared=args.allow_shared_gpu)
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('Expected one accessible CUDA GPU.')
            if torch.version.cuda != '11.8':
                raise RuntimeError('Use the already validated cu118 environment, not a replacement torch build.')
            device = torch.device('cuda:0')
            if args.cpu_affinity:
                cores = {int(x) for x in args.cpu_affinity.split(',')}
                if not hasattr(os, 'sched_getaffinity') or not cores <= os.sched_getaffinity(0):
                    raise RuntimeError('Requested CPU affinity is outside the allowed mask.')
                os.sched_setaffinity(0, cores)
            from accelerate import init_empty_weights, infer_auto_device_map
            cfg = AutoConfig.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
            show('Planning CPU/GPU placement with empty weights; disk offload is forbidden.')
            with init_empty_weights():
                empty = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
            mapping = infer_auto_device_map(empty, max_memory={0: f'{args.gpu_memory_gib}GiB',
                                                             'cpu': f'{args.cpu_memory_gib}GiB'},
                                            no_split_module_classes=empty._no_split_modules,
                                            dtype=torch.bfloat16)
            write_json(outdir/'device_map.json', {k:str(v) for k,v in mapping.items()})
            if any(str(v) == 'disk' for v in mapping.values()):
                raise RuntimeError('The placement requires disk offload; increase available resources rather than silently using disk.')
            cpu_bytes = sum(p.numel()*2 for name,p in empty.named_parameters()
                            if mapped_device(name, mapping) == 'cpu')
            available = snapshot['effective_available_bytes']
            if available is None or available < cpu_bytes + 8*GIB:
                raise RuntimeError('Insufficient host/cgroup memory headroom for CPU weights and loading reserve.')
            summary['cpu_weight_bytes_planned'] = cpu_bytes
            summary['model_revision'] = manifest['revision']
            summary['model_source'] = manifest.get('source', 'huggingface')
            summary['model_content_manifest_sha256'] = manifest.get('content_manifest_sha256')
            summary['shared_gpu_opt_in'] = args.allow_shared_gpu
            del empty
            gc.collect()
            show('Initializing CUDA before resetting memory statistics.')
            initialize_cuda_stats(torch, device)
            show('Loading BF16 checkpoint using CPU offload. No quantization or disk offload.')
            start = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(str(folder), config=cfg,
                       local_files_only=True, trust_remote_code=False, use_safetensors=True,
                       torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                       device_map=mapping, low_cpu_mem_usage=True, offload_state_dict=False).eval()
            torch.cuda.synchronize(device)
            summary['load_seconds'] = time.perf_counter()-start
            write_json(outdir/'model_config.json', cfg.to_dict())
            tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
            text = tokenizer.apply_chat_template([{'role':'user','content':args.prompt}],
                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids']
            if ids.shape[1] > args.max_prompt_tokens:
                raise RuntimeError('Prompt exceeds the smoke limit; it will not be silently truncated.')
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos,list) else [eos]) if eos is not None else set()
            write_json(outdir/'resources_after_load.json', resource_snapshot(folder))
            if request_runner is None:
                write_json(outdir/'request.json', {'request_id':'smoke-0001', 'source':'handwritten_smoke_prompt',
                        'prompt':args.prompt, 'rendered_text':text, 'input_token_ids':ids[0].tolist(),
                        'input_tokens':int(ids.shape[1]), 'enable_thinking':False,
                        'sampling':'greedy_for_code_smoke_not_a_quality_evaluation', 'seed':17})
        if request_runner is not None:
            if args.tiny:
                raise RuntimeError('Multi-request collection requires the real checkpoint.')
            summary['is_multi_request_trace'] = True
            write_json(outdir/'summary.json', summary)
            result = request_runner(torch, model, tokenizer, outdir, device, eos_ids)
        else:
            show(f'Forward trace starting: input={ids.shape[1]} tokens, output cap={args.new_tokens}.')
            result = generate_trace(torch, model, ids, args.new_tokens, eos_ids, outdir, device,
                                    expert_input_observer)
            if tokenizer is not None:
                result['generated_text'] = tokenizer.decode(result['generated_token_ids'], skip_special_tokens=True)
                print('Generated text:', result['generated_text'], flush=True)
        summary.update(result)
        if result['decode_forward_count'] < 1 and result.get('require_decode', True):
            raise RuntimeError('EOS ended the request before Decode. Prefill is recorded; change the smoke prompt before claiming P/D validation.')
        summary.update({'status':'passed', 'finished_utc':utc_now(),
                        'evidence_scope':result.get('evidence_scope',
                            'real_expert_invocations_and_KV_tensor_metadata' if not args.tiny else 'synthetic_adapter_test'),
                        'timing_scope':result.get('timing_scope',
                            'instrumented_diagnostic_only_not_TTFT_TPOT_or_speedup')})
    except BaseException as error:
        summary.update({'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                        'error_type':type(error).__name__, 'error':str(error), 'finished_utc':utc_now()})
        raise
    finally:
        write_json(outdir/'summary.json', summary)
        print('Results:', outdir, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check', help='Probe resources without importing torch or loading a model.')
    check.add_argument('--path', default='/workspace')
    cuda_probe = sub.add_parser('cuda-check', help='Test CUDA initialization and a one-value allocation, without a model.')
    cuda_probe.add_argument('--path', default='/workspace')
    prep = sub.add_parser('prepare', help='Fetch metadata, or explicitly download weights; never run a model.')
    prep.add_argument('--models-dir', default='/workspace/models')
    prep.add_argument('--source', choices=['huggingface', 'modelscope'], default='huggingface')
    prep.add_argument('--revision', help='HF: commit/main; ModelScope: branch/tag, defaults to master.')
    prep.add_argument('--weights', action='store_true', help='Download roughly 57 GiB of real checkpoint weights.')
    runner = sub.add_parser('run', help='Execute one short instrumented request, not a performance benchmark.')
    runner.add_argument('--checkpoint')
    runner.add_argument('--results-dir', default='/workspace/results/qwen_smoke')
    runner.add_argument('--prompt', default='请用一句话解释什么是缓存。')
    runner.add_argument('--new-tokens', type=int, default=8)
    runner.add_argument('--max-prompt-tokens', type=int, default=128)
    runner.add_argument('--gpu-memory-gib', type=int, default=24)
    runner.add_argument('--cpu-memory-gib', type=int, default=48)
    runner.add_argument('--cpu-threads', type=int, default=8)
    runner.add_argument('--cpu-affinity', help='Comma-separated allowed logical CPU IDs; optional.')
    runner.add_argument('--allow-shared-gpu', action='store_true', help='Explicit opt-in for trace only, never reliable timing.')
    runner.add_argument('--tiny', action='store_true', help='Random tiny CPU model: validate hooks without downloads or GPU.')
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(resource_snapshot(args.path), ensure_ascii=False, indent=2))
    elif args.command == 'cuda-check':
        cuda_check(args)
    elif args.command == 'prepare':
        prepare(args)
    else:
        if not 2 <= args.new_tokens <= 32 or not 1 <= args.cpu_threads <= 16:
            parser.error('Smoke limits: 2..32 output tokens, 1..16 CPU threads.')
        if args.gpu_memory_gib < 1 or args.cpu_memory_gib < 1 or args.max_prompt_tokens < 1:
            parser.error('Budgets and prompt limit must be positive.')
        run(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped by user; no external processes were terminated.', file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
