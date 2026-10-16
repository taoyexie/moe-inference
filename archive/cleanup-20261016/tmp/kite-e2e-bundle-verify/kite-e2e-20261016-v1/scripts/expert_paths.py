#!/usr/bin/env python3
"""Capture real Qwen activations, then time isolated expert recovery paths."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from kite_trace import GIB, check_gpu_budget, require_versions, resource_snapshot, utc_now, write_json
from qwen_smoke import initialize_cuda_stats, run as smoke_run, show, verify_checkpoint

PATHS = ('weight_h2d_only', 'gpu_hit', 'upload_gpu', 'cpu_return')
PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def new_output(root, kind):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    output = Path(root).resolve()/(kind+'-'+stamp)
    output.mkdir(parents=True, exist_ok=False)
    return output


def select_experts(reference, layer_ids):
    summary = json_file(reference/'summary.json')
    if summary.get('status') != 'passed' or summary.get('is_tiny_synthetic_test') is not False:
        raise RuntimeError('Reference must be a passed real-model smoke run.')
    request = json_file(reference/'request.json')
    best = {}
    with (reference/'expert_trace.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            layer = event['layer']
            if event['phase'] != 'prefill' or layer not in layer_ids:
                continue
            if event['request_id'] != request['request_id'] or event['step'] != 0:
                raise RuntimeError('Unexpected request/step in reference trace.')
            rank = (event['routed_tokens'], -event['expert'])
            if layer not in best or rank > (best[layer]['routed_tokens'], -best[layer]['expert']):
                best[layer] = event
    if set(best) != set(layer_ids):
        raise RuntimeError('Reference lacks a selected layer.')
    return summary, request, [best[layer] for layer in sorted(best)]


def capture(args):
    reference = Path(args.reference_run).resolve()
    folder = Path(args.checkpoint).resolve()
    manifest = verify_checkpoint(folder)
    layer_ids = parse_int_list(args.layers, 0, 47)
    summary, request, selections = select_experts(reference, layer_ids)
    if summary.get('model_content_manifest_sha256') != manifest.get('content_manifest_sha256'):
        raise RuntimeError('Reference and capture checkpoint identity differ.')
    if summary.get('model_source') != manifest.get('source') or summary.get('model_revision') != manifest['revision']:
        raise RuntimeError('Reference and capture checkpoint source/revision differ.')
    output = new_output(args.results_dir, 'capture')
    status = {'status': 'running', 'started_utc': utc_now(), 'reference_run': str(reference),
              'reference_trace_sha256': sha256_file(reference/'expert_trace.jsonl'),
              'model_content_manifest_sha256': manifest.get('content_manifest_sha256'),
              'model_source': manifest.get('source'), 'model_revision': manifest['revision'],
              'selection': 'Largest Prefill routed-token count in layers 0/23/47 by default; tie: lower expert ID.',
              'requested_layers': layer_ids, 'samples': [],
              'capture_timing_is_diagnostic': True}
    write_json(output/'capture.json', status)
    try:
        import torch
        from safetensors.torch import save_file
        selected = {(e['layer'], e['expert']): e for e in selections}
        captured = {}

        def observe(layer, expert, tensor, context):
            key = (layer, expert)
            if key not in selected or key in captured or context['phase'] != 'prefill' or not tensor.shape[0]:
                return
            if tensor.device.type != 'cuda' or tensor.dtype != torch.bfloat16:
                raise RuntimeError('Expected real BF16 CUDA expert input.')
            if tuple(tensor.shape) != (selected[key]['routed_tokens'], 2048):
                raise RuntimeError('Routing differs from reference; retain logs and investigate, not silently resample.')
            captured[key] = tensor.detach().to('cpu').clone().contiguous()

        run_args = SimpleNamespace(command='run', checkpoint=str(folder), results_dir=str(output/'trace'),
                                  prompt=request['prompt'], new_tokens=2, max_prompt_tokens=128,
                                  gpu_memory_gib=args.gpu_memory_gib, cpu_memory_gib=args.cpu_memory_gib,
                                  cpu_threads=args.cpu_threads, cpu_affinity=args.cpu_affinity,
                                  allow_shared_gpu=False, tiny=False)
        show('Capturing three real expert inputs during one short full-model inference.')
        smoke_run(run_args, expert_input_observer=observe)
        if set(captured) != set(selected):
            raise RuntimeError('Some selected expert inputs were not captured.')
        traces = list((output/'trace').glob('qwen-*'))
        if len(traces) != 1 or json_file(traces[0]/'summary.json')['status'] != 'passed':
            raise RuntimeError('Capture inference did not pass.')
        status['capture_trace'] = str(traces[0])
        status['request_sha256'] = sha256_file(traces[0]/'request.json')
        for key, tensor in sorted(captured.items()):
            layer, expert = key
            name = f'layer-{layer:02d}-expert-{expert:03d}.safetensors'
            save_file({'input': tensor}, str(output/name))
            status['samples'].append({'layer': layer, 'expert': expert, 'file': name,
                                      'sha256': sha256_file(output/name),
                                      'native_routed_tokens': int(tensor.shape[0]),
                                      'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                                      'phase': 'prefill', 'step': 0,
                                      'input_source': 'actual_nonempty_expert_input_before_computation'})
        status['status'] = 'passed'
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status['finished_utc'] = utc_now()
        write_json(output/'capture.json', status)
        print('Capture directory:', output, flush=True)


def parse_int_list(value, minimum, maximum):
    numbers = [int(item.strip()) for item in value.split(',')]
    if not numbers or len(set(numbers)) != len(numbers) or not all(minimum <= n <= maximum for n in numbers):
        raise ValueError(f'Expected distinct integers in [{minimum}, {maximum}].')
    return sorted(numbers)


def input_source(q, native):
    if q == native:
        return 'native_captured_input'
    if q < native:
        return 'first_q_rows_of_captured_input_controlled_size'
    return 'cyclic_repeat_of_captured_rows_controlled_size_not_new_real_routing'


def quantile(values, p):
    ordered = sorted(values)
    position = (len(ordered)-1)*p
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)


def summarize(values):
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('Cannot summarize missing/nonfinite timings.')
    return {'n': len(values), 'median_ms': quantile(values, .5),
            'p10_ms': quantile(values, .1), 'p90_ms': quantile(values, .9),
            'mean_ms': sum(values)/len(values)}


def timed(torch, device, operation):
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    result = operation()
    torch.cuda.synchronize(device)
    return (time.perf_counter_ns()-started)/1e6, result


def expert_shards(index, sample):
    prefix = f"model.layers.{sample['layer']}.mlp.experts.{sample['expert']}."
    mapping = {}
    for projection in PROJECTIONS:
        name = prefix+projection+'.weight'
        shard = index['weight_map'][name]
        if Path(shard).name != shard or '/' in shard or '\\' in shard or not shard.endswith('.safetensors'):
            raise RuntimeError('Unexpected shard path.')
        mapping[projection] = (name, shard)
    return mapping


def load_expert(torch, safe_open, cfg, folder, index, sample, device):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP
    # Actual pinned Qwen expert class, only three tensors loaded; no full-model allocation.
    weights = {}
    provenance = []
    for projection, (name, shard) in expert_shards(index, sample).items():
        with safe_open(str(folder/shard), framework='pt', device='cpu') as handle:
            tensor = handle.get_tensor(name)
            shape = (2048, 768) if projection == 'down_proj' else (768, 2048)
            if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
                raise RuntimeError('Checkpoint expert shape/precision mismatch.')
            weights[projection+'.weight'] = tensor.clone().contiguous().pin_memory()
        provenance.append({'tensor': name, 'shard': shard,
                           'tensor_bf16_sha256': hashlib.sha256(weights[projection+'.weight'].view(torch.uint8).numpy().tobytes()).hexdigest()})
    cpu_model = Qwen3MoeMLP(cfg, intermediate_size=768).to(dtype=torch.float32).eval()
    gpu_model = Qwen3MoeMLP(cfg, intermediate_size=768).to(device=device, dtype=torch.bfloat16).eval()
    cpu_model.load_state_dict(weights, strict=True)  # BF16 values promoted to FP32 outside timing.
    gpu_model.load_state_dict(weights, strict=True)
    return cpu_model, gpu_model, weights, provenance


def error_metrics(torch, output, reference):
    actual = output.detach().float()
    target = reference.detach().float()
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(target).all()):
        raise RuntimeError('Nonfinite expert output.')
    delta = actual-target
    return {'relative_l2': float((torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(target).clamp_min(1e-12)).item()),
            'max_abs': float(delta.abs().max().item()),
            'normalized_max_abs': float((delta.abs().max()/target.abs().max().clamp_min(1e-12)).item())}


def benchmark_sample(torch, safe_open, cfg, folder, index, capture_dir, sample, args, files, rng, device):
    from safetensors.torch import load_file
    name = sample['file']
    if Path(name).name != name or '/' in name or '\\' in name:
        raise RuntimeError('Unexpected activation path.')
    if sha256_file(capture_dir/name) != sample['sha256']:
        raise RuntimeError('Captured activation hash mismatch.')
    activation = load_file(str(capture_dir/name), device='cpu')['input']
    native = sample['native_routed_tokens']
    if tuple(activation.shape) != (native, 2048) or activation.dtype != torch.bfloat16:
        raise RuntimeError('Captured activation shape/dtype mismatch.')
    cpu_model, gpu_model, weights, provenance = load_expert(torch, safe_open, cfg, folder, index, sample, device)
    files['samples'].append({**sample, 'weight_bytes_bf16': 9*1024**2, 'weight_tensors': provenance,
                              'persistent_cpu_fp32_weight_bytes': 18*1024**2})
    points = sorted(set(parse_int_list(args.q, 1, 128) + [native]))
    rng.shuffle(points)
    results = []
    with torch.inference_mode():
        for q in points:
            indices = torch.arange(q) % native
            x_gpu = activation.index_select(0, indices).to(device)
            x_host = torch.empty((q, 2048), dtype=torch.bfloat16, pin_memory=True)
            y_host = torch.empty_like(x_host, pin_memory=True)
            y_gpu = torch.empty_like(x_gpu)

            def h2d():
                for projection in PROJECTIONS:
                    getattr(gpu_model, projection).weight.copy_(weights[projection+'.weight'], non_blocking=True)

            def hit():
                y_gpu.copy_(gpu_model(x_gpu))
                return y_gpu

            def upload():
                h2d()
                return hit()

            def cpu_return():
                x_host.copy_(x_gpu, non_blocking=True)
                torch.cuda.synchronize(device)  # Complete D2H before CPU reads.
                x_fp32 = x_host.to(torch.float32)
                y_host.copy_(cpu_model(x_fp32))  # Includes output FP32->BF16 conversion.
                y_gpu.copy_(y_host, non_blocking=True)
                return y_gpu

            functions = dict(zip(PATHS, (h2d, hit, upload, cpu_return)))
            timing = defaultdict(list)
            reference = gpu_model(x_gpu).clone()
            all_errors = []
            for warm in range(args.warmup):
                order = list(PATHS)
                rng.shuffle(order)
                for path in order:
                    timed(torch, device, functions[path])
            for trial in range(args.repeats):
                order = list(PATHS)
                rng.shuffle(order)
                paired = {}
                for path in order:
                    milliseconds, value = timed(torch, device, functions[path])
                    metrics = None if path == 'weight_h2d_only' else error_metrics(torch, value, reference)
                    correct = metrics is None or (metrics['relative_l2'] <= args.relative_l2_tolerance
                                                  and metrics['normalized_max_abs'] <= args.normalized_max_tolerance)
                    row = {'layer': sample['layer'], 'expert': sample['expert'], 'q': q,
                           'native_q': native, 'input_source': input_source(q, native),
                           'trial': trial, 'order': order.index(path), 'path': path,
                           'wall_ms': milliseconds, 'output_error': metrics, 'correctness_pass': correct}
                    files['raw'].write(json.dumps(row)+'\n')
                    files['raw'].flush()
                    if not correct:
                        raise RuntimeError(f'Output mismatch layer={sample["layer"]}, q={q}, path={path}: {metrics}')
                    if metrics is not None:
                        all_errors.append(metrics)
                    paired[path] = milliseconds
                    timing[path].append(milliseconds)
                for path in ('upload_gpu', 'cpu_return'):
                    timing[path+'_minus_hit'].append(paired[path]-paired['gpu_hit'])
            point = {'layer': sample['layer'], 'expert': sample['expert'], 'q': q, 'native_q': native,
                     'input_source': input_source(q, native),
                     'timings': {path: summarize(values) for path, values in timing.items()},
                     'worst_relative_l2': max(e['relative_l2'] for e in all_errors),
                     'worst_normalized_max_abs': max(e['normalized_max_abs'] for e in all_errors)}
            files['points'].write(json.dumps(point)+'\n')
            files['points'].flush()
            results.append(point)
            show(f'L{sample["layer"]} E{sample["expert"]} q={q}: upload+GPU={point["timings"]["upload_gpu"]["median_ms"]:.3f}ms, CPU+return={point["timings"]["cpu_return"]["median_ms"]:.3f}ms; correctness PASS')
    return results


def benchmark(args):
    versions = require_versions()
    folder = Path(args.checkpoint).resolve()
    capture_dir = Path(args.capture_dir).resolve()
    manifest = verify_checkpoint(folder)
    captured = json_file(capture_dir/'capture.json')
    if captured['status'] != 'passed' or len(captured['samples']) < 1:
        raise RuntimeError('Capture is not complete.')
    if captured['model_content_manifest_sha256'] != manifest.get('content_manifest_sha256'):
        raise RuntimeError('Capture/checkpoint identity mismatch.')
    if captured['model_source'] != manifest.get('source') or captured['model_revision'] != manifest['revision']:
        raise RuntimeError('Capture/checkpoint source/revision mismatch.')
    snapshot = resource_snapshot(folder)
    check_gpu_budget(snapshot, 1, reserve_gib=1)
    if snapshot['effective_available_bytes'] is None or snapshot['effective_available_bytes'] < 2*GIB:
        raise RuntimeError('Need at least 2 GiB effective host headroom.')
    output = new_output(args.results_dir, 'expert-paths')
    status = {'status': 'running', 'started_utc': utc_now(), 'software': versions,
              'args': vars(args).copy(), 'model_content_manifest_sha256': manifest.get('content_manifest_sha256'),
              'capture_manifest_sha256': sha256_file(capture_dir/'capture.json'),
              'resources_before': snapshot, 'samples': [], 'is_end_to_end_inference': False,
              'protocol': {'start': 'BF16 expert input ready on GPU, source weights ready in pinned host RAM',
                           'end': 'BF16 expert output ready in preallocated GPU buffer; device synchronized',
                           'gpu_compute_dtype': 'BF16', 'cpu_compute_dtype': 'FP32',
                           'cpu_weight_conversion': 'once outside timing; persistent 18 MiB/expert FP32 replica',
                           'allocation': 'weight staging and I/O buffers preallocated; module intermediate/output allocation inside timing on both paths',
                           'weight_h2d_only': 'diagnostic transfer time only, not complete expert recovery',
                           'cpu_return': 'input D2H + BF16->FP32 + expert + FP32->BF16 + output H2D',
                           'upload_gpu': 'three BF16 weight H2D copies + actual pinned Qwen expert + GPU output copy',
                           'extra_penalty': 'same-trial full path minus gpu_hit; raw differences retained, never clamped',
                           'activation_sizes': 'native q plus controlled prefix/cyclic resizing; not new real routed batches',
                           'repeat_unit': 'trials in one process; independent sessions require separate invocations',
                           'overlap': 'none; no exposed-latency claim',
                           'cache_condition': 'repeated hot weights/activation microbenchmark; not cold CPU LLC/DRAM replay'}}
    write_json(output/'summary.json', status)
    files = {}
    try:
        import torch
        from safetensors import safe_open
        from transformers import Qwen3MoeConfig
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.version.cuda != '11.8':
            raise RuntimeError('Expected one CUDA GPU in the validated cu118 environment.')
        if args.cpu_affinity:
            cores = set(parse_int_list(args.cpu_affinity, 0, 4095))
            if not cores <= os.sched_getaffinity(0):
                raise RuntimeError('CPU affinity exceeds allowed CPUs.')
            os.sched_setaffinity(0, cores)
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device('cuda:0')
        initialize_cuda_stats(torch, device)
        status['effective_cpu_affinity'] = sorted(os.sched_getaffinity(0))
        cfg = Qwen3MoeConfig.from_dict(json_file(folder/'config.json'))
        index = json_file(folder/'model.safetensors.index.json')
        rng = random.Random(args.seed)
        files = {'raw': (output/'trials.jsonl').open('x', encoding='utf-8'),
                 'points': (output/'points.jsonl').open('x', encoding='utf-8'), 'samples': status['samples']}
        points = []
        samples = list(captured['samples'])
        rng.shuffle(samples)
        for sample in samples:
            points.extend(benchmark_sample(torch, safe_open, cfg, folder, index, capture_dir,
                                           sample, args, files, rng, device))
            gc.collect()
        write_json(output/'point_summary.json', sorted(points, key=lambda r: (r['layer'], r['q'])))
        status.update(status='passed', point_count=len(points),
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      resources_after=resource_snapshot(folder))
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for key in ('raw', 'points'):
            if key in files:
                files[key].close()
        status['finished_utc'] = utc_now()
        write_json(output/'summary.json', status)
        print('Benchmark directory:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cap = sub.add_parser('capture', help='One full-model short inference; capture three real expert inputs.')
    cap.add_argument('--checkpoint', required=True)
    cap.add_argument('--reference-run', required=True)
    cap.add_argument('--results-dir', default='/workspace/results/expert_inputs')
    cap.add_argument('--layers', default='0,23,47')
    cap.add_argument('--gpu-memory-gib', type=int, default=24)
    cap.add_argument('--cpu-memory-gib', type=int, default=48)
    cap.add_argument('--cpu-threads', type=int, default=8)
    cap.add_argument('--cpu-affinity', default='24,25,26,27,28,29,30,31')
    bench = sub.add_parser('bench', help='Small isolated real-weight microbenchmark, not full inference.')
    bench.add_argument('--checkpoint', required=True)
    bench.add_argument('--capture-dir', required=True)
    bench.add_argument('--results-dir', default='/workspace/results/expert_paths')
    bench.add_argument('--q', default='1,2,4,8,16,32,64,128')
    bench.add_argument('--warmup', type=int, default=5)
    bench.add_argument('--repeats', type=int, default=30)
    bench.add_argument('--cpu-threads', type=int, default=8)
    bench.add_argument('--cpu-affinity', default='24,25,26,27,28,29,30,31')
    bench.add_argument('--seed', type=int, default=17)
    bench.add_argument('--relative-l2-tolerance', type=float, default=.03)
    bench.add_argument('--normalized-max-tolerance', type=float, default=.05)
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 16:
        parser.error('CPU thread count must be 1..16.')
    if args.command == 'capture':
        if args.gpu_memory_gib < 1 or args.cpu_memory_gib < 1:
            parser.error('Memory budgets must be positive.')
        capture(args)
    else:
        parse_int_list(args.q, 1, 128)
        if not 1 <= args.warmup <= 20 or not 2 <= args.repeats <= 200:
            parser.error('Warmup must be 1..20; repeats must be 2..200.')
        if not 0 < args.relative_l2_tolerance <= .03 or not 0 < args.normalized_max_tolerance <= .05:
            parser.error('Do not loosen the preregistered correctness limits.')
        benchmark(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; partial records retained. No other processes were stopped.', file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
