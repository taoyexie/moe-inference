"""Physical expert-cache acceptance test, NOT full inference or allocator speedup.

Uses three previously captured real expert inputs and checkpoint weights. The
invocation order and q resizing are controlled test stimuli, not a serving trace.
Only these three experts are loaded, not Qwen's complete weights.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

from expert_cache import ExpertCache, fit_costs
from expert_paths import (PROJECTIONS, error_metrics, expert_shards, input_source,
                          new_output, sha256_file, timed)
from kite_trace import GIB, check_gpu_budget, require_versions, resource_snapshot, utc_now, write_json
from qwen_smoke import initialize_cuda_stats, show, verify_checkpoint

OBJECT_BYTES = 9*1024**2


def schedule(keys):
    """Explicitly synthetic order over actual captured experts; deterministic."""
    if len(keys) != 3 or len(set(keys)) != 3:
        raise ValueError('This acceptance test requires three distinct captured experts.')
    a,b,c = keys
    return [(a,None),(b,None),(a,None),(c,None),(b,None),(a,None),(a,None),
            (c,1),(a,8),(b,32),(b,32),(c,8)]


def read_capture(folder, manifest):
    capture = json.loads((folder/'capture.json').read_text(encoding='utf-8'))
    if (capture['status'] != 'passed' or len(capture['samples']) != 3
            or capture['model_content_manifest_sha256'] != manifest.get('content_manifest_sha256')
            or capture['model_source'] != manifest.get('source')
            or capture['model_revision'] != manifest['revision']):
        raise RuntimeError('Need the passed three-expert capture matching this checkpoint.')
    identities = []
    for sample in capture['samples']:
        if not (type(sample['layer']) is int and 0 <= sample['layer'] < 48
                and type(sample['expert']) is int and 0 <= sample['expert'] < 128):
            raise RuntimeError('Invalid captured expert identity.')
        name = sample['file']
        if Path(name).name != name or '/' in name or '\\' in name:
            raise RuntimeError('Unexpected captured activation filename.')
        if sha256_file(folder/name) != sample['sha256']:
            raise RuntimeError('Captured activation hash differs.')
        identities.append((sample['layer'],sample['expert']))
    if len(set(identities)) != 3:
        raise RuntimeError('Duplicate captured expert identity.')
    return capture


class TorchBackend:
    def __init__(self, torch, cfg, checkpoint, capture_dir, capture, device, max_q=128):
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP
        self.torch, self.device, self.cfg = torch, device, cfg
        self.module_type = Qwen3MoeMLP
        self.sources = {}
        index = json.loads((checkpoint/'model.safetensors.index.json').read_text(encoding='utf-8'))
        for sample in capture['samples']:
            key = (sample['layer'],sample['expert'])
            weights, provenance = {}, []
            for projection, (name, shard) in expert_shards(index,sample).items():
                with safe_open(str(checkpoint/shard),framework='pt',device='cpu') as handle:
                    tensor = handle.get_tensor(name)
                    shape = (2048,768) if projection == 'down_proj' else (768,2048)
                    if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
                        raise RuntimeError('Unexpected checkpoint expert tensor.')
                    weights[projection+'.weight'] = tensor.clone().contiguous().pin_memory()
                provenance.append({'name':name,'shard':shard})
            cpu = Qwen3MoeMLP(cfg,intermediate_size=768).to(dtype=torch.float32).eval().requires_grad_(False)
            cpu.load_state_dict(weights,strict=True)
            activation = load_file(str(capture_dir/sample['file']),device='cpu')['input']
            if (tuple(activation.shape) != (sample['native_routed_tokens'],2048)
                    or activation.dtype != torch.bfloat16 or not 1 <= activation.shape[0] <= max_q):
                raise RuntimeError('Unexpected real captured activation.')
            self.sources[key] = dict(weights=weights,cpu=cpu,activation=activation,sample=sample,
                                     provenance=provenance)
        self.max_q = max_q
        self.x_host = torch.empty((max_q,2048),dtype=torch.bfloat16,pin_memory=True)
        self.y_host = torch.empty_like(self.x_host,pin_memory=True)
        self.y_gpu = torch.empty((max_q,2048),dtype=torch.bfloat16,device=device)
        self.scratch = self.allocate_slot()

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def allocate_slot(self):
        slot = self.module_type(self.cfg,intermediate_size=768).to(
            device=self.device,dtype=self.torch.bfloat16).eval().requires_grad_(False)
        if sum(p.numel()*p.element_size() for p in slot.parameters()) != OBJECT_BYTES:
            raise RuntimeError('GPU expert slot weight bytes differ.')
        return slot

    def release_slot(self, slot):
        # Explicitly release tensor references even if the Python module is retained.
        for projection in PROJECTIONS:
            getattr(slot,projection).weight = None

    def upload(self, key, slot):
        for projection in PROJECTIONS:
            getattr(slot,projection).weight.copy_(self.sources[key]['weights'][projection+'.weight'],
                                                  non_blocking=True)

    def gpu(self, slot, x):
        y = self.y_gpu[:x.shape[0]]
        y.copy_(slot(x))
        return y

    def cpu(self, key, x):
        q = x.shape[0]
        if not 1 <= q <= self.max_q:
            raise RuntimeError('q exceeds preallocated I/O workspace.')
        self.x_host[:q].copy_(x,non_blocking=True)
        self.synchronize()  # CPU may read input only after D2H completes.
        self.y_host[:q].copy_(self.sources[key]['cpu'](self.x_host[:q].float()))
        self.y_gpu[:q].copy_(self.y_host[:q],non_blocking=True)
        return self.y_gpu[:q]

    def scratch_upload(self, key, x):
        self.upload(key,self.scratch)
        return self.gpu(self.scratch,x)

    def activation(self, key, q):
        native = self.sources[key]['activation']
        q = native.shape[0] if q is None else q
        if not 1 <= q <= self.max_q:
            raise RuntimeError('Activation exceeds declared workspace.')
        indices = self.torch.arange(q) % native.shape[0]
        return native.index_select(0,indices).to(self.device), {
            'native_q':int(native.shape[0]),'input_source':input_source(q,int(native.shape[0]))}

    def validate(self, key, x, output):
        # Shared output buffer is overwritten by scratch reference; retain actual first.
        actual = output.clone()
        reference = self.scratch_upload(key,x).clone()
        self.synchronize()
        error = error_metrics(self.torch,actual,reference)
        if error['relative_l2'] > .02 or error['normalized_max_abs'] > .03:
            raise RuntimeError(f'Expert-cache numerical acceptance failed: {error}')
        return error


def calibrate(backend, key, output, warmup=3, repeats=5):
    """One startup calibration for the whole isolated acceptance job."""
    rng, rows = random.Random(17), []
    started = time.perf_counter_ns()
    with (output/'calibration.jsonl').open('x',encoding='utf-8') as stream:
        for q in (1,8,32):
            x, provenance = backend.activation(key,q)
            for trial in range(-warmup,repeats):
                paths = ['upload','cpu']
                rng.shuffle(paths)
                for path in paths:
                    operation = (lambda:backend.scratch_upload(key,x)) if path == 'upload' else (lambda:backend.cpu(key,x))
                    wall_ms, value = timed(backend.torch,backend.device,operation)
                    error = backend.validate(key,x,value)
                    row = dict(path=path,q=q,wall_ms=wall_ms,trial=trial,warmup=trial<0,
                               key=key,output_error=error,**provenance)
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    if trial >= 0:
                        rows.append(row)
            del x
    costs = fit_costs(rows)
    return costs, {'expert_only':True,'four_path_calibration_complete':False,
                   'samples':len(rows),'costs':asdict(costs),
                   'total_wall_ms_including_warmup_validation_and_fit':(time.perf_counter_ns()-started)/1e6,
                   'domain_policy':'No extrapolation: an out-of-domain q uses upload.',
                   'first_captured_expert_only':list(key)}


def run_scenarios(backend, costs, output, capacity):
    plan = schedule(sorted(backend.sources))
    results = []
    with (output/'events.jsonl').open('x',encoding='utf-8') as raw:
        for label, policy in [('upload_lru','upload'),('cpu_bypass','cpu'),
                              ('fixed_cost','fixed'),('affine_cost','calibrated'),
                              ('resize_acceptance','upload')]:
            started = time.perf_counter_ns()
            cache = ExpertCache(backend,OBJECT_BYTES,capacity,policy,costs)
            path_total, resize_rows = 0., []
            try:
                for position,(key,q) in enumerate(plan):
                    if label == 'resize_acceptance' and position in (3,5,6,7):
                        target = {3:min(1,capacity),5:capacity,6:0,7:capacity}[position]
                        tick = time.perf_counter_ns()
                        resize_rows.append({**cache.resize(target),'position':position,
                                            'wall_ms':(time.perf_counter_ns()-tick)/1e6})
                    x, provenance = backend.activation(key,q)
                    value, row = cache.execute(key,int(x.shape[0]),x)
                    error = backend.validate(key,x,value)
                    row.update(scenario=label,position=position,**provenance,output_error=error,
                               correctness_pass=True,
                               scratch_weight_bytes=OBJECT_BYTES,
                               torch_active_bytes_after_validation=backend.torch.cuda.memory_allocated(backend.device))
                    raw.write(json.dumps(row)+'\n')
                    raw.flush()
                    path_total += row['wall_ms']
                    del x,value
                counters = dict(cache.counts)
                if counters.get('calls',0) != len(plan) or counters.get('hits',0)+counters.get('misses',0) != len(plan):
                    raise RuntimeError('Physical hit/miss totals do not reconcile.')
                if policy == 'cpu' and (cache.resident or counters.get('hits',0)):
                    raise RuntimeError('CPU bypass unexpectedly admitted a GPU expert.')
                show(f'{label}: calls={len(plan)} hits={counters.get("hits",0)} misses={counters.get("misses",0)}; correctness PASS')
            finally:
                cache.close()
            results.append(dict(scenario=label,counters=counters,resize=resize_rows,
                invocation_wall_ms_sum=path_total,
                scenario_wall_ms_including_setup_validation_and_teardown=(time.perf_counter_ns()-started)/1e6))
            gc.collect()
    return results


def run(args):
    started = time.perf_counter_ns()
    versions = require_versions()
    checkpoint,capture_dir = Path(args.checkpoint).resolve(),Path(args.capture_dir).resolve()
    manifest = verify_checkpoint(checkpoint)
    captured = read_capture(capture_dir,manifest)
    snapshot = resource_snapshot(checkpoint)
    check_gpu_budget(snapshot,1,reserve_gib=1)
    if snapshot['effective_available_bytes'] is None or snapshot['effective_available_bytes'] < 2*GIB:
        raise RuntimeError('Need at least 2 GiB effective host headroom.')
    output = new_output(args.results_dir,'expert-cache')
    status = dict(status='running',started_utc=utc_now(),software=versions,args=vars(args),
                  model_content_manifest_sha256=manifest.get('content_manifest_sha256'),
                  capture_manifest_sha256=sha256_file(capture_dir/'capture.json'),resources_before=snapshot,
                  is_full_model_inference=False,is_kite_allocator=False,is_end_to_end_speedup=False,
                  stimulus='synthetic deterministic expert order; real captured inputs, controlled q resizing',
                  timing='synchronized GPU-input-to-GPU-output expert invocation including cache lookup/admission; not TTFT',
                  precision='BF16 GPU; persistent FP32 CPU experts as in prior path benchmark',
                  cache_budget_bytes=OBJECT_BYTES*args.capacity_experts,
                  additional_scratch_weight_bytes=OBJECT_BYTES,
                  persistent_cpu_weights_bf16_bytes=3*OBJECT_BYTES,
                  persistent_cpu_weights_fp32_bytes=6*OBJECT_BYTES,
                  resize_semantics='shrink evicts LRU and releases weight tensors; growth allocates empty slots',
                  memory_note='cache weight budget excludes separately recorded scratch, I/O and activation workspaces; CUDA reserved memory is not resident expert bytes',
                  comparison_note='acceptance scenarios, not a natural trace or WISP ablation; validation uses scratch upload after every call outside invocation timing')
    write_json(output/'summary.json',status)
    try:
        import torch
        from transformers import Qwen3MoeConfig
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.version.cuda != '11.8':
            raise RuntimeError('Expected the validated single CUDA GPU and cu118 environment.')
        cores = set(range(24,32))
        if not cores <= os.sched_getaffinity(0):
            raise RuntimeError('GPU-near CPU affinity is unavailable.')
        os.sched_setaffinity(0,cores)
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        device = torch.device('cuda:0')
        initialize_cuda_stats(torch,device)
        cfg = Qwen3MoeConfig.from_pretrained(str(checkpoint),local_files_only=True)
        show('Loading only three real experts and their captured inputs; no complete-model inference.')
        with torch.inference_mode():
            backend = TorchBackend(torch,cfg,checkpoint,capture_dir,captured,device)
            status['weights'] = [{'key':list(key),'tensors':source['provenance']}
                                 for key,source in backend.sources.items()]
            show('Calibrating expert upload constant and CPU affine cost once for this acceptance job.')
            costs, calibration = calibrate(backend,sorted(backend.sources)[0],output)
            write_json(output/'calibration.json',calibration)
            status['scenarios'] = run_scenarios(backend,costs,output,args.capacity_experts)
            status['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            status['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        status['status'] = 'passed'
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__,error=str(error))
        raise
    finally:
        status['finished_utc'] = utc_now()
        status['complete_job_wall_ms'] = (time.perf_counter_ns()-started)/1e6
        write_json(output/'summary.json',status)
        print('Results:',output,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--capture-dir',required=True)
    parser.add_argument('--capacity-experts',type=int,default=2)
    parser.add_argument('--results-dir',default='/workspace/results/expert_cache')
    args = parser.parse_args()
    if not 0 <= args.capacity_experts <= 16:
        parser.error('Capacity must be 0..16 experts for this small acceptance test.')
    run(args)


if __name__ == '__main__':
    main()
