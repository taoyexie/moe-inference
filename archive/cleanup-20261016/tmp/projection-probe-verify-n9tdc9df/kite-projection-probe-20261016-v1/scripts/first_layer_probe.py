#!/usr/bin/env python3
"""Real first-layer-only precision probe; no MoE/model load or performance claim."""
import argparse
import hashlib
import json
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from expert_paths import error_metrics,new_output
from kite_e2e import ResourceMonitor
from kite_e2e_core import Request,packed_layout
from kite_e2e_torch import batch_tensors,make_packed_cache
from kite_readiness import verify_source_bundle
from kite_trace import GIB,check_gpu_budget,parse_gpu_pids,require_versions,resource_snapshot,run_readonly,write_json
from qwen_smoke import initialize_cuda_stats,show,verify_checkpoint


@contextmanager
def bf16_reduction(torch,enabled):
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


def required_keys(index,token_ids):
    prefix = 'model.layers.0.self_attn.'
    keys = [name for name in index['weight_map'] if name.startswith(prefix)]
    if not keys or not token_ids or any(type(t) is not int or t<0 for t in token_ids):
        raise RuntimeError('Invalid checkpoint index or diagnostic token IDs.')
    return keys+['model.layers.0.input_layernorm.weight','model.embed_tokens.weight']


@contextmanager
def fp32_projection_control(torch,attention):
    """Diagnostic-only operator oracle. Restore every module and global flag."""
    saved = []
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            module = getattr(attention,name)
            saved.append((module,module.forward))
            weight = module.weight.detach().float()
            bias = module.bias.detach().float() if module.bias is not None else None
            def forward(x,weight=weight,bias=bias):
                return torch.nn.functional.linear(x.float(),weight,bias).to(x.dtype)
            module.forward = forward
        yield
    finally:
        for module,forward in saved:
            module.forward = forward
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def load_first_layer(torch,checkpoint,token_ids,device):
    """Selective safetensors access: attention weights and embedding rows only."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention,Qwen3MoeRMSNorm,Qwen3MoeRotaryEmbedding
    index = json.loads((checkpoint/'model.safetensors.index.json').read_text(encoding='utf-8'))
    keys = required_keys(index,token_ids)
    config = AutoConfig.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    if (config.hidden_size,config.num_attention_heads,config.num_key_value_heads,config.head_dim)!=(2048,32,4,128):
        raise RuntimeError('Expected fixed Qwen3-30B-A3B attention dimensions.')
    config._attn_implementation = 'sdpa'
    state,embeddings = {},{}
    loaded_bytes = 0
    for shard in sorted({index['weight_map'][key] for key in keys}):
        path = (checkpoint/shard).resolve()
        if not path.is_relative_to(checkpoint.resolve()):
            raise RuntimeError('Unsafe shard path.')
        with safe_open(str(path),framework='pt',device='cpu') as handle:
            for name in keys:
                if index['weight_map'][name]!=shard:
                    continue
                if name=='model.embed_tokens.weight':
                    sliced = handle.get_slice(name)
                    if max(token_ids)>=sliced.get_shape()[0]:
                        raise RuntimeError('Token outside embedding vocabulary.')
                    for token in sorted(set(token_ids)):
                        row = sliced[token:token+1].clone()
                        embeddings[token] = row.to(device)
                        loaded_bytes += row.numel()*row.element_size()
                else:
                    value = handle.get_tensor(name)
                    if value.dtype!=torch.bfloat16:
                        raise RuntimeError('Expected real BF16 checkpoint tensor.')
                    state[name] = value
                    loaded_bytes += value.numel()*value.element_size()
    with init_empty_weights():
        attention = Qwen3MoeAttention(config,layer_idx=0)
        norm = Qwen3MoeRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
    attention.load_state_dict({key.removeprefix('model.layers.0.self_attn.'):value
        for key,value in state.items() if '.self_attn.' in key},strict=True,assign=True)
    norm.load_state_dict({'weight':state['model.layers.0.input_layernorm.weight']},strict=True,assign=True)
    attention.to(device).eval().requires_grad_(False)
    norm.to(device).eval().requires_grad_(False)
    rotary = Qwen3MoeRotaryEmbedding(config).to(device)
    return attention,norm,rotary,embeddings,dict(checkpoint_bytes_read=loaded_bytes,
        unique_embedding_rows=len(embeddings),loaded_parameter_names=keys,
        scope='first attention layer only; no experts, full embedding table, LM head or full model')


def probe(torch,attention,norm,rotary,embeddings,inputs,device):
    from transformers import DynamicCache
    prefix,token,prompt = inputs['decode_prefix'],inputs['next_token'],inputs['prefill']
    full = prefix+[token]
    def hidden(ids):
        return norm(torch.cat([embeddings[t] for t in ids],dim=0).unsqueeze(0))
    def run(ids,positions,cache,mask=None):
        captured = {}
        def before(module,args):
            captured['o_input'] = args[0].detach().clone()
        handle = attention.o_proj.register_forward_pre_hook(before)
        try:
            x = hidden(ids)
            positions = torch.tensor([positions],dtype=torch.long,device=device)
            result = attention(x,position_embeddings=rotary(x,positions),attention_mask=mask,
                past_key_value=cache,cache_position=positions[0] if isinstance(cache,DynamicCache)
                    else torch.arange(len(ids),device=device))
            captured['output'] = result[0].detach().clone()
            return captured
        finally:
            handle.remove()
    reference = run(full,list(range(len(full))),DynamicCache())
    prefix_cache = DynamicCache()
    run(prefix,list(range(len(prefix))),prefix_cache)
    cached = run([token],[len(prefix)],prefix_cache)
    # Rebuild prefix cache independently, before its one-token append.
    prefix_cache = DynamicCache()
    run(prefix,list(range(len(prefix))),prefix_cache)
    d = Request('D',prefix,0,2,generated=[token],kv_length=len(prefix))
    p = Request('P',prompt,0,1)
    layout = packed_layout([d,p])
    dense = {r.request_id:torch.empty((48,2,1,4,row['after'],128),dtype=torch.bfloat16,device=device)
             for r,row in zip((d,p),layout)}
    keys,values = prefix_cache[0]
    dense['D'][0,0,:,:,:len(prefix),:].copy_(keys)
    dense['D'][0,1,:,:,:len(prefix),:].copy_(values)
    cache = make_packed_cache(torch,layout,dense,pad_keys=True)
    tensors = batch_tensors(torch,[d,p],layout,device,pad_keys=True)
    mixed = run([token]+prompt,[len(prefix)]+list(range(len(prompt))),cache,tensors['attention_mask'])
    def delta(actual,target):
        return error_metrics(torch,actual,target)
    metrics = dict(mixed_o_input_vs_full=delta(mixed['o_input'][0,0],reference['o_input'][0,-1]),
        mixed_o_output_vs_full=delta(mixed['output'][0,0],reference['output'][0,-1]),
        cached_o_input_vs_full=delta(cached['o_input'][0,0],reference['o_input'][0,-1]),
        cached_o_output_vs_full=delta(cached['output'][0,0],reference['output'][0,-1]))
    # Same real row repeated at three matrix heights isolates projection arithmetic.
    controls = {}
    for name,module,row in [('o_proj',attention.o_proj,reference['o_input'][0,-1]),
                            ('v_proj',attention.v_proj,hidden(full)[0,-1])]:
        outputs = {n:module(row.unsqueeze(0).repeat(n,1))[0] for n in (1,9,13)}
        controls[name] = {str(n)+'_vs_1':delta(outputs[n],outputs[1]) for n in (9,13)}
    return dict(first_attention=metrics,identical_row_shape_controls=controls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--diagnostic',type=Path,required=True)
    parser.add_argument('--results-dir',default='/workspace/results/kite_projection_probe')
    args = parser.parse_args()
    output = new_output(args.results_dir,'first-layer')
    state = dict(status='running',is_performance=False,variants={})
    monitor = None
    started = time.perf_counter_ns()
    try:
        state['software'] = require_versions()
        state['source_bundle'] = verify_source_bundle(Path(__file__).resolve().parents[1])
        manifest = verify_checkpoint(args.checkpoint)
        state['model_manifest_sha256'] = manifest.get('content_manifest_sha256')
        raw = args.diagnostic.read_bytes()
        diagnostic = json.loads(raw)
        if diagnostic['status']!='diagnostic_complete' or diagnostic['is_performance']:
            raise RuntimeError('Expected the completed original short diagnostic.')
        state['input_diagnostic_sha256'] = hashlib.sha256(raw).hexdigest()
        state['inputs'] = diagnostic['input_ids']
        snapshot = resource_snapshot(output)
        state['resources_before'] = snapshot
        check_gpu_budget(snapshot,1,reserve_gib=1)
        pids = run_readonly(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
        state['gpu_pid_query_before_cuda'] = pids
        if parse_gpu_pids(pids):
            raise RuntimeError('GPU has another process; do not run the probe.')
        if snapshot['effective_available_bytes'] is None or snapshot['effective_available_bytes']<GIB:
            raise RuntimeError('Need at least 1 GiB available host RAM for the small probe.')
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.version.cuda!='11.8':
            raise RuntimeError('Expected the fixed one-GPU cu118 installation.')
        os.sched_setaffinity(0,set(range(24,32)))
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda:0')
        initialize_cuda_stats(torch,device)
        monitor = ResourceMonitor(evidence_path=output/'gpu_monitor_startup.json')
        inputs = state['inputs']
        ids = inputs['decode_prefix']+[inputs['next_token']]+inputs['prefill']
        show('Reading only first-layer attention weights and selected embedding rows; no full Qwen loading.')
        attention,norm,rotary,embeddings,state['loading'] = load_first_layer(torch,args.checkpoint,ids,device)
        state['default_bf16_reduced_precision_reduction'] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            for name,enabled in [('default',state['default_bf16_reduced_precision_reduction']),('no_reduced_accumulation',False)]:
                monitor.check()
                with bf16_reduction(torch,enabled):
                    state['variants'][name] = probe(torch,attention,norm,rotary,embeddings,inputs,device)
                write_json(output/'projection_probe.json',state)
                error = state['variants'][name]['first_attention']['mixed_o_output_vs_full']
                show(f'{name}: first-layer mixed/full output relative_l2={error["relative_l2"]:.8f}; '
                     f'max_abs={error["max_abs"]:.8f}')
            with fp32_projection_control(torch,attention):
                state['variants']['fp32_projection_control_not_a_formal_change'] = probe(
                    torch,attention,norm,rotary,embeddings,inputs,device)
            error = state['variants']['fp32_projection_control_not_a_formal_change']['first_attention']['mixed_o_output_vs_full']
            show(f'FP32 projection control only: relative_l2={error["relative_l2"]:.8f}; max_abs={error["max_abs"]:.8f}')
        monitor.check()
        state.update(status='probe_complete',cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error))
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        if monitor is not None:
            monitor.close()
            write_json(output/'resource_monitor.json',monitor.record())
            if monitor.error:
                state.update(status='failed',error_type='ResourceInterference',error=monitor.error)
        state['diagnostic_wall_ms_not_performance'] = (time.perf_counter_ns()-started)/1e6
        write_json(output/'projection_probe.json',state)
        print('Results:',output,flush=True)
    if state['status']=='failed':
        raise RuntimeError(state.get('error','Probe failed.'))


if __name__=='__main__':
    main()
