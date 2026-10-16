"""Actual Qwen model, unified GPU frames and packed attention adapter.

Torch/Transformers are imported only by the remote entrypoint. No Accelerate
offload hooks are active: master experts stay CPU, fixed model tensors stay GPU.
"""
from __future__ import annotations

import gc
import json
import time
from collections import Counter, OrderedDict
from types import MethodType

from kite_e2e_core import EXPERT_BYTES, FRAME_BYTES, KV_BYTES, FramePool
from kite_trace import GIB


PAGE_SHAPE = (48,2,1,4,128,128)


class TorchArena:
    def __init__(self, torch, pool, device):
        self.torch,self.pool,self.device = torch,pool,device
        self.data = torch.empty((pool.frame_count,FRAME_BYTES//2),dtype=torch.bfloat16,device=device)
        if self.data.numel()*self.data.element_size()!=pool.usable_bytes:
            raise RuntimeError('Physical GPU arena bytes differ from the declared frame pool.')

    def weights(self, entry):
        a,b,c = (self.data[i] for i in entry['frames'])
        return a.view(768,2048),b.view(768,2048),c.view(2048,768)

    def page_frames(self, entry):
        return [self.data[i].view(12,2,1,4,128,128) for i in entry['frames']]

    def store_page(self, entry, page):
        for group,frame in enumerate(self.page_frames(entry)):
            frame.copy_(page[group*12:(group+1)*12])

    def read_page(self, entry, dense, start, count):
        for group,frame in enumerate(self.page_frames(entry)):
            dense[group*12:(group+1)*12,:,:,:,start:start+count,:].copy_(frame[:,:,:,:,:count,:])

    def write_range(self, entry, dense, absolute_start, count, page_offset):
        for group,frame in enumerate(self.page_frames(entry)):
            frame[:,:,:,:,page_offset:page_offset+count,:].copy_(
                dense[group*12:(group+1)*12,:,:,:,absolute_start:absolute_start+count,:])


def make_packed_cache(torch, layout, dense):
    from transformers import Cache

    class PackedCache(Cache):
        def __init__(self):
            super().__init__()
            self.updated = set()

        def get_seq_length(self, layer_idx=0):
            return sum(row['before'] for row in layout)

        def get_max_cache_shape(self):
            return None

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if layer_idx in self.updated:
                raise RuntimeError('Packed cache layer updated twice.')
            self.updated.add(layer_idx)
            if tuple(key_states.shape)!=(1,4,layout[-1]['q_end'],128):
                raise RuntimeError('Unexpected packed incoming KV shape.')
            keys,values = [],[]
            for row in layout:
                block = dense[row['request_id']]
                source = slice(row['q_start'],row['q_end'])
                dest = slice(row['before'],row['after'])
                block[layer_idx,0,:,:,dest,:].copy_(key_states[:,:,source,:])
                block[layer_idx,1,:,:,dest,:].copy_(value_states[:,:,source,:])
                keys.append(block[layer_idx,0])
                values.append(block[layer_idx,1])
            return torch.cat(keys,dim=-2),torch.cat(values,dim=-2)

    return PackedCache()


def batch_tensors(torch, requests, layout, device):
    ids = [token for req in requests for token in req.query]
    positions = [p for row in layout for p in range(row['before'],row['after'])]
    mask = torch.full((1,1,layout[-1]['q_end'],layout[-1]['k_end']),
                      torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16,device=device)
    for row in layout:
        valid = (torch.arange(row['after'],device=device).unsqueeze(0)
                 <=torch.arange(row['before'],row['after'],device=device).unsqueeze(1))
        mask[0,0,row['q_start']:row['q_end'],row['k_start']:row['k_end']].masked_fill_(valid,0)
    return dict(input_ids=torch.tensor([ids],dtype=torch.long,device=device),
                position_ids=torch.tensor([positions],dtype=torch.long,device=device),
                cache_position=torch.arange(len(ids),device=device),attention_mask=mask,
                logits_to_keep=torch.tensor([r['q_end']-1 for r in layout],dtype=torch.long,device=device))


class QwenEngine:
    def __init__(self, torch, model, device, cpu_cache_bytes=8*GIB):
        self.torch,self.model,self.device = torch,model,device
        self.cpu_cache_limit = cpu_cache_bytes
        self.cpu_cache,self.cpu_cache_bytes = OrderedDict(),0
        self.cpu_cache_peak = 0
        self.costs,self.pool,self.arena = None,None,None
        self.force_path = None
        self.in_replay = False
        self.events,self.counts = [],Counter()
        self.capture,self.want_capture = None,False
        self.health_check = None
        self.validate_cpu_outputs = False
        self.cpu_validation = dict(comparisons=0,worst_relative_l2=0.,worst_normalized_max_abs=0.)
        self.replay_workspace_peak = 0
        self.upload_host = torch.empty((3,768*2048),dtype=torch.bfloat16,pin_memory=True)
        self.scratch_weights = tuple(torch.empty(shape,dtype=torch.bfloat16,device=device)
                                     for shape in ((768,2048),(768,2048),(2048,768)))
        self.kv_scratch = torch.empty(PAGE_SHAPE,dtype=torch.bfloat16,device=device)
        for layer_id,layer in enumerate(model.model.layers):
            layer.self_attn.to(device)
            layer.input_layernorm.to(device)
            layer.post_attention_layernorm.to(device)
            layer.mlp.gate.to(device)
            engine = self
            def forward(block, hidden_states, layer_id=layer_id):
                return engine.moe(block,hidden_states,layer_id)
            layer.mlp.forward = MethodType(forward,layer.mlp)
        model.model.embed_tokens.to(device)
        model.model.norm.to(device)
        model.model.rotary_emb.to(device)
        model.lm_head.to(device)
        if any(getattr(module,'_hf_hook',None) is not None for module in model.modules()):
            raise RuntimeError('Unexpected Accelerate hooks: cannot guarantee physical expert cache.')
        fixed = [p for name,p in model.named_parameters() if '.mlp.experts.' not in name]
        if any(p.device.type!='cuda' for p in fixed):
            raise RuntimeError('All fixed model parameters must be resident on GPU.')
        self.fixed_gpu_bytes = sum(p.numel()*p.element_size() for p in fixed)
        self.master_cpu_bytes = sum(p.numel()*p.element_size() for name,p in model.named_parameters()
                                    if '.mlp.experts.' in name)
        if self.master_cpu_bytes!=48*128*EXPERT_BYTES:
            raise RuntimeError('Expected all 6,144 genuine BF16 CPU expert masters.')

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def attach_pool(self, budget_bytes):
        self.synchronize()
        self.pool,self.arena = None,None
        gc.collect()
        self.torch.cuda.empty_cache()
        self.pool = FramePool(budget_bytes)
        self.arena = TorchArena(self.torch,self.pool,self.device)

    def clear_between_tasks(self):
        """Harness reset, outside both policies' clocks; never bill a predecessor to its successor."""
        self.synchronize()
        self.cpu_cache.clear()
        self.cpu_cache_bytes = 0
        self.capture = None
        if self.pool is not None:
            self.pool.clear()
        self.pool,self.arena = None,None
        gc.collect()
        self.torch.cuda.empty_cache()

    def reset_task(self):
        self.synchronize()
        self.pool.clear()
        self.cpu_cache.clear()
        self.cpu_cache_bytes,self.cpu_cache_peak = 0,0
        self.replay_workspace_peak = 0
        self.events,self.counts = [],Counter()
        self.costs,self.force_path = None,None
        self.capture,self.want_capture = None,False

    def masters(self, key):
        layer,expert = key
        module = self.model.model.layers[layer].mlp.experts[expert]
        weights = tuple(getattr(module,name).weight for name in ('gate_proj','up_proj','down_proj'))
        if any(w.device.type!='cpu' or w.dtype!=self.torch.bfloat16 for w in weights):
            raise RuntimeError('Expert masters must remain CPU BF16.')
        return weights

    def upload(self, key, weights):
        for i,(source,target) in enumerate(zip(self.masters(key),weights)):
            stage = self.upload_host[i].view(source.shape)
            stage.copy_(source)
            target.copy_(stage,non_blocking=True)
        # The shared pinned staging buffer cannot be overwritten until DMA ends.
        self.synchronize()

    def gpu_expert(self, weights, x):
        functional = self.torch.nn.functional
        gate,up,down = weights
        return functional.linear(functional.silu(functional.linear(x,gate))*functional.linear(x,up),down)

    def cpu_weights(self, key):
        item = self.cpu_cache.get(key)
        tag = 'replay_' if self.in_replay else ''
        if item is not None:
            self.cpu_cache.move_to_end(key)
            self.counts[tag+'cpu_fp32_cache_hits'] += 1
            return item
        need = 2*EXPERT_BYTES
        if need>self.cpu_cache_limit:
            raise RuntimeError('CPU cache cannot fit a single FP32 expert.')
        while self.cpu_cache_bytes+need>self.cpu_cache_limit:
            self.cpu_cache.popitem(last=False)
            self.cpu_cache_bytes -= need
            self.counts[tag+'cpu_fp32_evictions'] += 1
        item = tuple(w.to(dtype=self.torch.float32,copy=True) for w in self.masters(key))
        self.cpu_cache[key] = item
        self.cpu_cache_bytes += need
        self.cpu_cache_peak = max(self.cpu_cache_peak,self.cpu_cache_bytes)
        self.counts[tag+'cpu_fp32_conversions'] += 1
        return item

    def cpu_expert(self, key, x):
        # Cold FP32 conversion is deliberately inside the actual path.
        weights = self.cpu_weights(key)
        host = x.to(device='cpu',dtype=self.torch.float32)
        return self.gpu_expert(weights,host).to(device=self.device,dtype=self.torch.bfloat16)

    def expert(self, key, x):
        q = int(x.shape[0])
        if q<1:
            raise RuntimeError('Empty experts must be skipped by the router adapter.')
        tag = 'replay_' if self.in_replay else ''
        if not self.in_replay:
            self.events.append((key,q))
        if self.want_capture and self.capture is None and key[0]==0 and not self.in_replay:
            self.capture = (key,x.detach().clone())
        entry = self.pool.get('expert',key)
        self.counts[tag+'expert_calls'] += 1
        if entry is not None and self.force_path is None:
            self.counts[tag+'expert_hits'] += 1
            return self.gpu_expert(self.arena.weights(entry),x)
        self.counts[tag+'expert_misses'] += 1
        path = self.force_path
        if path is None:
            upload,cpu = self.costs.expert(q) if self.costs is not None else (0,math_inf())
            path = 'cpu' if cpu<upload else 'upload'
        if path=='cpu':
            self.counts[tag+'expert_cpu'] += 1
            output = self.cpu_expert(key,x)
            if self.validate_cpu_outputs:
                from expert_paths import error_metrics
                self.upload(key,self.scratch_weights)
                reference = self.gpu_expert(self.scratch_weights,x)
                error = error_metrics(self.torch,output,reference)
                if error['relative_l2']>.02 or error['normalized_max_abs']>.03:
                    raise RuntimeError(f'Full-model CPU expert check failed at {key}: {error}')
                self.cpu_validation['comparisons'] += 1
                for metric in ('relative_l2','normalized_max_abs'):
                    self.cpu_validation['worst_'+metric] = max(self.cpu_validation['worst_'+metric],error[metric])
            return output
        entry,_ = self.pool.admit('expert',key)
        weights = self.arena.weights(entry) if entry is not None else self.scratch_weights
        try:
            self.upload(key,weights)
            result = self.gpu_expert(weights,x)
        except BaseException:
            self.pool.remove('expert',key)
            raise
        self.counts[tag+'expert_upload'] += 1
        return result

    def moe(self, block, hidden_states, layer_id):
        if self.health_check is not None:
            self.health_check()
        torch = self.torch
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1,shape[-1])
        logits = block.gate(flat)
        probability = torch.nn.functional.softmax(logits,dim=-1,dtype=torch.float32)
        weights,ids = torch.topk(probability,block.top_k,dim=-1)
        if block.norm_topk_prob:
            weights = weights/weights.sum(dim=-1,keepdim=True)
        weights = weights.to(flat.dtype)
        mask = torch.nn.functional.one_hot(ids,num_classes=block.num_experts).permute(2,1,0)
        out = torch.zeros_like(flat)
        routed = 0
        for expert_id in range(block.num_experts):
            rank,token = torch.where(mask[expert_id])
            q = int(token.shape[0])
            if not q:
                continue
            routed += q
            value = self.expert((layer_id,expert_id),flat.index_select(0,token))
            weighted = value*weights[token,rank,None]
            out.index_add_(0,token,weighted.to(flat.dtype))
        if routed!=flat.shape[0]*block.top_k:
            raise RuntimeError('Router token conservation failed.')
        return out.reshape(shape),logits

    def dense_from_cache(self, cache, length):
        if len(cache.key_cache)!=48 or len(cache.value_cache)!=48:
            raise RuntimeError('Expected full-model replay KV for all 48 layers.')
        dense = self.torch.empty((48,2,1,4,length,128),dtype=self.torch.bfloat16,device=self.device)
        for layer in range(48):
            dense[layer,0].copy_(cache.key_cache[layer])
            dense[layer,1].copy_(cache.value_cache[layer])
        return dense

    def replay_prefix(self, token_ids):
        from transformers import DynamicCache
        if not 1<=len(token_ids)<=2080:
            raise RuntimeError('Replay exceeds the declared 2080-token workspace bound.')
        previous = self.in_replay
        self.in_replay = True
        try:
            ids = self.torch.tensor([token_ids],dtype=self.torch.long,device=self.device)
            result = self.model(input_ids=ids,past_key_values=DynamicCache(),use_cache=True,
                                output_router_logits=False,return_dict=True,logits_to_keep=1)
            dense = self.dense_from_cache(result.past_key_values,len(token_ids))
            self.replay_workspace_peak = max(self.replay_workspace_peak,2*len(token_ids)*48*2*4*128*2)
            return dense
        finally:
            self.in_replay = previous


def math_inf():
    return float('inf')


def load_engine(torch, checkpoint, device, cpu_cache_bytes, snapshot):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    index = json.loads((checkpoint/'model.safetensors.index.json').read_text(encoding='utf-8'))
    master_bytes = index['metadata']['total_size']
    required = master_bytes+cpu_cache_bytes+9*GIB
    if snapshot['effective_available_bytes'] is None or snapshot['effective_available_bytes']<required:
        raise RuntimeError(f'Need {required/GIB:.1f} GiB available RAM for masters, CPU cache and reserve; do not load anyway.')
    started = time.perf_counter_ns()
    model,info = AutoModelForCausalLM.from_pretrained(str(checkpoint),local_files_only=True,
        trust_remote_code=False,use_safetensors=True,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map={'':'cpu'},low_cpu_mem_usage=True,
        offload_state_dict=False,output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('Full checkpoint loading is incomplete; no random substitutes allowed.')
    model.eval().requires_grad_(False)
    engine = QwenEngine(torch,model,device,cpu_cache_bytes)
    # CPU masters must be materialized RAM, not demand-paged checkpoint mappings.
    for name,parameter in model.named_parameters():
        if '.mlp.experts.' in name:
            parameter.data = parameter.detach().clone()
    engine.synchronize()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    return engine,tokenizer,{'load_seconds':(time.perf_counter_ns()-started)/1e9,
        'fixed_gpu_weight_bytes':engine.fixed_gpu_bytes,'master_cpu_weight_bytes':engine.master_cpu_bytes,
        'cpu_fp32_cache_limit_bytes':cpu_cache_bytes,'loading_info':info,
        'expert_master_placement':'CPU BF16; no Accelerate execution hooks',
        'fixed_model_placement':'GPU BF16',
        'extra_gpu_scratch_bytes':EXPERT_BYTES+KV_BYTES,
        'extra_pinned_upload_bytes':EXPERT_BYTES}
