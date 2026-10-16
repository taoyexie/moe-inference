"""Actual KV paging, full-prefix recovery and workload-start calibration."""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict

from kite_controller import Controller
from kite_e2e_core import KV_BYTES, Request, RuntimeCosts, packed_layout, ready_batch, validate_requests
from kite_e2e_torch import PAGE_SHAPE, packed_forward


class Runtime:
    def __init__(self, engine, budget_bytes, mode, options=None):
        self.engine,self.torch,self.mode = engine,engine.torch,mode
        self.options = options or {}
        self.host_kv_limit_bytes = self.options.get('host_kv_limit_bytes',1024**3)
        self.progress_every = self.options.get('progress_every',0)
        engine.attach_pool(budget_bytes)
        engine.reset_task()
        self.controller = Controller(engine.pool.usable_bytes,current_weight=.65 if mode=='kite' else 0.)
        self.counters = Counter()
        self.recoveries = []
        self.calibration_rows = []
        self.cpu_cold_setup_rows = []
        self.batch_records = []
        self.host_kv_live = self.host_kv_peak = 0
        self.workspace_peak = 0
        self.pool_snapshot = None
        self.apply(self.controller.proposal)

    def apply(self, candidate):
        self.engine.synchronize()
        evicted = self.engine.pool.set_limits(candidate.expert_objects,candidate.kv_objects)
        self.counters['split_evictions'] += len(evicted)
        self.pool_snapshot = self.engine.pool.snapshot()

    def expire(self, request):
        self.engine.synchronize()
        for key in list(self.engine.pool.entries['kv']):
            if key[0]==request.request_id:
                self.engine.pool.remove('kv',key)
        self.host_kv_live -= len(request.cpu_pages)*KV_BYTES
        request.cpu_pages.clear()
        request.versions.clear()
        if self.host_kv_live<0:
            raise RuntimeError('Host KV byte accounting underflow.')

    def host_page(self, request, block):
        if block not in request.cpu_pages:
            request.cpu_pages[block] = self.torch.zeros(PAGE_SHAPE,dtype=self.torch.bfloat16,pin_memory=True)
            request.versions[block] = 0
            self.host_kv_live += KV_BYTES
            self.host_kv_peak = max(self.host_kv_peak,self.host_kv_live)
            if self.host_kv_live>self.host_kv_limit_bytes:
                raise RuntimeError('Host KV exceeds the declared workload memory bound.')
        return request.cpu_pages[block]

    def resident(self, request, block, touch=True):
        key = (request.request_id,block)
        entry = self.engine.pool.get('kv',key,touch=touch)
        if entry is not None and entry['version']!=request.versions.get(block):
            self.engine.pool.remove('kv',key)
            entry = None
        return entry

    def restore(self, request, block, dense):
        engine = self.engine
        start = block*128
        n = min(128,request.kv_length-start)
        if n<1 or block not in request.cpu_pages:
            raise RuntimeError('Restore needs a valid host copy of this exact KV page.')
        engine.kv_scratch.copy_(request.cpu_pages[block],non_blocking=True)
        entry,_ = engine.pool.admit('kv',(request.request_id,block))
        if entry is not None:
            engine.arena.store_page(entry,engine.kv_scratch)
            entry['version'] = request.versions[block]
        dense[:,:,:,:,start:start+n,:].copy_(engine.kv_scratch[:,:,:,:,:n,:])

    def commit(self, request, dense, before, after, replace=False):
        """Write newly computed ranges to CPU truth and optional shared GPU frames."""
        engine = self.engine
        changed = []
        start = 0 if replace else before
        for block in range(start//128,(after+127)//128):
            absolute = max(start,block*128)
            end = min(after,(block+1)*128)
            n,offset = end-absolute,absolute-block*128
            page = self.host_page(request,block)
            page[:,:,:,:,offset:offset+n,:].copy_(dense[:,:,:,:,absolute:end,:],non_blocking=True)
            key = (request.request_id,block)
            entry = self.resident(request,block,touch=True)
            fresh = entry is None
            if fresh:
                entry,_ = engine.pool.admit('kv',key)
            if entry is not None:
                if fresh:
                    valid = min(128,after-block*128)
                    engine.kv_scratch.zero_()
                    engine.kv_scratch[:,:,:,:,:valid,:].copy_(dense[:,:,:,:,block*128:block*128+valid,:])
                    engine.arena.store_page(entry,engine.kv_scratch)
                else:
                    engine.arena.write_range(entry,dense,absolute,n,offset)
            request.versions[block] += 1
            if entry is not None:
                entry['version'] = request.versions[block]
            changed.append(key)
        engine.synchronize()  # CPU truth is not valid before all D2H writes complete.
        request.kv_length = after
        return changed

    def rebuild(self, request):
        engine = self.engine
        engine.synchronize()
        started = time.perf_counter_ns()
        old = request.kv_length
        tokens = request.prefix
        if len(tokens)!=old or old<1:
            raise RuntimeError('Replay needs original/previously generated token prefix, not the unconsumed next token.')
        dense = engine.replay_prefix(tokens)
        # Replace the complete recovered prefix consistently, never mix versions.
        self.commit(request,dense,0,old,replace=True)
        engine.synchronize()
        return dense,(time.perf_counter_ns()-started)/1e6

    def prepare(self, requests, layout, force_kv=None):
        engine = self.engine
        dense,groups = {},[]
        for req,row in zip(requests,layout):
            work = self.torch.empty((48,2,1,4,row['after'],128),
                                    dtype=self.torch.bfloat16,device=engine.device)
            blocks = list(range((row['before']+127)//128))
            objects = [(req.request_id,b) for b in blocks]
            if blocks:
                groups.append(dict(group_id=(req.request_id,len(req.generated)),
                                   context=row['before'],objects=objects))
            absent = [b for b in blocks if self.resident(req,b,touch=False) is None]
            restore,replay = engine.costs.kv(row['before']) if engine.costs is not None and blocks else (0,math.inf)
            use_replay = bool(absent) and (force_kv=='replay' or (force_kv is None and replay<len(absent)*restore))
            if use_replay:
                recovered,elapsed = self.rebuild(req)
                work[:,:,:,:,:row['before'],:].copy_(recovered)
                self.counters['kv_read_pages'] += len(blocks)
                self.counters['kv_hits'] += len(blocks)-len(absent)
                self.counters['kv_misses'] += len(absent)
                self.counters['kv_replay_tasks'] += 1
                self.recoveries.append(dict(request_id=req.request_id,kind='replay',wall_ms=elapsed,
                    missing_pages=len(absent),context=row['before'],macro_charged_once=True))
                del recovered
            else:
                for block in blocks:
                    self.counters['kv_read_pages'] += 1
                    entry = self.resident(req,block)
                    if entry is not None and force_kv is None:
                        self.counters['kv_hits'] += 1
                        start = block*128
                        engine.arena.read_page(entry,work,start,min(128,row['before']-start))
                    else:
                        self.counters['kv_misses'] += 1
                        engine.synchronize()
                        tick = time.perf_counter_ns()
                        self.restore(req,block,work)
                        engine.synchronize()
                        elapsed = (time.perf_counter_ns()-tick)/1e6
                        self.counters['kv_restore_pages'] += 1
                        self.recoveries.append(dict(request_id=req.request_id,kind='restore',wall_ms=elapsed,
                            missing_pages=1,context=row['before'],bytes=KV_BYTES))
            dense[req.request_id] = work
        self.workspace_peak = max(self.workspace_peak,sum(x.numel()*x.element_size() for x in dense.values()))
        return dense,groups

    def forward(self, requests, observe=True, force_kv=None):
        engine = self.engine
        layout = packed_layout(requests)
        p = sum(row['phase']=='prefill' for row in layout)
        decision = self.controller.before_batch(p,len(layout)-p) if observe else None
        tick = time.perf_counter_ns()
        if decision is not None:
            chosen = next(c for c in self.controller.choices if asdict(c)==decision['candidate'])
            self.apply(chosen)
            decision['physical_split_applied'] = True
            decision['stage'] = 'runtime_inference' if self.controller.costs is not None else 'runtime_calibration'
        engine.synchronize()
        resize_ms = (time.perf_counter_ns()-tick)/1e6
        dense,groups = self.prepare(requests,layout,force_kv=force_kv)
        engine.events = []
        result = packed_forward(self.torch,engine.model,requests,layout,dense,engine.device,
                                pad_keys=getattr(engine,'pad_attention_keys',False))
        logits = result.logits[0].clone()
        del result
        writes = []
        for req,row in zip(requests,layout):
            writes.extend(self.commit(req,dense[req.request_id],row['before'],row['after']))
        if observe:
            # Internal replay invocations are not duplicated in normal MRC evidence.
            self.controller.after_batch(engine.events,groups)
            for key in writes:
                self.controller.history.kv_stack.access(key)
            routed = sum(q for _,q in engine.events)
            expected = 48*8*sum(len(r.query) for r in requests)
            if routed!=expected:
                raise RuntimeError('Full-model batch router conservation failed.')
            self.batch_records.append(dict(step=len(self.batch_records),prefill=p,decode=len(layout)-p,
                query_tokens=sum(len(r.query) for r in requests),decision=decision,
                resize_ms=resize_ms,pool=engine.pool.snapshot(),routed_tokens=routed,
                expert_accesses=list(engine.events),kv_read_groups=groups,
                expert_counts=dict(engine.counts),kv_counts=dict(self.counters)))
        del dense
        engine.synchronize()
        return logits

    def timed_sample(self, path, x, operation, warmup=False):
        self.engine.synchronize()
        tick = time.perf_counter_ns()
        value = operation()
        self.engine.synchronize()
        ms = (time.perf_counter_ns()-tick)/1e6
        self.calibration_rows.append(dict(path=path,x=x,wall_ms=ms,warmup=warmup))
        if not warmup:
            self.controller.calibration.add(path,x,ms)
        return value

    def check_startup_replay(self, actual, reference, context, warmup, trial):
        """Keep the original tolerances and preserve failed evidence before raising."""
        from expert_paths import error_metrics
        check = dict(kind='kv_replay',context=context,warmup=warmup,trial=trial,
                     reference_source='original prefill KV captured before any replay',
                     tolerances=dict(relative_l2=.03,normalized_max_abs=.05),passed=False)
        self.calibration_checks.append(check)
        try:
            error = error_metrics(self.torch,actual,reference)
        except RuntimeError as failure:
            check['error_message'] = str(failure)
            raise RuntimeError(f'Startup replay check failed at context={context}, '
                               f'warmup={warmup}, trial={trial}: {failure}') from failure
        check['error'] = error
        check['passed'] = error['relative_l2']<=.03 and error['normalized_max_abs']<=.05
        if not check['passed']:
            # Failure-only diagnostics, never insert reference recomputation in formal inference.
            check['layer_errors'] = [dict(layer=layer,**error_metrics(self.torch,actual[layer],reference[layer]))
                                     for layer in range(48)]
            check['first_nonzero_layer'] = next((r['layer'] for r in check['layer_errors'] if r['max_abs']!=0),None)
            raise RuntimeError(f'Startup replay numerical check failed at context={context}, '
                               f'warmup={warmup}, trial={trial}: {error}; '
                               f'first_nonzero_layer={check["first_nonzero_layer"]}. '
                               'Per-layer errors saved in partial_calibration.json.')
        print(f'Startup KV context={context}, warmup={warmup}, trial={trial}: '
              f'replay relative_l2={error["relative_l2"]:.6f}, '
              f'normalized_max_abs={error["normalized_max_abs"]:.6f}; PASS',flush=True)

    def calibrate(self, prompt_ids):
        """Physically execute all four paths once per workload, with fixed sampling protocol."""
        engine = self.engine
        started = time.perf_counter_ns()
        self.calibration_checks = []
        engine.want_capture = True
        first = Request('calibration-128',list(prompt_ids[:128]),0,1)
        self.forward([first])
        if engine.capture is None:
            raise RuntimeError('No actual layer-0 expert input captured for task calibration.')
        key,native = engine.capture
        engine.want_capture = False
        self.calibration_capture = dict(key=key,native_q=int(native.shape[0]),
            construction='actual captured input, prefix/cyclic row resizing for startup q probes')
        startup_keys = list(dict.fromkeys(k for k,_ in engine.events))
        for q in (1,8,32):
            x = native.index_select(0,self.torch.arange(q,device=engine.device)%native.shape[0])
            def upload():
                engine.upload(key,engine.scratch_weights)
                return engine.gpu_expert(engine.scratch_weights,x)
            for warmup,repeats in ((True,1),(False,self.options.get('expert_repeats',3))):
                for _ in range(repeats):
                    gpu = self.timed_sample('expert_upload',q,upload,warmup)
                    cpu = self.timed_sample('expert_cpu',q,lambda:engine.cpu_expert(key,x),warmup)
                    from expert_paths import error_metrics
                    error = error_metrics(self.torch,cpu,gpu)
                    if error['relative_l2']>.02 or error['normalized_max_abs']>.03:
                        raise RuntimeError(f'Startup expert numerical check failed: {error}')
                    self.calibration_checks.append(dict(kind='expert',q=q,error=error))
            del x
        if self.options.get('measure_cpu_cold_setup',False):
            # Distinct real routed experts, not repeated conversion of LLC-hot masters.
            eligible = [k for k in startup_keys if k not in engine.cpu_cache]
            n = 1+self.options.get('cpu_cold_setup_repeats',5)
            if len(eligible)<n:
                raise RuntimeError('Not enough distinct genuine startup experts for cold setup samples.')
            selected = [eligible[i*(len(eligible)-1)//(n-1)] for i in range(n)]
            for i,cold_key in enumerate(selected):
                if cold_key in engine.cpu_cache:
                    raise RuntimeError('A cold setup sample unexpectedly has a cached FP32 copy.')
                engine.synchronize()
                tick = time.perf_counter_ns()
                engine.cpu_weights(cold_key)
                engine.synchronize()
                self.cpu_cold_setup_rows.append(dict(key=cold_key,warmup=i==0,
                    wall_ms=(time.perf_counter_ns()-tick)/1e6))
        engine.capture = None
        for length in (128,512):
            print(f'Startup KV context={length}: checking restore and full-prefix replay.',flush=True)
            req = first if length==128 else Request('calibration-512',list(prompt_ids[:512]),0,1)
            if length!=128:
                self.forward([req])
            block = (length-1)//128
            target = self.torch.empty((48,2,1,4,length,128),dtype=self.torch.bfloat16,device=engine.device)
            replay_reference = req.cpu_pages[block].to(engine.device).clone()
            for warmup,repeats in ((True,1),(False,2)):
                for trial in range(repeats):
                    from expert_paths import error_metrics
                    target[:,:,:,:,length-128:length,:].fill_(float('nan'))
                    self.timed_sample('kv_restore',length,lambda:self.restore(req,block,target),warmup)
                    reference = req.cpu_pages[block].to(engine.device)
                    restore_error = error_metrics(self.torch,target[:,:,:,:,length-128:length,:],reference)
                    if restore_error['max_abs']!=0:
                        raise RuntimeError('Startup restore must be bit-exact.')
                    self.calibration_checks.append(dict(kind='kv_restore',context=length,error=restore_error))
                    recovered = self.timed_sample('kv_replay',length,lambda:self.rebuild(req)[0],warmup)
                    actual = recovered[:,:,:,:,length-128:length,:]
                    self.check_startup_replay(actual,replay_reference,length,warmup,trial)
                    del recovered,reference,actual
            del target,replay_reference
            self.expire(req)
        cold_ms = max((r['wall_ms'] for r in self.cpu_cold_setup_rows if not r['warmup']),default=0.)
        self.controller.finish_calibration(lambda samples,model:RuntimeCosts(samples,model,self.mode,
            robust=self.options.get('robust_costs',False),cpu_cold_setup_ms=cold_ms))
        engine.costs = self.controller.costs
        # Keep measured history, but no hidden calibration KV or warm GPU cache.
        engine.synchronize()
        engine.pool.clear()
        engine.events = []
        self.counters = Counter()
        self.recoveries = []
        engine.counts = Counter()
        self.batch_records = []
        self.apply(self.controller.proposal)
        return dict(wall_ms=(time.perf_counter_ns()-started)/1e6,
                    rows=self.calibration_rows,checks=self.calibration_checks,
                    fixed=engine.costs.fixed,affine=asdict(engine.costs.fitted),
                    domains=engine.costs.domains,capture=self.calibration_capture,
                    cpu_cold_setup_rows=self.cpu_cold_setup_rows,cpu_cold_setup_ms=cold_ms,
                    cpu_cold_estimator='maximum of formal distinct-expert setup samples' if self.cpu_cold_setup_rows else None,
                    cost_estimator=engine.costs.estimator,
                    warm_gpu_cache_cleared=True,cpu_fp32_cache_bytes=engine.cpu_cache_bytes)


def execute_requests(runtime, requests, task_started_ns):
    validate_requests(requests)
    engine = runtime.engine
    step = 0
    # Round zero requests exist before startup calibration, so their TTFT includes it.
    for req in requests:
        if req.release_step==0:
            req.submitted_ns = task_started_ns
    while not all(r.done for r in requests):
        batch = ready_batch(requests,step)
        if not batch:
            step += 1
            continue
        tick = time.perf_counter_ns()
        for req in batch:
            if req.submitted_ns is None:
                req.submitted_ns = tick
        logits = runtime.forward(batch)
        eos = engine.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos,list) else [eos] if eos is not None else []
        # Explicit fixed-length generation protocol shared by both policies.
        if eos_ids:
            logits[:,eos_ids] = -math.inf
        tokens = logits.argmax(dim=-1).tolist()
        engine.synchronize()
        finished = time.perf_counter_ns()
        for req,token in zip(batch,tokens):
            req.generated.append(int(token))
            req.token_ticks.append(finished)
            if req.done:
                runtime.expire(req)
        every = getattr(runtime,'progress_every',0)
        if every and (step+1)%every==0:
            print(f'Workload batch {step+1}: active={len(batch)}, '
                  f'elapsed={(finished-task_started_ns)/1e9:.2f}s (startup included)',flush=True)
        del logits
        step += 1
        if step>128:
            raise RuntimeError('Controlled schedule exceeded its declared 128-round bound.')
    engine.synchronize()
    return step
