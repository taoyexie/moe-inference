"""Short, instrumented correctness diagnosis only; NEVER a performance workload."""
from contextlib import contextmanager

from expert_paths import error_metrics
from kite_e2e_core import KV_BYTES,Request,packed_layout
from kite_e2e_runtime import Runtime


TOLERANCES = dict(relative_l2=.03,normalized_max_abs=.05)


def accepted(error):
    return all(error[key]<=limit for key,limit in TOLERANCES.items())


def query_indices(row):
    # Compare the decode query with the last token of its full-prefix reference.
    return [row['q_end']-1] if row['label']=='decode' else list(range(row['q_start'],row['q_end']))


def compare(torch, actual, reference):
    if set(actual['vectors'])!=set(reference['vectors']):
        raise RuntimeError('Diagnostic traces have different stage/token identities.')
    stages = []
    for key,target in reference['vectors'].items():
        stages.append(dict(stage=key,**error_metrics(torch,actual['vectors'][key],target)))
    routes = []
    for key,target in reference['routes'].items():
        value = actual['routes'][key]
        routes.append(dict(stage=key,tokens=int(target.shape[0]),
            changed_tokens=int((value!=target).any(dim=-1).sum().item())))
    first_attention = []
    for key,target in reference['first_attention'].items():
        first_attention.append(dict(tensor=key,**error_metrics(torch,actual['first_attention'][key],target)))
    return dict(logits=error_metrics(torch,actual['logits'],reference['logits']),
        per_request=[dict(request=label,**error_metrics(torch,actual['logits'][i],reference['logits'][i]))
                     for i,label in enumerate(('decode','prefill'))],
        argmax_equal=bool(torch.equal(actual['logits'].argmax(-1),reference['logits'].argmax(-1))),
        first_nonzero_stage=next((r['stage'] for r in stages if r['max_abs']!=0),None),
        first_changed_route=next((r['stage'] for r in routes if r['changed_tokens']),None),
        first_layer_attention=first_attention,stages=stages,routes=routes)


class Capture:
    def __init__(self,engine):
        self.engine,self.torch = engine,engine.torch
        self.rows,self.handles = [],[]
        self.data = dict(vectors={},routes={},first_attention={},attention_oracle=[])

    def select_rows(self,tensor,row,dim=0):
        index = self.torch.tensor(query_indices(row),dtype=self.torch.long,device=tensor.device)
        return tensor.index_select(dim,index).detach().float().cpu().clone()

    def hook(self,stage,router=False):
        def collect(module,inputs,output):
            value = output[0] if isinstance(output,tuple) else output
            flat = value.reshape(-1,value.shape[-1])
            selected = None
            if router:
                probabilities = self.torch.nn.functional.softmax(flat,dim=-1,dtype=self.torch.float32)
                selected = self.torch.topk(probabilities,8,dim=-1).indices.sort(dim=-1).values
            for row in self.rows:
                key = stage+'/'+row['label']
                self.data['vectors'][key] = self.select_rows(flat,row)
                if router:
                    index = self.torch.tensor(query_indices(row),dtype=self.torch.long,device=flat.device)
                    self.data['routes'][key] = selected.index_select(0,index).detach().cpu().clone()
        return collect

    @contextmanager
    def record(self,rows):
        """Observe only; don't change logits, router choices, masks or expert paths."""
        self.rows = rows
        model = self.engine.model.model
        self.handles = [model.embed_tokens.register_forward_hook(self.hook('embedding'))]
        for i,layer in enumerate(model.layers):
            for name,module in [('input_norm',layer.input_layernorm),('q_proj',layer.self_attn.q_proj),
                                ('k_proj',layer.self_attn.k_proj),('v_proj',layer.self_attn.v_proj),
                                ('attention',layer.self_attn),('router',layer.mlp.gate),
                                ('moe',layer.mlp),('layer_output',layer)]:
                self.handles.append(module.register_forward_hook(self.hook(f'L{i:02d}/{name}',name=='router')))
        self.handles.append(model.norm.register_forward_hook(self.hook('final_norm')))
        functional = self.torch.nn.functional
        original = functional.scaled_dot_product_attention
        calls = 0
        def inspect(query,key,value,*args,**kwargs):
            nonlocal calls
            result = original(query,key,value,*args,**kwargs)
            layer = calls
            calls += 1
            if layer==0:
                from torch.nn.attention import SDPBackend,sdpa_kernel
                # FP32 MATH on identical *short* attention inputs is an oracle only.
                # It never changes or replaces the real EFFICIENT_ATTENTION output.
                if args:
                    raise RuntimeError('Unexpected positional SDPA options in fixed Transformers version.')
                oracle_options = dict(kwargs)
                if oracle_options.get('attn_mask') is not None:
                    oracle_options['attn_mask'] = oracle_options['attn_mask'].float()
                with sdpa_kernel(SDPBackend.MATH):
                    oracle = original(query.float(),key.float(),value.float(),**oracle_options)
                self.data['attention_oracle'].append(error_metrics(self.torch,result,oracle))
                for row in rows:
                    label = row['label']
                    self.data['first_attention'][label+'/query'] = self.select_rows(query,row,2)
                    self.data['first_attention'][label+'/output'] = self.select_rows(result,row,2)
                    for name,tensor in [('key',key),('value',value)]:
                        self.data['first_attention'][label+'/'+name] = tensor[:,:,row['k_start']:row['k_end'],:].detach().float().cpu().clone()
            return result
        functional.scaled_dot_product_attention = inspect
        try:
            yield
        finally:
            functional.scaled_dot_product_attention = original
            for handle in self.handles:
                handle.remove()
            self.handles = []


def stock_row(label,length):
    return dict(label=label,q_start=0,q_end=length,k_start=0,k_end=length)


def labeled_layout(requests):
    return [{**row,'label':'decode' if i==0 else 'prefill'}
            for i,row in enumerate(packed_layout(requests))]


def diagnose(engine,specs,persist):
    """Five short routes distinguish native masking, paging and mixed-batch shape.

    persist() writes partial findings after every completed variant. No startup
    calibration or main workload is run; no measured times are performance data.
    """
    from transformers import DynamicCache
    torch = engine.torch
    runtime = Runtime(engine,32*1024**2,'baseline')
    engine.force_path = 'upload'
    report = dict(status='running',is_performance=False,tolerances=TOLERANCES,
        scope='short real-model numerical diagnosis; fixed genuine token shared across equivalent tests',
        precision='BF16 model; FP32 MATH oracle only for short first-layer attention, never substitutes actual output',
        comparisons={},variants={})
    traces = {}
    persist(report)
    seed = Request('numerical-seed',specs[0]['input_ids'][:8],0,3)
    try:
        first = runtime.forward([seed],observe=False)
        token = int(first[0].argmax().item())
        prefix = list(seed.prompt_ids)
        full_decode = prefix+[token]
        prompt = list(specs[1]['input_ids'][:12])
        report['input_ids'] = dict(decode_prefix=prefix,next_token=token,prefill=prompt)

        def begin(name):
            report['current_variant'] = name
            persist(report)
            print('Short diagnostic: '+name,flush=True)

        def finish(name,capture,logits):
            capture.data['logits'] = logits.detach().float().cpu().clone()
            traces[name] = capture.data
            report['variants'][name] = dict(status='completed',
                argmax=traces[name]['logits'].argmax(-1).tolist(),
                first_layer_efficient_vs_fp32_math=capture.data['attention_oracle'])
            if 'stock_full' in traces and name!='stock_full':
                report['comparisons'][name+'_vs_stock_full'] = compare(torch,traces[name],traces['stock_full'])
                row = report['comparisons'][name+'_vs_stock_full']
                print(f'{name}: relative_l2={row["logits"]["relative_l2"]:.6f}; '
                      f'first_nonzero_stage={row["first_nonzero_stage"]}; '
                      f'first_changed_route={row["first_changed_route"]}',flush=True)
            persist(report)

        def native(ids,cache=None):
            return engine.model(input_ids=torch.tensor([ids],dtype=torch.long,device=engine.device),
                past_key_values=cache if cache is not None else DynamicCache(),use_cache=True,
                return_dict=True,output_router_logits=False,logits_to_keep=1)

        # A: native full-prefix reference, no packed adapter.
        begin('stock_full')
        capture = Capture(engine)
        logits = []
        for label,ids in [('decode',full_decode),('prefill',prompt)]:
            with capture.record([stock_row(label,len(ids))]):
                result = native(ids)
            logits.append(result.logits[0,-1].clone())
            del result
        finish('stock_full',capture,torch.stack(logits))

        # B: same full-prefix inputs, but through explicit-mask packed adapter, serial.
        begin('packed_full_serial')
        capture = Capture(engine)
        logits = []
        for label,ids in [('decode',full_decode),('prefill',prompt)]:
            req = Request('numerical-full-'+label,ids,0,1)
            row = {**packed_layout([req])[0],'label':label}
            with capture.record([row]):
                result = runtime.forward([req],observe=False)
            logits.append(result[0].clone())
            runtime.expire(req)
        finish('packed_full_serial',capture,torch.stack(logits))

        # C: native DynamicCache incremental decode, independently recomputed prefix.
        begin('stock_cached_serial')
        capture = Capture(engine)
        result = native(prefix)
        cache = result.past_key_values
        del result
        row = dict(label='decode',q_start=0,q_end=1,k_start=0,k_end=len(full_decode))
        with capture.record([row]):
            result = native([token],cache)
        logits = [result.logits[0,-1].clone()]
        del result,cache
        with capture.record([stock_row('prefill',len(prompt))]):
            result = native(prompt)
        logits.append(result.logits[0,-1].clone())
        del result
        finish('stock_cached_serial',capture,torch.stack(logits))

        def restored_decode():
            req = Request('numerical-cached-D',list(prefix),0,3,generated=[token],kv_length=len(prefix))
            # Fresh independent CPU truth for each test; never mutate seed KV.
            req.cpu_pages = {k:v.clone() for k,v in seed.cpu_pages.items()}
            req.versions = dict(seed.versions)
            runtime.host_kv_live += len(req.cpu_pages)*KV_BYTES
            runtime.host_kv_peak = max(runtime.host_kv_peak,runtime.host_kv_live)
            return req

        # D/E: identical saved prefix KV and token, serial then mixed, both forced restore.
        for mixed in (False,True):
            begin('packed_cached_mixed' if mixed else 'packed_cached_serial')
            engine.pool.clear()
            d = restored_decode()
            p = Request('numerical-cached-P',list(prompt),0,1)
            capture = Capture(engine)
            if mixed:
                with capture.record(labeled_layout([d,p])):
                    result = runtime.forward([d,p],observe=False,force_kv='restore')
                logits = result
            else:
                logits = []
                for label,req in [('decode',d),('prefill',p)]:
                    row = {**packed_layout([req])[0],'label':label}
                    with capture.record([row]):
                        result = runtime.forward([req],observe=False,force_kv='restore')
                    logits.append(result[0].clone())
                logits = torch.stack(logits)
            finish('packed_cached_mixed' if mixed else 'packed_cached_serial',capture,logits)
            runtime.expire(d)
            runtime.expire(p)
        report['comparisons']['packed_cached_mixed_vs_serial'] = compare(
            torch,traces['packed_cached_mixed'],traces['packed_cached_serial'])
        report['status'] = 'diagnostic_complete'
        report['all_stock_reference_checks_within_original_tolerance'] = all(
            accepted(row['logits']) for name,row in report['comparisons'].items() if name.endswith('_vs_stock_full'))
        persist(report)
        return report
    except BaseException:
        report['status'] = 'failed'
        persist(report)
        raise
    finally:
        engine.force_path = None
        engine.synchronize()
        engine.clear_between_tasks()
