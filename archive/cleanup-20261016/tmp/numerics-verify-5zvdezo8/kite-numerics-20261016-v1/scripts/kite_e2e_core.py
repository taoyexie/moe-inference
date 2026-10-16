"""Dependency-free scheduling, shared frame ownership and cost policy for KITE E2E.

Frame ownership is real only when paired with TorchArena. Reference MRCs are
historical admit-on-access LRU statistics, not exact bypass/expiry predictions.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from statistics import mean, median

FRAME_BYTES = 3*1024**2
EXPERT_BYTES = 3*FRAME_BYTES
KV_BYTES = 4*FRAME_BYTES
WIDTHS = {'expert':3,'kv':4}


class FramePool:
    def __init__(self, budget_bytes):
        if type(budget_bytes) is not int or budget_bytes < EXPERT_BYTES+KV_BYTES:
            raise ValueError('Pool must fit one expert and one KV page.')
        self.frame_count = budget_bytes//FRAME_BYTES
        self.usable_bytes = self.frame_count*FRAME_BYTES
        self.free = list(range(self.frame_count))
        self.entries = {kind:OrderedDict() for kind in WIDTHS}
        self.limits = {kind:0 for kind in WIDTHS}

    def remove(self, kind, key):
        entry = self.entries[kind].pop(key,None)
        if entry is not None:
            self.free.extend(entry['frames'])
        return entry

    def set_limits(self, expert, kv):
        limits = dict(expert=expert,kv=kv)
        if any(type(v) is not int or v<0 for v in limits.values()):
            raise ValueError('Object capacities must be nonnegative integers.')
        if sum(limits[k]*WIDTHS[k] for k in WIDTHS)>self.frame_count:
            raise ValueError('Expert and KV quotas exceed the shared pool.')
        evicted = []
        for kind in WIDTHS:
            while len(self.entries[kind])>limits[kind]:
                key = next(iter(self.entries[kind]))
                self.remove(kind,key)
                evicted.append((kind,key))
        self.limits = limits
        return evicted

    def get(self, kind, key, touch=True):
        entry = self.entries[kind].get(key)
        if entry is not None and touch:
            self.entries[kind].move_to_end(key)
        return entry

    def admit(self, kind, key, version=0):
        existing = self.get(kind,key)
        if existing is not None:
            return existing,None
        if self.limits[kind]==0:
            return None,None
        evicted = None
        if len(self.entries[kind])>=self.limits[kind]:
            evicted = next(iter(self.entries[kind]))
            self.remove(kind,evicted)
        width = WIDTHS[kind]
        if len(self.free)<width:
            raise RuntimeError('Frame accounting violated despite feasible quotas.')
        entry = {'frames':tuple(self.free.pop() for _ in range(width)), 'version':version}
        self.entries[kind][key] = entry
        return entry,evicted

    def clear(self):
        for kind in WIDTHS:
            for key in list(self.entries[kind]):
                self.remove(kind,key)

    def snapshot(self):
        return dict(usable_bytes=self.usable_bytes,free_bytes=len(self.free)*FRAME_BYTES,
                    resident_bytes={k:len(v)*WIDTHS[k]*FRAME_BYTES for k,v in self.entries.items()},
                    limits=dict(self.limits),objects={k:len(v) for k,v in self.entries.items()})


class RuntimeCosts:
    def __init__(self, calibration, fitted, mode, robust=False, cpu_cold_setup_ms=0.):
        if mode not in ('baseline','kite'):
            raise ValueError('Expected baseline or kite.')
        if not math.isfinite(cpu_cold_setup_ms) or cpu_cold_setup_ms<0:
            raise ValueError('CPU cold setup must be an actually measured nonnegative cost.')
        samples = calibration.samples
        if robust:
            # Same predeclared estimator for both policies. Retain all raw samples.
            from kite_controller import CostCalibration
            grouped = CostCalibration()
            for path,rows in samples.items():
                for x in sorted({x for x,_ in rows}):
                    grouped.add(path,x,median(y for at,y in rows if at==x))
            fitted = grouped.freeze()
            samples = grouped.samples
        self.mode,self.fitted = mode,fitted
        self.cpu_cold_setup_ms = cpu_cold_setup_ms
        self.estimator = 'median at each x, then equal-weight x groups' if robust else 'raw sample means / OLS'
        self.fixed = {p:mean(y for _,y in rows) for p,rows in samples.items()}
        self.domains = {p:(min(x for x,_ in rows),max(x for x,_ in rows))
                        for p,rows in calibration.samples.items()}

    def predicted(self, path, x):
        lo,hi = self.domains[path]
        if not lo<=x<=hi:
            return math.inf
        if self.mode=='baseline':
            return self.fixed[path]
        model = self.fitted
        value = {'expert_cpu':lambda:model.expert_cpu_intercept_ms+model.expert_cpu_slope_ms*x,
                 'kv_replay':lambda:model.kv_replay_intercept_ms+model.kv_replay_slope_ms*x}[path]()
        return value if math.isfinite(value) and value>=0 else math.inf

    def expert(self, q):
        # Historical q-only evidence does not establish that a CPU copy exists.
        return self.expert_for_cache(q,cpu_cached=False)

    def expert_for_cache(self, q, cpu_cached):
        return self.fixed['expert_upload'],self.predicted('expert_cpu',q)+(0. if cpu_cached else self.cpu_cold_setup_ms)

    def kv(self, context):
        return self.fixed['kv_restore'],self.predicted('kv_replay',context)


@dataclass
class Request:
    request_id: str
    prompt_ids: list
    release_step: int
    output_tokens: int
    generated: list = field(default_factory=list)
    kv_length: int = 0
    cpu_pages: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    submitted_ns: int | None = None
    token_ticks: list = field(default_factory=list)

    @property
    def done(self):
        return len(self.generated)>=self.output_tokens

    @property
    def query(self):
        return [self.generated[-1]] if self.generated else self.prompt_ids

    @property
    def prefix(self):
        # The last sampled token has not yet entered the model/KV.
        return (self.prompt_ids+self.generated)[:self.kv_length]


def validate_requests(requests):
    if not requests or len({r.request_id for r in requests})!=len(requests):
        raise ValueError('Expected nonempty requests with unique IDs.')
    for req in requests:
        if (not req.prompt_ids or not all(type(v) is int and v>=0 for v in req.prompt_ids)
                or type(req.output_tokens) is not int or req.output_tokens<1
                or type(req.release_step) is not int or req.release_step<0):
            raise ValueError('Invalid request specification.')


def ready_batch(requests, step):
    active = [r for r in requests if not r.done and r.release_step<=step]
    if len(active)>4:
        raise RuntimeError('Workload exceeds four active requests; never silently drop a request.')
    return active


def packed_layout(requests):
    if not requests or len({r.request_id for r in requests})!=len(requests):
        raise ValueError('Expected distinct requests in a nonempty batch.')
    layout,qs,ks = [],0,0
    for req in requests:
        if req.done or req.kv_length != (len(req.prompt_ids)+len(req.generated)-1 if req.generated else 0):
            raise ValueError('Request tokens and KV length are inconsistent.')
        q = len(req.query)
        layout.append(dict(request_id=req.request_id,q_start=qs,q_end=qs+q,
                           k_start=ks,k_end=ks+req.kv_length+q,before=req.kv_length,
                           after=req.kv_length+q,phase='decode' if req.generated else 'prefill'))
        qs += q
        ks += req.kv_length+q
    return layout


def allowed_attention(row, query_index, key_index):
    return (row['q_start']<=query_index<row['q_end']
            and row['k_start']<=key_index<row['k_end']
            and key_index-row['k_start']<=row['before']+query_index-row['q_start'])
