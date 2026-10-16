"""Workload-scoped KITE decision core. No physical cache, CUDA, or fallback execution.

MRC counts refer to a fixed-capacity, equal-size-object, admit-on-access LRU
reference stream. CPU bypass, expiry and resize are NOT simulated by this core.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean


class Fenwick:
    def __init__(self):
        self.size = 1
        self.values = [0, 0]
        self.tree = [0, 0]

    def ensure(self, index):
        if index <= self.size:
            return
        while self.size < index:
            self.size *= 2
        self.values.extend([0]*(self.size+1-len(self.values)))
        self.tree = self.values.copy()
        for i in range(1, self.size+1):
            parent = i+(i & -i)
            if parent <= self.size:
                self.tree[parent] += self.tree[i]

    def add(self, index, delta):
        self.ensure(index)
        self.values[index] += delta
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def prefix(self, index):
        total = 0
        while index:
            total += self.tree[index]
            index -= index & -index
        return total


class ReuseStack:
    """Distinct-object reuse distance in O(log accesses), amortized over resizing.

    Memory is O(total accesses) in this first implementation; no silent sampling.
    Periodic curve refresh does not reset stack history or drop intervening accesses.
    """
    def __init__(self):
        self.last = {}
        self.clock = 0
        self.bits = Fenwick()

    def access(self, key):
        previous = self.last.get(key)
        distance = None if previous is None else self.bits.prefix(self.clock)-self.bits.prefix(previous)
        if previous is not None:
            self.bits.add(previous, -1)
        self.clock += 1
        self.bits.add(self.clock, 1)
        self.last[key] = self.clock
        return distance


def is_miss(distance, capacity):
    if type(capacity) is not int or capacity < 0:
        raise ValueError('Capacity must be a nonnegative integer object count.')
    return distance is None or distance >= capacity


def miss_count(histogram, capacity):
    return sum(count for distance,count in histogram.items() if is_miss(distance,capacity))


def affine_fit(samples):
    if len(samples) < 2 or len({x for x,_ in samples}) < 2:
        raise ValueError('Affine calibration needs at least two distinct x values.')
    xm, ym = mean(x for x,_ in samples), mean(y for _,y in samples)
    slope = sum((x-xm)*(y-ym) for x,y in samples)/sum((x-xm)**2 for x,_ in samples)
    return ym-slope*xm, slope


@dataclass(frozen=True)
class CostModel:
    expert_upload_ms: float
    expert_cpu_intercept_ms: float
    expert_cpu_slope_ms: float
    kv_restore_ms: float
    kv_replay_intercept_ms: float
    kv_replay_slope_ms: float

    def expert(self, q):
        return self._valid(self.expert_upload_ms), self._valid(self.expert_cpu_intercept_ms+self.expert_cpu_slope_ms*q)

    def kv(self, context):
        return self._valid(self.kv_restore_ms), self._valid(self.kv_replay_intercept_ms+self.kv_replay_slope_ms*context)

    @staticmethod
    def _valid(value):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid cost prediction; do not silently clip or use it in a decision.')
        return value


class CostCalibration:
    """Collect actual elapsed-ms samples during ONE workload's startup prefix.

    This is an ingestion/fitting interface, NOT an implementation of measurements.
    The backend must provide matched recovery endpoints and actual synchronized times.
    """
    PATHS = ('expert_upload', 'expert_cpu', 'kv_restore', 'kv_replay')

    def __init__(self):
        self.samples = {path:[] for path in self.PATHS}
        self.frozen = False

    def add(self, path, x, elapsed_ms):
        if self.frozen:
            raise RuntimeError('Calibration is frozen for this workload.')
        if path not in self.samples or not all(math.isfinite(v) for v in (x,elapsed_ms)) or x <= 0 or elapsed_ms <= 0:
            raise ValueError('Invalid synchronized cost observation.')
        self.samples[path].append((x,elapsed_ms))

    def freeze(self):
        if self.frozen:
            raise RuntimeError('Calibration may only finish once per workload.')
        if not all(self.samples.values()):
            raise ValueError('All four recovery paths need actual measurements.')
        cpu_a,cpu_b = affine_fit(self.samples['expert_cpu'])
        kv_a,kv_b = affine_fit(self.samples['kv_replay'])
        result = CostModel(mean(y for _,y in self.samples['expert_upload']),cpu_a,cpu_b,
                           mean(y for _,y in self.samples['kv_restore']),kv_a,kv_b)
        for path,items in self.samples.items():
            for x,_ in items:
                result.expert(x) if path.startswith('expert') else result.kv(x)
        self.frozen = True
        return result


class History:
    def __init__(self):
        self.expert_stack, self.kv_stack = ReuseStack(), ReuseStack()
        self.clear_window()

    def clear_window(self):
        # Marking by routed q preserves the correlation between misses and cost.
        self.expert_bins = Counter()  # (reuse distance, observed q) -> count
        self.kv_groups = []           # one request/forward recovery task per group
        self.batches = 0

    def observe(self, experts, kv_groups):
        for key,q in experts:
            if type(q) is not int or q <= 0:
                raise ValueError('Expected positive actual routed-token count.')
            self.expert_bins[(self.expert_stack.access(key),q)] += 1
        seen = set()
        for group in kv_groups:
            identity = group['group_id']
            if identity in seen or group['context'] <= 0:
                raise ValueError('One KV recovery group per request/forward, positive context required.')
            seen.add(identity)
            objects = list(group['objects'])
            if len(set(objects)) != len(objects):
                raise ValueError('Duplicate KV object in one recovery group.')
            bins = Counter(self.kv_stack.access(key) for key in objects)
            self.kv_groups.append((group['context'],bins))
        self.batches += 1

    def score(self, expert_capacity, kv_capacity, costs):
        result = {'expert_misses':0, 'expert_upload_events':0, 'expert_cpu_events':0,
                  'kv_block_misses':0, 'kv_restore_events':0, 'kv_replay_tasks':0, 'recovery_ms':0.}
        for (distance,q),count in self.expert_bins.items():
            if not is_miss(distance,expert_capacity):
                continue
            upload,cpu = costs.expert(q)
            path = 'expert_cpu_events' if cpu < upload else 'expert_upload_events'
            result['expert_misses'] += count
            result[path] += count
            result['recovery_ms'] += count*min(upload,cpu)
        for context,bins in self.kv_groups:
            misses = miss_count(bins,kv_capacity)
            if not misses:
                continue
            restore,replay = costs.kv(context)
            result['kv_block_misses'] += misses
            if replay < misses*restore:
                result['kv_replay_tasks'] += 1
                result['recovery_ms'] += replay  # Never multiply a whole replay by block misses.
            else:
                result['kv_restore_events'] += misses
                result['recovery_ms'] += misses*restore
        return result


@dataclass(frozen=True)
class Candidate:
    expert_objects: int
    kv_objects: int
    expert_bytes: int
    kv_bytes: int
    slack_bytes: int


def candidates(budget,expert_bytes,kv_bytes,fractions):
    if min(budget,expert_bytes,kv_bytes) <= 0:
        raise ValueError('Positive byte budgets and object sizes are required.')
    choices = {}
    for fraction in fractions:
        if not 0 <= fraction <= 1:
            raise ValueError('Invalid split fraction.')
        expert = int(budget*fraction)//expert_bytes
        kv = (budget-expert*expert_bytes)//kv_bytes
        if expert >= 1 and kv >= 1:
            choices[(expert,kv)] = Candidate(expert,kv,expert*expert_bytes,kv*kv_bytes,
                                            budget-expert*expert_bytes-kv*kv_bytes)
    if not choices:
        raise ValueError('No feasible split leaves one object of each kind.')
    return sorted(choices.values(),key=lambda c:c.expert_bytes)


class Controller:
    """Decision-only controller. before_batch gets ONLY already scheduled P/D counts.

    after_batch receives observations, so the current batch's expert IDs/q cannot
    influence its own split. History scores are LRU-reference cost proxies, not
    actual bypass/resize execution or measured future miss counts.
    """
    def __init__(self, budget, expert_bytes=9*1024**2, kv_bytes=12*1024**2,
                 fractions=(.1,.25,.5,.75,.9), refresh_batches=16, current_weight=.65,
                 prefill_expert_share=.8, decode_expert_share=.2):
        if refresh_batches < 1 or not 0 <= current_weight <= 1:
            raise ValueError('Invalid refresh period or blend weight.')
        if not 0 <= decode_expert_share <= prefill_expert_share <= 1:
            raise ValueError('P/D expert-share priors must be ordered in [0,1].')
        self.budget = budget
        self.choices = candidates(budget,expert_bytes,kv_bytes,fractions)
        self.refresh_batches,self.current_weight = refresh_batches,current_weight
        self.prefill_share,self.decode_share = prefill_expert_share,decode_expert_share
        self.history,self.calibration = History(),CostCalibration()
        self.costs = None
        self.completed = self.refresh_count = self.last_refresh = 0
        self.proposal = min(self.choices,key=lambda c:abs(c.expert_bytes-budget*.5))
        self.last_scores = []
        self.pending = False

    def before_batch(self,prefill_requests,decode_requests):
        if self.pending:
            raise RuntimeError('The previous batch has not been observed.')
        if (any(type(x) is not int or x < 0 for x in (prefill_requests,decode_requests))
                or prefill_requests+decode_requests == 0):
            raise ValueError('Expected nonempty already scheduled request counts.')
        self.pending = True
        p = prefill_requests/(prefill_requests+decode_requests)
        signal_bytes = self.budget*(p*self.prefill_share+(1-p)*self.decode_share)
        if self.costs is None:
            target = self.proposal.expert_bytes
        else:
            target = (1-self.current_weight)*self.proposal.expert_bytes+self.current_weight*signal_bytes
        chosen = min(self.choices,key=lambda c:abs(c.expert_bytes-target))
        return {'stage':'calibration' if self.costs is None else 'decision_only',
                'batch_index':self.completed,'refresh_count':self.refresh_count,
                'prefill_request_fraction':p,'history_expert_bytes':self.proposal.expert_bytes,
                'current_signal_expert_bytes':signal_bytes,'blended_expert_bytes':target,
                'candidate':asdict(chosen),'physical_split_applied':False}

    def after_batch(self,experts,kv_groups):
        if not self.pending:
            raise RuntimeError('Call before_batch before observing actual accesses.')
        self.history.observe(experts,kv_groups)
        self.pending = False
        self.completed += 1
        if self.costs is not None and self.completed-self.last_refresh >= self.refresh_batches:
            self._refresh()

    def finish_calibration(self):
        if self.pending or self.completed == 0:
            raise RuntimeError('Finish an observed startup prefix before enabling KITE decisions.')
        self.costs = self.calibration.freeze()
        self._refresh()

    def _refresh(self):
        started = time.perf_counter_ns()
        self.last_scores = [{'candidate':asdict(c),**self.history.score(c.expert_objects,c.kv_objects,self.costs)}
                            for c in self.choices]
        best = min(range(len(self.choices)),key=lambda i:(self.last_scores[i]['recovery_ms'],
                   abs(self.choices[i].expert_bytes-self.proposal.expert_bytes)))
        self.proposal = self.choices[best]
        self.last_refresh = self.completed
        self.refresh_count += 1
        self.last_refresh_ms = (time.perf_counter_ns()-started)/1e6
        self.history.clear_window()  # Access stacks intentionally continue across windows.


class RecoveryLedger:
    """Record actual top-level recovery times; absorb explicitly marked macro children."""
    def __init__(self):
        self.events = {}
        self.total_ms = 0.

    def add(self,event_id,path,elapsed_ms,parent_id=None):
        if event_id in self.events or path not in CostCalibration.PATHS or not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError('Invalid or duplicate recovery event.')
        if parent_id is not None:
            if parent_id not in self.events or self.events[parent_id]['path'] != 'kv_replay':
                raise ValueError('Nested charges require an existing KV replay macro.')
        charged = elapsed_ms if parent_id is None else 0.
        self.events[event_id] = {'path':path,'elapsed_ms':elapsed_ms,'parent_id':parent_id,'charged_ms':charged}
        self.total_ms += charged
        return charged
