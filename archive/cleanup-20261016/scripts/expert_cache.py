"""Bounded physical expert-cache driver, independent of Torch.

Backend slots must own exactly object_bytes of GPU weights. A separate scratch
slot serves uploads at capacity zero and correctness checks; it is NOT cache.
CPU misses bypass GPU admission and do not change resident LRU order.
"""
from __future__ import annotations

import math
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from statistics import mean

from kite_controller import affine_fit


@dataclass(frozen=True)
class ExpertCosts:
    upload_ms: float
    cpu_intercept_ms: float
    cpu_slope_ms: float
    fixed_cpu_ms: float
    min_q: int
    max_q: int

    def choose(self, q, fixed=False):
        if not self.min_q <= q <= self.max_q:
            return 'upload', 'outside_calibration_domain_upload'
        cpu = self.fixed_cpu_ms if fixed else self.cpu_intercept_ms+self.cpu_slope_ms*q
        if not math.isfinite(cpu) or cpu < 0:
            return 'upload', 'invalid_cpu_prediction_upload'
        return ('cpu' if cpu < self.upload_ms else 'upload'), 'fixed' if fixed else 'affine'


def fit_costs(samples):
    """Only the expert portion of workload-start calibration; not all four paths."""
    rows = list(samples)
    for row in rows:
        if (row['path'] not in ('upload', 'cpu') or type(row['q']) is not int
                or row['q'] <= 0 or not math.isfinite(row['wall_ms']) or row['wall_ms'] <= 0):
            raise ValueError('Invalid expert calibration measurement.')
    cpu = [(r['q'], r['wall_ms']) for r in rows if r['path'] == 'cpu']
    upload = [r['wall_ms'] for r in rows if r['path'] == 'upload']
    if not upload or {r['q'] for r in rows if r['path'] == 'upload'} != {q for q, _ in cpu}:
        raise ValueError('Both paths need matched q coverage.')
    intercept, slope = affine_fit(cpu)
    return ExpertCosts(mean(upload), intercept, slope, mean(y for _, y in cpu),
                       min(q for q, _ in cpu), max(q for q, _ in cpu))


class ExpertCache:
    """One synchronized expert invocation at a time; no overlap or concurrency.

    Backend API: allocate_slot(), release_slot(slot), upload(key,slot),
    gpu(slot,x), cpu(key,x), scratch_upload(key,x), synchronize().
    resize is outside the invocation timer but must be included in workload wall
    time by the caller. Caller must not retain slot-backed outputs across calls.
    """
    def __init__(self, backend, object_bytes, capacity, policy='upload', costs=None):
        if type(object_bytes) is not int or object_bytes <= 0:
            raise ValueError('Expected positive expert weight bytes.')
        if policy not in ('upload', 'cpu', 'calibrated', 'fixed'):
            raise ValueError('Unknown expert fallback policy.')
        if policy in ('calibrated', 'fixed') and costs is None:
            raise ValueError('This policy requires measured startup costs.')
        self.backend, self.object_bytes = backend, object_bytes
        self.policy, self.costs = policy, costs
        self.resident, self.free = OrderedDict(), []
        self.capacity = 0
        self.counts = Counter()
        self.closed = False
        self.resize(capacity)

    def _open(self):
        if self.closed:
            raise RuntimeError('Cache is closed.')

    @property
    def allocated_weight_bytes(self):
        return (len(self.resident)+len(self.free))*self.object_bytes

    def resize(self, capacity):
        self._open()
        if type(capacity) is not int or capacity < 0:
            raise ValueError('Capacity must be a nonnegative expert count.')
        self.backend.synchronize()
        evicted = []
        while len(self.resident) > capacity:
            key, slot = self.resident.popitem(last=False)
            self.free.append(slot)
            evicted.append(key)
        allocated = len(self.resident)+len(self.free)
        while allocated > capacity:
            self.backend.release_slot(self.free.pop())
            allocated -= 1
        try:
            while allocated < capacity:
                self.free.append(self.backend.allocate_slot())
                allocated += 1
        finally:
            # If allocation fails, retain a valid budget for already allocated slots.
            self.capacity = len(self.resident)+len(self.free)
        self.backend.synchronize()
        self._check()
        return {'capacity':self.capacity, 'evicted':evicted,
                'allocated_weight_bytes':self.allocated_weight_bytes}

    def _check(self):
        if len(self.resident)+len(self.free) != self.capacity:
            raise RuntimeError('GPU slot accounting violated.')

    def execute(self, key, q, x):
        self._open()
        if type(q) is not int or q <= 0:
            raise ValueError('Only nonempty expert calls enter the physical cache.')
        self.backend.synchronize()
        started = time.perf_counter_ns()
        hit = key in self.resident
        path, reason = ('hit', 'resident') if hit else (self.policy, 'forced_smoke_policy')
        if not hit and self.policy in ('calibrated', 'fixed'):
            path, reason = self.costs.choose(q, fixed=self.policy == 'fixed')
        evicted, admitted = None, False
        if hit:
            slot = self.resident[key]
            output = self.backend.gpu(slot, x)
            self.resident.move_to_end(key)
        elif path == 'cpu':
            output = self.backend.cpu(key, x)
        elif self.capacity == 0:
            output = self.backend.scratch_upload(key, x)
        else:
            if self.free:
                slot = self.free.pop()
            else:
                evicted, slot = self.resident.popitem(last=False)
            try:
                self.backend.upload(key, slot)
                output = self.backend.gpu(slot, x)
            except BaseException:
                # A partially overwritten slot must never remain labelled as its old key.
                self.free.append(slot)
                self._check()
                raise
            self.resident[key] = slot
            admitted = True
        self.backend.synchronize()
        elapsed = (time.perf_counter_ns()-started)/1e6
        self.counts['calls'] += 1
        self.counts['hits' if hit else 'misses'] += 1
        if not hit:
            self.counts[path] += 1
        self._check()
        return output, {'key':key, 'q':q, 'hit':hit, 'path':path, 'reason':reason,
                        'admitted':admitted, 'evicted':evicted, 'wall_ms':elapsed,
                        'capacity':self.capacity, 'resident_lru':list(self.resident),
                        'resident_weight_bytes':len(self.resident)*self.object_bytes,
                        'allocated_weight_bytes':self.allocated_weight_bytes}

    def close(self):
        if self.closed:
            return
        self.resize(0)
        self.closed = True
