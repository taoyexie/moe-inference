"""Read-only resource probes and logical trace bookkeeping; no torch import."""
from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024 ** 3
PINNED = {
    'transformers': '4.51.3', 'accelerate': '1.6.0',
    'huggingface-hub': '0.30.2', 'safetensors': '0.5.3',
    'tokenizers': '0.21.1', 'numpy': '2.2.6',
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def software_versions():
    versions = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ['torch', *PINNED, 'psutil', 'modelscope']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def require_versions():
    versions = software_versions()
    wrong = [f'{name}: expected {v}, found {versions[name]}'
             for name, v in PINNED.items() if versions[name] != v]
    if versions['torch'] != '2.7.1+cu118':
        wrong.append('torch: retain the validated 2.7.1+cu118 environment')
    if wrong:
        raise RuntimeError('Dependency check failed:\n' + '\n'.join(wrong))
    return versions


def read_optional(path):
    try:
        return Path(path).read_text(encoding='utf-8').strip()
    except (OSError, UnicodeError):
        return None


def parse_meminfo(text):
    out = {}
    for line in (text or '').splitlines():
        match = re.fullmatch(r'(\w+):\s+(\d+)\s+kB', line.strip())
        if match:
            out[match[1]] = int(match[2]) * 1024
    return out


def cgroup_memory():
    """Inspect visible cgroup ancestors, not just host physical memory."""
    root = Path('/sys/fs/cgroup')
    candidates = {root}
    memberships = read_optional('/proc/self/cgroup') or ''
    for line in memberships.splitlines():
        pieces = line.split(':', 2)
        if len(pieces) != 3:
            continue
        _, controllers, member = pieces
        if controllers not in ('', 'memory') and 'memory' not in controllers.split(','):
            continue
        mount = root if controllers == '' else root / 'memory'
        candidate = (mount / member.lstrip('/')).resolve()
        if candidate.is_relative_to(mount.resolve()):
            while candidate.is_relative_to(mount.resolve()):
                candidates.add(candidate)
                if candidate == mount.resolve():
                    break
                candidate = candidate.parent
        candidates.add(mount)
    records = []
    for folder in sorted(candidates):
        for limit_name, usage_name in [('memory.max', 'memory.current'),
                                       ('memory.limit_in_bytes', 'memory.usage_in_bytes')]:
            limit, usage = read_optional(folder / limit_name), read_optional(folder / usage_name)
            if limit is None or usage is None:
                continue
            finite = int(limit) if limit.isdigit() and int(limit) < (1 << 60) else None
            current = int(usage) if usage.isdigit() else None
            records.append({'path': str(folder), 'limit_bytes': finite,
                            'usage_bytes': current,
                            'headroom_bytes': max(0, finite-current)
                            if finite is not None and current is not None else None})
    return records


def run_readonly(command):
    try:
        p = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
        return {'returncode': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'returncode': -1, 'stdout': '', 'stderr': str(e)}


def parse_gpu_pids(result):
    """Strict PID-only query parsing. Unknown output must never mean idle."""
    if result['returncode']!=0:
        raise RuntimeError(f'GPU PID query failed: {result!r}')
    lines = [s.strip() for s in result['stdout'].splitlines() if s.strip()]
    if any(not s.isascii() or not s.isdigit() or int(s)<1 for s in lines):
        raise RuntimeError(f'GPU PID query returned unrecognized output: {result!r}')
    return set(lines)


def parse_gpus(text):
    records = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) != 6:
            continue
        index, name, total, used, free, util = [x.strip() for x in row]
        try:
            records.append({'index': int(index), 'name': name,
                            'total_bytes': int(total)*1024**2, 'used_bytes': int(used)*1024**2,
                            'free_bytes': int(free)*1024**2, 'util_percent': int(util)})
        except ValueError:
            continue
    return records


def resource_snapshot(path):
    path = Path(path).resolve()
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    memory = parse_meminfo(read_optional('/proc/meminfo'))
    groups = cgroup_memory()
    available = memory.get('MemAvailable')
    headrooms = [r['headroom_bytes'] for r in groups if r['headroom_bytes'] is not None]
    limits = ([available] if available is not None else []) + headrooms
    gpu_cmd = run_readonly(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
                            '--format=csv,noheader,nounits'])
    disk = shutil.disk_usage(existing)
    return {'time_utc': utc_now(), 'software': software_versions(), 'meminfo': memory,
            'cgroup_memory': groups, 'effective_available_bytes': min(limits) if limits else None,
            'allowed_logical_cpus': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'cpu_quota': read_optional('/sys/fs/cgroup/cpu.max'),
            'gpu_query': gpu_cmd, 'gpus': parse_gpus(gpu_cmd['stdout']),
            'gpu_processes': run_readonly(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                          '--format=csv,noheader,nounits']),
            'disk': {'path': str(existing), 'total_bytes': disk.total, 'free_bytes': disk.free},
            'notes': ['No CUDA context or model loaded by this probe.',
                      'A snapshot does not reserve resources; hidden host cgroup limits may not be visible.']}


def check_gpu_budget(snapshot, budget_gib, reserve_gib=4, allow_shared=False):
    if len(snapshot['gpus']) != 1:
        raise RuntimeError('This smoke runner expects exactly one visible server GPU.')
    gpu = snapshot['gpus'][0]
    if not allow_shared and (gpu['used_bytes'] > 1024**3 or gpu['util_percent'] > 0):
        raise RuntimeError('GPU appears busy. Do not stop anyone else\'s process. Recheck or arrange a window.')
    if gpu['free_bytes'] < (budget_gib + reserve_gib) * GIB:
        raise RuntimeError('Free GPU memory is below the weight budget plus safety reserve.')


def mapped_device(name, mapping):
    matches = [key for key in mapping if key == '' or name == key or name.startswith(key + '.')]
    if not matches:
        raise RuntimeError(f'No placement found for {name}')
    return str(mapping[max(matches, key=len)])


class ExpertRecorder:
    """Record nonempty Python expert invocations, NOT actual physical misses."""
    def __init__(self, layer_ids, top_k, request_id='smoke-0001'):
        self.layer_ids = set(layer_ids)
        self.top_k = top_k
        self.request_id = request_id
        self.sequence = 0
        self.context = None
        self.events = []
        self.empty_calls = 0
        self.counts = defaultdict(int)

    def begin(self, step, phase, query_tokens, context_before):
        if self.context is not None:
            raise RuntimeError('Previous trace step was not finished.')
        self.context = {'request_id': self.request_id, 'step': step, 'phase': phase,
                        'query_tokens': query_tokens, 'context_before': context_before}
        self.events = []
        self.empty_calls = 0
        self.counts.clear()

    def observe(self, layer, expert, x, weight_bytes, placement):
        if self.context is None:
            raise RuntimeError('Expert invoked outside a labelled forward.')
        if len(x.shape) != 2:
            raise RuntimeError(f'Unexpected expert input shape {tuple(x.shape)}')
        tokens = int(x.shape[0])
        if not tokens:
            self.empty_calls += 1
            return
        self.counts[layer] += tokens
        self.events.append({**self.context, 'sequence': self.sequence,
                            'layer': layer, 'expert': expert, 'routed_tokens': tokens,
                            'weight_bytes': weight_bytes, 'weight_storage_placement': placement,
                            'input_device': str(x.device),
                            'semantics': 'nonempty_expert_module_invocation'})
        self.sequence += 1

    def finish(self):
        if self.context is None:
            raise RuntimeError('No active trace step.')
        expected = self.context['query_tokens'] * self.top_k
        rows = [{'layer': layer, 'expected': expected, 'observed': self.counts.get(layer, 0),
                 'pass': self.counts.get(layer, 0) == expected} for layer in sorted(self.layer_ids)]
        result = {**self.context, 'checks': rows, 'pass': all(x['pass'] for x in rows),
                  'nonempty_calls': len(self.events), 'ignored_empty_calls': self.empty_calls}
        events = self.events
        self.context = None
        return events, result


def cache_snapshot(cache, expected_layers, expected_length, heads, head_dim):
    if not hasattr(cache, 'key_cache') or not hasattr(cache, 'value_cache'):
        raise RuntimeError('Expected the pinned Transformers DynamicCache API.')
    if len(cache.key_cache) != expected_layers or len(cache.value_cache) != expected_layers:
        raise RuntimeError('KV layer count is inconsistent with the model.')
    rows = []
    for i, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
        expected_shape = (1, heads, expected_length, head_dim)
        if tuple(key.shape) != expected_shape or tuple(value.shape) != expected_shape:
            raise RuntimeError(f'KV shape mismatch at layer {i}: {tuple(key.shape)}, {tuple(value.shape)}')
        rows.append({'layer': i, 'key_shape': list(key.shape), 'value_shape': list(value.shape),
                     'key_dtype': str(key.dtype), 'value_dtype': str(value.dtype),
                     'key_device': str(key.device), 'value_device': str(value.device),
                     'logical_tensor_bytes': key.numel()*key.element_size() + value.numel()*value.element_size()})
    return {'length': expected_length, 'layers': rows,
            'logical_tensor_bytes': sum(r['logical_tensor_bytes'] for r in rows),
            'semantics': 'actual_KV_tensor_metadata_not_block_access_or_physical_miss_trace'}
