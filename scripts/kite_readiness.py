"""Fast read-only readiness checks: no torch import, CUDA context or model load."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kite_trace import GIB,check_gpu_budget,parse_gpu_pids,require_versions,resource_snapshot,run_readonly,utc_now
from qwen_smoke import verify_checkpoint


def verify_source_bundle(folder):
    folder = Path(folder).resolve()
    manifest_path = folder/'bundle_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    seen = set()
    for row in manifest['files']:
        relative = Path(row['path'])
        target = (folder/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(folder):
            raise RuntimeError('Unsafe source bundle manifest path.')
        if row['path'] in seen:
            raise RuntimeError('Duplicate source bundle file.')
        seen.add(row['path'])
        data = target.read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise RuntimeError(f'Packaged source changed or is incomplete: {row["path"]}')
    if not manifest['files']:
        raise RuntimeError('Empty source bundle manifest.')
    return dict(status='passed',files=len(seen),name=manifest['name'],
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())


def readiness(checkpoint, bundle_folder):
    checkpoint = Path(checkpoint).resolve()
    report = dict(time_utc=utc_now(),status='checking',no_cuda_context_created=True,
        no_model_loaded=True,checks={},errors=[],blockers=[],
        notes=['Readiness is a snapshot, not a GPU reservation.',
               'Checkpoint uses previously verified file sizes/mtime/hash records; does not rehash all model weights.',
               'A ready result does not execute GPU kernels: fused attention, numerical gates and long-context memory still require an idle-GPU run.'])
    for name,operation in [('source_bundle',lambda:verify_source_bundle(bundle_folder)),
                           ('software',require_versions),('checkpoint',lambda:verify_checkpoint(checkpoint))]:
        try:
            value = operation()
            if name=='checkpoint':
                value = dict(model_id=value['model_id'],source=value.get('source','huggingface'),
                    content_manifest_sha256=value.get('content_manifest_sha256'),weights_ready=value['weights_ready'])
            report['checks'][name] = value
        except Exception as error:
            report['errors'].append(dict(check=name,message=str(error)))
    try:
        snapshot = resource_snapshot(Path(bundle_folder).parent)
        report['resources'] = snapshot
        required = json.loads((checkpoint/'model.safetensors.index.json').read_text(encoding='utf-8'))['metadata']['total_size']+17*GIB
        available = snapshot['effective_available_bytes']
        report['checks']['host_memory'] = dict(required_bytes=required,available_bytes=available)
        if available is None or available<required:
            report['blockers'].append(dict(check='host_memory',message='Insufficient available RAM for model, 8 GiB CPU cache and 9 GiB reserve.'))
        if snapshot['disk']['free_bytes']<2*GIB:
            report['blockers'].append(dict(check='disk',message='Less than 2 GiB disk space available.'))
        affinity = snapshot['allowed_logical_cpus']
        if affinity is None or not set(range(24,32))<=set(affinity):
            report['errors'].append(dict(check='cpu_affinity',message='Required GPU-near CPU IDs 24..31 are unavailable.'))
        try:
            check_gpu_budget(snapshot,16,reserve_gib=4)
        except RuntimeError as error:
            report['blockers'].append(dict(check='gpu_budget',message=str(error)))
    except Exception as error:
        report['errors'].append(dict(check='resources',message=str(error)))
    raw = run_readonly(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
    report['checks']['gpu_pid_query'] = raw
    try:
        pids = sorted(parse_gpu_pids(raw))
        report['checks']['gpu_pids_before_cuda'] = pids
        if pids:
            report['blockers'].append(dict(check='gpu_pids',message=f'Existing GPU processes: {pids}; no CUDA/model started.'))
    except RuntimeError as error:
        report['errors'].append(dict(check='gpu_pid_query',message=str(error)))
    report['status'] = 'failed' if report['errors'] else 'blocked' if report['blockers'] else 'ready'
    return report


def require_ready(report):
    if report['status']!='ready':
        messages = [r['message'] for r in report['errors']+report['blockers']]
        raise RuntimeError('Readiness '+report['status']+': '+'; '.join(messages))
