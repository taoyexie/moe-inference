"""Audit the second exploration's actual clocks, memory, routes and startup costs."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_kite_e2e import audit_task,read,require,same
from kite_advantage import OPTIONS,PROTOCOL,choose_selection,implementation_hash,protocol_hash,specifications,structural_screen


def audit(folder):
    root = read(folder/'summary.json')
    require(root['status']=='passed','Run is incomplete or failed.')
    require(root['protocol']==PROTOCOL and root['protocol_sha256']==protocol_hash(),'Wrong frozen protocol.')
    require(root['runtime_options']==OPTIONS,'Runtime options differ.')
    require(root['attention_gate']==dict(status='passed',backend='EFFICIENT_ATTENTION',mask_columns=8),
            'Required fused attention capability gate was not passed.')
    require(root['implementation_sha256']==implementation_hash(),'Executed source hash differs.')
    ready = read(folder/'readiness.json')
    require(ready['status']=='ready' and ready['no_cuda_context_created'] and ready['no_model_loaded'],
            'Pre-CUDA readiness check was not passed.')
    require(ready['checks']['gpu_pids_before_cuda']==[],'GPU already had another process before CUDA startup.')
    work = read(folder/'workload.json')
    require(hashlib.sha256((folder/'workload.json').read_bytes()).hexdigest()==root['workload_sha256'],'Workload bytes differ.')
    specs = work['requests']
    expected = specifications()
    require(len(specs)==len(expected),'Missing requests.')
    for actual,wanted in zip(specs,expected):
        for k,v in wanted.items():
            require(actual[k]==v,'Request shape/release differs.')
        require(len(actual['input_ids'])==actual['input_tokens'],'Prompt length differs.')
    require(root['command'] in ('search','confirm'),'Wrong run command.')
    require(work['split']==('search' if root['command']=='search' else 'confirmation'),'Wrong topic split.')
    same(read(folder/'structural_screen.json'),structural_screen())
    validation = read(folder/'validation.json')
    require(validation['status']=='passed' and len(validation['checks'])==3,'Numerical integration gate did not pass.')
    monitor = read(folder/'resource_monitor.json')
    allowed = set(monitor['allowed_gpu_pids'])
    require(monitor['error'] is None and len(allowed)==1 and monitor['samples'],'Resource monitoring missing or failed.')
    require(all(set(r['gpu_pids'])==allowed for r in monitor['samples']),'Another GPU job or missing own PID was observed.')
    require(monitor['startup']['status']=='passed' and read(folder/'gpu_monitor_startup.json')==monitor['startup'],
            'GPU monitoring did not start successfully before loading.')
    pair = {}
    for mode in ('baseline','kite'):
        perf = audit_task(folder/f'8192-{mode}',specs,mode,8192,runtime_options=OPTIONS)
        require(perf==next(r for r in root['tasks'] if r['mode']==mode),'Root/child performance differs.')
        require(perf['cuda_peak_allocated_bytes']<=root['resources_before']['gpus'][0]['total_bytes']-4*1024**3,
                'GPU peak exceeded the declared safety reserve.')
        pair[mode] = perf
    require(len(root['tasks'])==2,'Unexpected extra comparisons.')
    if root['command']=='search':
        selected = read(folder/'selection.json')
        for key,value in choose_selection(root['tasks']).items():
            same(selected[key],value)
    else:
        frozen = root['frozen_selection']
        require(frozen['status']=='selected' and frozen['protocol_sha256']==protocol_hash()
                and frozen['implementation_sha256']==implementation_hash()
                and frozen['model_manifest_sha256']==root['model_manifest_sha256'],'Unfrozen confirmation.')
    a,b = pair['baseline']['workload_total_ms'],pair['kite']['workload_total_ms']
    return dict(status='passed',run=str(folder.resolve()),command=root['command'],seed=root['args']['seed'],
        budget_mib=8192,baseline_ms=a,kite_ms=b,speedup=a/b,reduction=1-b/a,
        outputs_identical=[r['generated_ids'] for r in read(folder/'8192-baseline'/'requests.json')]
                       ==[r['generated_ids'] for r in read(folder/'8192-kite'/'requests.json')],
        main_baseline_ms=pair['baseline']['execution_inclusive_ms'],main_kite_ms=pair['kite']['execution_inclusive_ms'],
        expert={m:p['expert_counts'] for m,p in pair.items()},kv={m:p['kv_counts'] for m,p in pair.items()},
        scope='selected controlled long-context KV shock; all startup costs included; no independent module attribution',
        raw_sha256={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(folder.rglob('*')) if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    require(not args.output.exists(),'Refusing to overwrite an audit.')
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text = ('# KITE 长上下文突发场景实测核验\n\n'
        f'阶段：{result["command"]}；seed：{result["seed"]}；8 GiB 共享缓存。\n\n'
        '| 基线 E2E s | KITE E2E s | 加速比 | 耗时降低 |\n|---:|---:|---:|---:|\n'
        f'| {result["baseline_ms"]/1000:.3f} | {result["kite_ms"]/1000:.3f} | '
        f'{result["speedup"]:.3f}× | {100*result["reduction"]:.2f}% |\n\n'
        '包含本 workload 启动标定、池初始化、在线决策、实际推理及搬运。模型加载单列。'
        '这是主动选择的受控场景，不是线上流量，不是原始 WiSP，不单独归因 P/D 或 fallback。\n')
    (args.output/'report.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw_sha256','expert','kv')},ensure_ascii=False))


if __name__=='__main__':
    main()
