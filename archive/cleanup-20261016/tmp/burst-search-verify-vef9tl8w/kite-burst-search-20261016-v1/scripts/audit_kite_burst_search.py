"""Audit measured short-prefill burst E2E; never infer times from miss costs."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_kite_e2e import audit_task,read,require,same
from kite_advantage import performance_validation_record
from kite_burst_search import CASES,OPTIONS,PROTOCOL,implementation_hash,mode_order,protocol_hash,selection,structural_screen


def audit(folder):
    folder = Path(folder)
    root = read(folder/'summary.json')
    require(root['status']=='passed','Run failed or incomplete.')
    require(root['protocol']==PROTOCOL and root['protocol_sha256']==protocol_hash(),'Wrong frozen protocol.')
    require(root['implementation_sha256']==implementation_hash() and root['runtime_options']==OPTIONS,'Wrong sources/options.')
    ready = read(folder/'readiness.json')
    require(ready['status']=='ready' and ready['checks']['gpu_pids_before_cuda']==[],'Pre-CUDA resources were not clear.')
    require(read(folder/'validation.json')==performance_validation_record(),'Wrong performance policy.')
    work = read(folder/'workload.json')
    require(hashlib.sha256((folder/'workload.json').read_bytes()).hexdigest()==root['workload_sha256'],'Workload modified.')
    require(work['protocol']==PROTOCOL and work['split']==root['command'],'Wrong input split.')
    monitor = read(folder/'resource_monitor.json')
    require(monitor['error'] is None and len(monitor['allowed_gpu_pids'])==1 and monitor['samples'],'Missing/failed resource monitoring.')
    require(all(r['gpu_pids']==monitor['allowed_gpu_pids'] for r in monitor['samples']),'Resource interference observed.')
    require(read(folder/'gpu_monitor_startup.json')==monitor['startup'] and monitor['startup']['status']=='passed','Bad monitor startup.')
    completed = {t['case_name'] for t in root['tasks']}
    require(root['command'] in ('search','confirm'),'Wrong command.')
    if root['command']=='search':
        require(completed in ({CASES[0]['name']},{c['name'] for c in CASES}),'Unexpected case search.')
    else:
        frozen = root['frozen_selection']
        require(frozen['status']=='selected' and frozen['implementation_sha256']==implementation_hash()
            and frozen['protocol_sha256']==protocol_hash() and frozen['model_manifest_sha256']==root['model_manifest_sha256'],
            'Confirmation is not frozen.')
        require(completed=={frozen['selected']['case_name']},'Confirmation ran extra cases.')
    report = []
    expected_order = []
    for i,case in enumerate(CASES):
        if case['name'] not in completed:
            continue
        specs = work['cases'][case['name']]
        require(len(specs)==16 and specs[0]['input_tokens']==512,'Wrong request count/initial request.')
        expected_text = case['search_prompt'] if root['command']=='search' else case['confirm_prompt']
        for step,spec in enumerate(specs):
            require(spec['release_step']==step and spec['output_tokens']==1 and spec['source_text']==expected_text,'Wrong declared input/release/output.')
            require(len(spec['input_ids'])==spec['input_tokens'],'Prompt length mismatch.')
            if step:
                require(spec['input_ids']==specs[1]['input_ids'],'Hot prompts are not the same genuine input.')
        same(read(folder/'structural_screen.json')[case['name']],structural_screen(case,specs))
        pair = {}
        for mode in mode_order(i,root['args']['seed']):
            expected_order.append((case['name'],mode))
            path = folder/f'{case["name"]}-{case["budget_mib"]}-{mode}'
            perf = audit_task(path,specs,mode,case['budget_mib'],runtime_options=OPTIONS)
            record, = [t for t in root['tasks'] if (t['case_name'],t['mode'])==(case['name'],mode)]
            require({k:v for k,v in record.items() if k!='case_name'}==perf,'Root/child actual performance differs.')
            require(perf['cuda_peak_allocated_bytes']<=root['resources_before']['gpus'][0]['total_bytes']-4*1024**3,'GPU safety reserve exceeded.')
            pair[mode] = perf
        a,b = pair['baseline']['workload_total_ms'],pair['kite']['workload_total_ms']
        report.append(dict(case_name=case['name'],budget_mib=case['budget_mib'],baseline_ms=a,kite_ms=b,
            speedup=a/b,reduction=1-b/a,main_baseline_ms=pair['baseline']['execution_inclusive_ms'],
            main_kite_ms=pair['kite']['execution_inclusive_ms'],output_tokens=16,
            expert_counts={m:p['expert_counts'] for m,p in pair.items()},kv_counts={m:p['kv_counts'] for m,p in pair.items()}))
    require([(t['case_name'],t['mode']) for t in root['tasks']]==expected_order,'Actual pair order differs.')
    if root['command']=='search':
        chosen = read(folder/'selection.json')
        for k,v in selection(root['tasks']).items():
            same(chosen[k],v)
        require((len(report)==1)==(report[0]['speedup']>=1.10),'Search stopping differs from predeclared rule.')
    return dict(status='passed',run=str(folder.resolve()),command=root['command'],seed=root['args']['seed'],
        cases=report,accuracy_certified=False,scope=PROTOCOL['phase_scope'],
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
    (args.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# KITE 短提示突发端到端实测','',
        '| 场景 | 共享池 GiB | 基线 E2E s | KITE E2E s | 加速比 | 耗时下降 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in result['cases']:
        lines.append(f'| {r["case_name"]} | {r["budget_mib"]/1024:g} | {r["baseline_ms"]/1000:.3f} | '
            f'{r["kite_ms"]/1000:.3f} | {r["speedup"]:.3f}× | {100*r["reduction"]:.2f}% |')
    lines += ['', '计入本 workload 启动标定、池初始化、实际生成和在线决策。模型加载另列。',
        '这是重复短提示、首 token 完成的受控突发，不是 decode-heavy 流量，不是原始 WiSP，也不单独归因各模块。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='raw_sha256'},ensure_ascii=False))


if __name__=='__main__':
    main()
