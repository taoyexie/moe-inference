"""Read-only audit of actual parallel-decode E2E clocks and cache mechanisms."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_kite_e2e import audit_task,read,require,same
from kite_advantage import OPTIONS,SEARCH_TOPICS,CONFIRM_TOPICS,performance_validation_record
from kite_parallel_decode import CASES,PROTOCOL,implementation_hash,mode_order,protocol_hash,selection,structural_screen


def audit(folder):
    folder = Path(folder)
    root = read(folder/'summary.json')
    require(root['status']=='passed' and root['command'] in ('search','confirm'),'Incomplete/failed run.')
    require(root['protocol']==PROTOCOL and root['protocol_sha256']==protocol_hash(),'Protocol differs.')
    require(root['implementation_sha256']==implementation_hash() and root['runtime_options']==OPTIONS,'Wrong sources/runtime.')
    ready = read(folder/'readiness.json')
    require(ready['status']=='ready' and ready['checks']['gpu_pids_before_cuda']==[],'Resources were not clear before CUDA.')
    require(read(folder/'validation.json')==performance_validation_record(),'Wrong performance trial policy.')
    same(read(folder/'structural_screen.json'),structural_screen())
    work = read(folder/'workload.json')
    require(hashlib.sha256((folder/'workload.json').read_bytes()).hexdigest()==root['workload_sha256'],'Workload modified.')
    require(work['protocol']==PROTOCOL and work['split']==root['command'],'Wrong workload split.')
    specs = work['requests']
    require(len(specs)==4,'Wrong request count.')
    topics = SEARCH_TOPICS if root['command']=='search' else CONFIRM_TOPICS
    for i,spec in enumerate(specs):
        require(spec['request_id']==f'parallel-{i}' and spec['release_step']==0 and spec['output_tokens']==16
            and spec['input_tokens']==len(spec['input_ids'])==PROTOCOL['requests']['input_tokens'][i]
            and spec['source_text']==topics[i],'Request schedule/text differs.')
    monitor = read(folder/'resource_monitor.json')
    require(monitor['error'] is None and len(monitor['allowed_gpu_pids'])==1 and monitor['samples'],'Missing/failed monitoring.')
    require(all(r['gpu_pids']==monitor['allowed_gpu_pids'] for r in monitor['samples']),'Observed GPU interference.')
    require(read(folder/'gpu_monitor_startup.json')==monitor['startup'] and monitor['startup']['status']=='passed','Bad monitor startup.')
    active = list(enumerate(CASES))
    if root['command']=='confirm':
        frozen = root['frozen_selection']
        require(frozen['status']=='selected' and frozen['implementation_sha256']==implementation_hash()
            and frozen['protocol_sha256']==protocol_hash() and frozen['model_manifest_sha256']==root['model_manifest_sha256'],'Unfrozen confirmation.')
        active = [(i,c) for i,c in active if c['name']==frozen['selected']['case_name']]
        require(len(active)==1,'Unknown frozen scene.')
    report,order = [],[]
    for index,case in active:
        pair,splits = {},{}
        for mode in mode_order(index,root['args']['seed']):
            order.append((case['name'],mode))
            task_folder = folder/f'{case["name"]}-{case["budget_mib"]}-{mode}'
            perf = audit_task(task_folder,specs,mode,case['budget_mib'],runtime_options=OPTIONS)
            matches = [t for t in root['tasks'] if (t['case_name'],t['mode'])==(case['name'],mode)]
            require(len(matches)==1 and {k:v for k,v in matches[0].items() if k!='case_name'}==perf,'Root/child record differs.')
            require(perf['cuda_peak_allocated_bytes']<=root['resources_before']['gpus'][0]['total_bytes']-4*1024**3,'GPU safety reserve exceeded.')
            pair[mode] = perf
            batches = [json.loads(s) for s in (task_folder/'batches.jsonl').read_text(encoding='utf-8').splitlines()]
            splits[mode] = [dict(step=r['step'],expert=r['pool']['limits']['expert'],kv=r['pool']['limits']['kv'],
                history_expert_bytes=r['decision']['history_expert_bytes']) for r in batches]
        a,b = pair['baseline'],pair['kite']
        report.append(dict(case_name=case['name'],budget_mib=case['budget_mib'],baseline_ms=a['workload_total_ms'],
            kite_ms=b['workload_total_ms'],speedup=a['workload_total_ms']/b['workload_total_ms'],
            main_baseline_ms=a['execution_inclusive_ms'],main_kite_ms=b['execution_inclusive_ms'],
            startup_baseline_ms=a['initialization_ms']+a['startup_inclusive_ms'],
            startup_kite_ms=b['initialization_ms']+b['startup_inclusive_ms'],output_tokens=64,
            expert_counts={m:p['expert_counts'] for m,p in pair.items()},kv_counts={m:p['kv_counts'] for m,p in pair.items()},splits=splits))
    require([(t['case_name'],t['mode']) for t in root['tasks']]==order,'Wrong/extra pair order.')
    if root['command']=='search':
        chosen = read(folder/'selection.json')
        for key,value in selection(root['tasks']).items():
            same(chosen[key],value)
    return dict(status='passed',run=str(folder.resolve()),command=root['command'],seed=root['args']['seed'],cases=report,
        accuracy_certified=False,scope=PROTOCOL['scope'],
        raw_sha256={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*')) if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    require(not args.output.exists(),'Refusing to overwrite an audit.')
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['# KITE 四请求短输出容量窗口：真实端到端核验','','| 场景 | 基线 E2E s | KITE E2E s | E2E 加速比 | 基线主执行 s | KITE 主执行 s |',
        '|---|---:|---:|---:|---:|---:|']
    for r in result['cases']:
        lines.append(f'| {r["case_name"]} | {r["baseline_ms"]/1000:.3f} | {r["kite_ms"]/1000:.3f} | {r["speedup"]:.3f}× | {r["main_baseline_ms"]/1000:.3f} | {r["main_kite_ms"]/1000:.3f} |')
    lines += ['','每组均计入自己的标定、初始化及完整四请求生成；模型加载单列。',
        '受控同步到达、每请求 16 输出 tokens，不代表长生成稳态，也不把本基线称为原始 WiSP。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',output=str(args.output)),ensure_ascii=False))


if __name__=='__main__':
    main()
