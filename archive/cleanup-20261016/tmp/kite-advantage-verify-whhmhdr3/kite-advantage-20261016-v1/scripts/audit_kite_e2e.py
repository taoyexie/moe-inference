"""Audit actual E2E clocks, requests, calibration, routing and applied budgets.

This reads records only. It never manufactures E2E times from per-miss costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from kite_controller import CostCalibration,candidates
from kite_e2e import PROTOCOL,choose_selection,implementation_hash,protocol_hash
from kite_e2e_core import FRAME_BYTES,EXPERT_BYTES,KV_BYTES,RuntimeCosts,packed_layout,ready_batch
from kite_e2e import new_requests
from workload_baseline import latency_summary


def require(value,message):
    if not value:
        raise ValueError(message)


def same(actual,expected):
    if isinstance(expected,dict):
        require(set(actual)==set(expected),'Statistic keys differ.')
        for key,value in expected.items():
            same(actual[key],value)
    elif expected is None:
        require(actual is None,'Expected null statistic.')
    elif isinstance(expected,list):
        require(len(actual)==len(expected),'Array coverage differs.')
        for a,b in zip(actual,expected):
            same(a,b)
    elif isinstance(expected,(float,int)):
        require(math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-6),
                'Timing/aggregate differs from raw records.')
    else:
        require(actual==expected,'Recorded value differs.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def expected_batches(specs):
    requests = new_requests(specs)
    rows = []
    step = 0
    while not all(r.done for r in requests):
        active = ready_batch(requests,step)
        if not active:
            step += 1
            continue
        layout = packed_layout(active)
        p = sum(r['phase']=='prefill' for r in layout)
        rows.append(dict(step=len(rows),prefill=p,decode=len(active)-p,
                         query_tokens=sum(len(r.query) for r in active)))
        for req,row in zip(active,layout):
            req.kv_length = row['after']
            req.generated.append(0)
        step += 1
        require(step<=128,'Schedule exceeded the executor round bound.')
    return rows


def audit_task(folder,specs,mode,budget,runtime_options=None):
    options = runtime_options or {}
    schedule = expected_batches(specs)
    perf = read(folder/'performance.json')
    require(perf['status']=='passed' and perf['is_full_model'] and perf['is_end_to_end'],
            'Task did not complete full-model inference.')
    require(perf['mode']==mode and perf['budget_mib']==budget
            and perf['actual_kite_enabled']==(mode=='kite'),'Wrong mode/budget.')
    clock = perf['task_clock_ns']
    require(clock['start']<=clock['initialized']<=clock['calibrated']<clock['finished'],'Invalid raw task clocks.')
    for key,a,b in [('workload_total_ms','start','finished'),('initialization_ms','start','initialized'),
                    ('startup_inclusive_ms','initialized','calibrated'),('execution_inclusive_ms','calibrated','finished')]:
        same(perf[key],(clock[b]-clock[a])/1e6)
    require(perf['calibration_ms']<=perf['startup_inclusive_ms'],'Calibration missing from E2E time.')
    requests = read(folder/'requests.json')
    require(len(requests)==len(specs) and len({r['request_id'] for r in requests})==len(specs),'Request coverage differs.')
    for row,spec in zip(requests,specs):
        require(row['request_id']==spec['request_id'] and row['input_tokens']==len(spec['input_ids'])
                and row['release_step']==spec['release_step'],'Request metadata differs.')
        require(row['output_tokens']==len(row['generated_ids'])==spec['output_tokens'],'Output length differs.')
        require(row['final_kv_length']==row['input_tokens']+row['output_tokens']-1,'Final KV length differs.')
        times = row['token_task_offsets_ms']
        require(len(times)==row['output_tokens'] and all(a<b for a,b in zip(times,times[1:])),
                'Invalid raw token time points.')
        require(times[0]>=perf['initialization_ms']+perf['startup_inclusive_ms']
                and times[-1]<=perf['workload_total_ms'],'Token clock outside actual execution.')
        offsets = [t-row['submitted_task_offset_ms'] for t in times]
        same(row['token_completion_offsets_ms'],offsets)
        same(row['ttft_ms'],offsets[0])
        same(row['completion_ms'],offsets[-1])
        gaps = [b-a for a,b in zip(times,times[1:])]
        same(row['itl_ms'],gaps)
        same(row['tpot_ms'],mean(gaps))
        if row['release_step']==0:
            same(row['submitted_task_offset_ms'],0)
    same(perf['request_metrics'],latency_summary(requests))
    require(perf['output_tokens']==sum(s['output_tokens'] for s in specs)
            and perf['request_count']==len(specs) and perf['batch_count']==len(schedule),'Wrong workload totals.')
    batches = [json.loads(line) for line in (folder/'batches.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(batches)==len(schedule),'Missing mixed batches.')
    total_q,total_calls = 0,0
    split_counts = set()
    for step,row in enumerate(batches):
        require(row['step']==step,'Batch order differs.')
        expected_p,expected_d = schedule[step]['prefill'],schedule[step]['decode']
        require((row['prefill'],row['decode'])==(expected_p,expected_d),'Actual P/D schedule differs.')
        require(row['query_tokens']==schedule[step]['query_tokens'],'Actual query length differs.')
        decision,pool = row['decision'],row['pool']
        require(decision['physical_split_applied'],'A decision was not physically applied.')
        same(decision['prefill_request_fraction'],expected_p/(expected_p+expected_d))
        candidate = decision['candidate']
        usable = budget*1024**2//FRAME_BYTES*FRAME_BYTES
        signal = usable*(.8*expected_p/(expected_p+expected_d)+.2*expected_d/(expected_p+expected_d))
        target = decision['history_expert_bytes'] if mode=='baseline' else .35*decision['history_expert_bytes']+.65*signal
        same(decision['current_signal_expert_bytes'],signal)
        same(decision['blended_expert_bytes'],target)
        choices = candidates(usable,EXPERT_BYTES,KV_BYTES,(.1,.25,.5,.75,.9))
        require(decision['history_expert_bytes'] in {c.expert_bytes for c in choices},'Invalid history proposal.')
        require(candidate==asdict(min(choices,key=lambda c:abs(c.expert_bytes-target))),
                'Actual split did not follow the declared history/current blend.')
        require(pool['limits']==dict(expert=candidate['expert_objects'],kv=candidate['kv_objects']),
                'Actual quotas differ from decision.')
        split_counts.add(candidate['expert_objects'])
        require(candidate['expert_bytes']==candidate['expert_objects']*EXPERT_BYTES
                and candidate['kv_bytes']==candidate['kv_objects']*KV_BYTES,'Incorrect object bytes.')
        require(pool['usable_bytes']==budget*1024**2//FRAME_BYTES*FRAME_BYTES,'Incorrect actual pool size.')
        require(sum(pool['resident_bytes'].values())+pool['free_bytes']==pool['usable_bytes'],
                'Shared physical frame accounting differs.')
        for resource,size in [('expert',EXPERT_BYTES),('kv',KV_BYTES)]:
            require(pool['resident_bytes'][resource]==pool['objects'][resource]*size
                    and pool['objects'][resource]<=pool['limits'][resource],'Resident memory exceeds quota.')
        events = row['expert_accesses']
        require(all(0<=key[0]<48 and 0<=key[1]<128 and type(q) is int and q>0 for key,q in events),
                'Invalid actual router event.')
        require(sum(q for _,q in events)==48*8*row['query_tokens']==row['routed_tokens'],
                'Router conservation failed.')
        total_q += row['query_tokens']
        total_calls += len(events)
    require(total_q==sum(len(s['input_ids'])+s['output_tokens']-1 for s in specs),
            'Input/decode query-token total differs.')
    expert,kv = perf['expert_counts'],perf['kv_counts']
    require(expert['expert_calls']==total_calls
            and expert.get('expert_hits',0)+expert.get('expert_misses',0)==total_calls
            and expert.get('expert_cpu',0)+expert.get('expert_upload',0)==expert.get('expert_misses',0),
            'Actual expert counters do not reconcile.')
    require(kv.get('kv_hits',0)+kv.get('kv_misses',0)==kv.get('kv_read_pages',0),'KV read counts differ.')
    recovery = read(folder/'recoveries.json')
    replays = [r for r in recovery if r['kind']=='replay']
    restores = [r for r in recovery if r['kind']=='restore']
    require(len(restores)==kv.get('kv_restore_pages',0) and len(replays)==kv.get('kv_replay_tasks',0),
            'Recovery path counters differ.')
    require(len({(r['request_id'],r['context']) for r in replays})==len(replays),'Duplicate replay macro.')
    require(len(restores)+sum(r['missing_pages'] for r in replays)==kv.get('kv_misses',0),
            'Missing pages not accounted for by actual recovery operations.')
    calibration = read(folder/'calibration.json')
    cal = CostCalibration()
    repeats = options.get('expert_repeats',3)
    require(len(calibration['rows'])==6*(1+repeats)+12,'Startup calibration sample coverage differs.')
    for sample in calibration['rows']:
        if not sample['warmup']:
            cal.add(sample['path'],sample['x'],sample['wall_ms'])
    cold_ms = 0.
    if options.get('measure_cpu_cold_setup',False):
        cold = calibration['cpu_cold_setup_rows']
        require(len(cold)==1+options.get('cpu_cold_setup_repeats',5),'Missing cold setup samples.')
        require(sum(r['warmup'] for r in cold)==1 and len({tuple(r['key']) for r in cold})==len(cold),
                'Cold setup samples must use distinct genuinely cold experts.')
        require(all(math.isfinite(r['wall_ms']) and r['wall_ms']>0 for r in cold),'Invalid cold setup time.')
        cold_ms = max(r['wall_ms'] for r in cold if not r['warmup'])
        same(calibration['cpu_cold_setup_ms'],cold_ms)
    costs = RuntimeCosts(cal,cal.freeze(),mode,robust=options.get('robust_costs',False),cpu_cold_setup_ms=cold_ms)
    same(calibration['affine'],asdict(costs.fitted))
    same(calibration['fixed'],costs.fixed)
    require(len(calibration['checks'])==3*(1+repeats)+12,'Missing startup numerical checks.')
    for row in calibration['checks']:
        error = row['error']
        limits = (.02,.03) if row['kind']=='expert' else (.03,.05)
        require(error['relative_l2']<=limits[0] and error['normalized_max_abs']<=limits[1],
                'Startup numerical limit exceeded.')
        if row['kind']=='kv_restore':
            same(error['max_abs'],0)
    require(perf['cpu_fp32_peak_bytes']<=8*1024**3
            and perf['host_kv_peak_bytes']<=options.get('host_kv_limit_bytes',1024**3),'Host cache budget exceeded.')
    require(perf['host_kv_peak_bytes']>0 and perf['packed_workspace_peak_bytes']>0,'No actual KV execution recorded.')
    if mode=='kite':
        require(len(split_counts)>1,'Full KITE never changed actual memory allocation.')
    return perf


def audit(folder):
    folder = Path(folder)
    root = read(folder/'summary.json')
    require(root['status']=='passed' and root['command'] in ('search','confirm'),'Expected completed actual E2E run.')
    require(root['protocol']==PROTOCOL and root['protocol_sha256']==protocol_hash(),'Protocol differs.')
    require(root['implementation_sha256']==implementation_hash(),'Sources differ from executed version.')
    validation = read(folder/'validation.json')
    require(validation['status']=='passed' and len(validation['checks'])==3,'Integration gate was not passed.')
    monitoring = read(folder/'resource_monitor.json')
    require(monitoring['error'] is None,'GPU/host interference was reported.')
    allowed = set(monitoring['allowed_gpu_pids'])
    require(len(allowed)==1 and all(set(r['gpu_pids'])<=allowed for r in monitoring['samples']),
            'Another GPU process was observed.')
    workload = read(folder/'workload.json')
    require(hashlib.sha256((folder/'workload.json').read_bytes()).hexdigest()==root['workload_sha256'],
            'Workload hash changed.')
    budgets = PROTOCOL['budgets_mib'] if root['command']=='search' else [root['frozen_selection']['selected']['budget_mib']]
    require(Counter((r['budget_mib'],r['mode']) for r in root['tasks'])==Counter(
        (b,m) for b in budgets for m in ('baseline','kite')),'Missing/duplicated primary comparisons.')
    rows = []
    for budget in budgets:
        pair = {mode:audit_task(folder/f'{budget}-{mode}',workload['requests'],mode,budget)
                for mode in ('baseline','kite')}
        for mode,perf in pair.items():
            require(perf==next(r for r in root['tasks'] if r['budget_mib']==budget and r['mode']==mode),
                    'Root task summary differs from child records.')
        baseline,kite = pair['baseline']['workload_total_ms'],pair['kite']['workload_total_ms']
        rows.append(dict(budget_mib=budget,baseline_ms=baseline,kite_ms=kite,speedup=baseline/kite,
                         reduction=1-kite/baseline,
                         output_ids_identical=[r['generated_ids'] for r in read(folder/f'{budget}-baseline'/'requests.json')]
                            ==[r['generated_ids'] for r in read(folder/f'{budget}-kite'/'requests.json')],
                         baseline_startup_ms=pair['baseline']['startup_inclusive_ms'],
                         kite_startup_ms=pair['kite']['startup_inclusive_ms']))
    if root['command']=='search':
        calculated = choose_selection(root['tasks'])
        selection = read(folder/'selection.json')
        for key,value in calculated.items():
            same(selection[key],value)
    return dict(status='passed',run=str(folder.resolve()),command=root['command'],seed=root['args']['seed'],
                comparisons=rows,scope='measured complete-model controlled workload including startup; not original WiSP or natural service traffic',
                raw_sha256={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(folder.rglob('*')) if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    result = [audit(path) for path in args.runs]
    require(not args.output.exists(),'Refusing to overwrite an audit.')
    args.output.mkdir(parents=True)
    write = lambda name,value:(args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write('audit.json',result)
    lines = ['# 完整 KITE 受控端到端实测核验','',
             '主指标包含显存池初始化、启动标定、在线决策、实际模型计算及 KV 搬运/写回；模型加载单列。','',
             '| 阶段 | Seed | 预算 MiB | 基线 s | KITE s | 加速比 | 耗时降低 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for run in result:
        for row in run['comparisons']:
            lines.append(f"| {run['command']} | {run['seed']} | {row['budget_mib']} | "
                         f"{row['baseline_ms']/1000:.3f} | {row['kite_ms']/1000:.3f} | "
                         f"{row['speedup']:.3f}× | {row['reduction']*100:.2f}% |")
    lines += ['', '确认结果与探索结果分列，不以负结果为由删除已执行的配置。'
              '本表不是自然线上服务评测，不证明阶段修正具有独立收益，也不是原始 WiSP 对照。', '']
    (args.output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status='passed',report=str(args.output/'report.md')),ensure_ascii=False))


if __name__=='__main__':
    main()
