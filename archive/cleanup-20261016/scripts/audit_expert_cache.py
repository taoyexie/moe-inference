"""Offline audit of the first physical expert-cache acceptance run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, OrderedDict
from dataclasses import asdict
from pathlib import Path

from expert_cache import ExpertCosts, fit_costs
from expert_cache_smoke import OBJECT_BYTES, schedule


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    require(math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-6),
            'Numeric aggregate differs from original records.')


def audit(folder):
    folder = Path(folder)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    calibration = json.loads((folder/'calibration.json').read_text(encoding='utf-8'))
    def rows(name):
        return [json.loads(line) for line in (folder/name).read_text(encoding='utf-8').splitlines()]
    samples,events = rows('calibration.jsonl'),rows('events.jsonl')
    require(summary['status']=='passed' and not summary['is_kite_allocator'], 'Unexpected experiment scope/status.')
    require(len(samples)==48 and sum(not r['warmup'] for r in samples)==30, 'Calibration sample count differs.')
    require(calibration['samples']==30 and calibration['expert_only']
            and not calibration['four_path_calibration_complete'], 'Unexpected calibration scope.')
    require(Counter((r['q'],r['path'],r['warmup']) for r in samples) == Counter({
        (q,path,warm):3 if warm else 5 for q in (1,8,32) for path in ('upload','cpu') for warm in (True,False)}),
        'Calibration q/path coverage differs.')
    costs = fit_costs(r for r in samples if not r['warmup'])
    for key,value in asdict(costs).items():
        close(calibration['costs'][key],value)
    require(len(events)==60, 'Expected 60 acceptance calls.')
    keys = sorted(tuple(row['key']) for row in summary['weights'])
    plan = schedule(keys)
    output = []
    for scenario in summary['scenarios']:
        label = scenario['scenario']
        stream = [r for r in events if r['scenario']==label]
        require(len(stream)==12, 'Wrong scenario length.')
        resident,counts = OrderedDict(),Counter()
        capacity = summary['args']['capacity_experts']
        resize = {r['position']:r for r in scenario['resize']}
        elapsed = 0.
        for position,(row,(key,q)) in enumerate(zip(stream,plan)):
            if position in resize:
                capacity = resize[position]['capacity']
                removed = []
                while len(resident)>capacity:
                    removed.append(list(resident.popitem(last=False)[0]))
                require(removed==resize[position]['evicted'], 'Resize eviction differs from independent LRU.')
                require(resize[position]['allocated_weight_bytes']==capacity*OBJECT_BYTES, 'Resize bytes differ.')
            require(row['position']==position and tuple(row['key'])==key, 'Invocation order differs.')
            require(row['q']==(row['native_q'] if q is None else q), 'Controlled q differs.')
            require(row['capacity']==capacity, 'Cache capacity differs.')
            hit = key in resident
            path = 'hit' if hit else 'upload'
            if not hit:
                if label=='cpu_bypass':
                    path = 'cpu'
                elif label in ('fixed_cost','affine_cost'):
                    path = costs.choose(row['q'],fixed=label=='fixed_cost')[0]
            removed,admitted = None,False
            if hit:
                resident.move_to_end(key)
            elif path=='upload' and capacity:
                if len(resident)==capacity:
                    removed = list(resident.popitem(last=False)[0])
                resident[key] = None
                admitted = True
            require(row['hit']==hit and row['path']==path and row['admitted']==admitted
                    and row['evicted']==removed, 'Execution path/LRU transition differs.')
            require(row['resident_lru']==[list(key) for key in resident], 'Resident order differs.')
            require(row['resident_weight_bytes']==len(resident)*OBJECT_BYTES
                    and row['allocated_weight_bytes']==capacity*OBJECT_BYTES
                    and row['scratch_weight_bytes']==OBJECT_BYTES, 'Declared GPU bytes differ.')
            require(row['correctness_pass'], 'Reported numerical check failed.')
            counts['calls'] += 1
            counts['hits' if hit else 'misses'] += 1
            if not hit:
                counts[path] += 1
            require(math.isfinite(row['wall_ms']) and row['wall_ms']>0, 'Invalid elapsed time.')
            elapsed += row['wall_ms']
        require(dict(counts)==scenario['counters'], 'Scenario counts differ.')
        close(elapsed,scenario['invocation_wall_ms_sum'])
        require(scenario['scenario_wall_ms_including_setup_validation_and_teardown']>=elapsed,
                'Scenario wall time too short.')
        output.append(dict(scenario=label,**dict(counts),invocation_wall_ms=elapsed,
                           inclusive_wall_ms=scenario['scenario_wall_ms_including_setup_validation_and_teardown']))
    for row in samples+events:
        require(all(math.isfinite(value) and value>=0 for value in row['output_error'].values()),
                'Invalid numerical error.')
        require(row['output_error']['relative_l2']<=.02 and row['output_error']['normalized_max_abs']<=.03,
                'Numerical error exceeds acceptance limits.')
    return dict(status='passed',run=str(folder.resolve()),costs=asdict(costs),scenarios=output,
                checked_calls=60,checked_calibration_rows=48,
                worst_relative_l2=max(r['output_error']['relative_l2'] for r in samples+events),
                worst_normalized_max_abs=max(r['output_error']['normalized_max_abs'] for r in samples+events),
                complete_job_wall_ms=summary['complete_job_wall_ms'],
                calibration_inclusive_wall_ms=calibration['total_wall_ms_including_warmup_validation_and_fit'],
                cuda_peak_allocated_bytes=summary['cuda_peak_allocated_bytes'],
                raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(folder.iterdir()) if p.is_file()},
                limitation='Functional acceptance only: synthetic order, one run, reference uploads between calls; no KITE/WISP performance conclusion.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    result = audit(args.folder)
    require(not args.output.exists(), 'Refusing to overwrite an audit.')
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['# Expert 物理缓存验收核验（2026-10-16）','',
             '60 次调用、48 次启动计时已核对；独立 LRU 重放与实际记录一致，CPU bypass 不准入。','',
             '| 场景 | Hit | Miss | Upload | CPU | 调用计时合计 ms |',
             '|---|---:|---:|---:|---:|---:|']
    for row in result['scenarios']:
        lines.append(f"| {row['scenario']} | {row.get('hits',0)} | {row.get('misses',0)} | "
                     f"{row.get('upload',0)} | {row.get('cpu',0)} | {row['invocation_wall_ms']:.6f} |")
    lines += ['',f"最坏 relative L2={result['worst_relative_l2']:.6f}，"
              f"normalized max={result['worst_normalized_max_abs']:.6f}。",
              f"启动 Expert-only 标定含预热与参考检查 {result['calibration_inclusive_wall_ms']:.3f} ms；"
              f"完整验收 {result['complete_job_wall_ms']:.3f} ms。",'',
              'GPU cache 两槽为 18 MiB，单独 scratch 9 MiB；零容量和缩扩容的声明字节计数一致。',
              '字节会计来自代码与日志，不能替代硬件内存地址级观测。','',
              '线性选择 miss 更多但调用计时合计更少，说明 miss 数不能单独充当性能指标。'
              '这是单次、受控短序列，每次调用之间有参考上传；不可写成 KITE 加速或 WISP 对照。','',
              '下一步先预定义有利适用场景和稳定热点对照，进行随机交错重复测量；'
              '不在正式计时中插入逐调用参考上传，所有场景和原始数据保留。','']
    (args.output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':result['status'],'report':str(args.output/'report.md')},ensure_ascii=False))


if __name__=='__main__':
    main()
