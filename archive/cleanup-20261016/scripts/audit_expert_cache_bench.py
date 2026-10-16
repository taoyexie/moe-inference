"""Audit controlled paired cache timings offline, without Torch or GPU execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, OrderedDict
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from expert_cache import fit_costs
from expert_cache_bench import case_plan, paired_summary
from expert_cache_smoke import OBJECT_BYTES


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a,b):
    require(math.isfinite(a) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-6),
            'Raw timing/aggregate mismatch.')


def replay(plan, policy, costs):
    resident, counts, paths, events = OrderedDict(),Counter(),Counter(),[]
    for row in plan:
        key,q = tuple(row['key']),row['q']
        hit = key in resident
        path = 'hit' if hit else costs.choose(q,fixed=policy=='fixed')[0]
        evicted, admitted = None,False
        if hit:
            resident.move_to_end(key)
        elif path=='upload':
            if len(resident)==2:
                evicted = list(resident.popitem(last=False)[0])
            resident[key] = None
            admitted = True
        counts['calls'] += 1
        counts['hits' if hit else 'misses'] += 1
        if not hit:
            counts[path] += 1
        paths[path] += 1
        events.append(dict(key=list(key),q=q,hit=hit,path=path,evicted=evicted,admitted=admitted,
                           resident_lru=[list(k) for k in resident],capacity=2,
                           resident_weight_bytes=len(resident)*OBJECT_BYTES,
                           allocated_weight_bytes=2*OBJECT_BYTES))
    return dict(counts),dict(paths),events


def audit(folder):
    folder = Path(folder)
    def read(name):
        return json.loads((folder/name).read_text(encoding='utf-8'))
    def rows(name):
        return [json.loads(line) for line in (folder/name).read_text(encoding='utf-8').splitlines()]
    summary,protocol,calibration = [read(name) for name in ('summary.json','protocol.json','calibration.json')]
    passes,samples,correctness = rows('passes.jsonl'),rows('calibration.jsonl'),read('correctness.json')
    require(summary['status']=='passed' and all(summary[key] is False for key in
            ('is_kite_allocator','is_end_to_end','is_WiSP','tests_phase_correction')), 'Incorrect scope/status.')
    require(summary['cache_weight_bytes']==2*OBJECT_BYTES
            and summary['extra_scratch_weight_bytes']==OBJECT_BYTES, 'Incorrect memory budget.')
    args = summary['args']
    require(all(args[k]==protocol[k] for k in ('seed','repeats','warmup_pairs')), 'Arguments/protocol differ.')
    require(Counter((r['q'],r['path'],r['warmup']) for r in samples)==Counter({
        (q,path,warm):3 if warm else 5 for q in (1,8,32) for path in ('upload','cpu') for warm in (True,False)}),
        'Calibration coverage differs.')
    require(calibration['samples']==30 and len(samples)==48, 'Wrong calibration count.')
    costs = fit_costs(r for r in samples if not r['warmup'])
    for key,value in asdict(costs).items():
        close(calibration['costs'][key],value)
    close(summary['calibration_inclusive_wall_ms'],calibration['total_wall_ms_including_warmup_validation_and_fit'])
    keys = sorted({tuple(row['key']) for plan in protocol['cases'].values() for row in plan})
    expected_plans = {name:[dict(key=list(key),q=q) for key,q in plan]
                      for name,plan in case_plan(keys,args['cycles']).items()}
    require(protocol['cases']==expected_plans, 'Case protocol differs from declared construction.')
    n = 3*args['cycles']
    require(protocol['prepared_input_gpu_bytes']==sum(q*2048*2 for key,q in
            { (tuple(row['key']),row['q']) for plan in protocol['cases'].values() for row in plan}),
            'Prepared GPU input bytes differ.')
    rng,expected_order = random.Random(args['seed']),[]
    for trial in range(-args['warmup_pairs'],args['repeats']):
        names = list(protocol['cases'])
        rng.shuffle(names)
        for name in names:
            order = ['fixed','calibrated']
            rng.shuffle(order)
            expected_order += [(trial,name,policy,position) for position,policy in enumerate(order)]
    require([(r['trial'],r['case'],r['policy'],r['order']) for r in passes]==expected_order,
            'Paired execution order/coverage differs.')
    expected = {(name,policy):replay(plan,policy,costs) for name,plan in protocol['cases'].items()
                for policy in ('fixed','calibrated')}
    detailed = 0
    for row in passes:
        require(row['warmup']==(row['trial']<0), 'Warmup incorrectly labelled.')
        counts,paths,events = expected[(row['case'],row['policy'])]
        require(row['counters']==counts and row['paths']==paths, 'Physical path counters differ from replay.')
        require(all(math.isfinite(row[k]) and row[k]>0 for k in
                ('execution_wall_ms','setup_execution_teardown_wall_ms','invocation_wall_ms_sum')),
                'Invalid timing.')
        require(row['execution_wall_ms']>=row['invocation_wall_ms_sum']
                and row['setup_execution_teardown_wall_ms']>=row['execution_wall_ms'], 'Timing scopes inconsistent.')
        if row['trial']==0:
            require(len(row['events'])==n, 'First trial lacks complete detailed events.')
            for observed,event in zip(row['events'],events):
                require(all(observed[k]==v for k,v in event.items()), 'Detailed cache state differs from replay.')
                require(math.isfinite(observed['wall_ms']) and observed['wall_ms']>0, 'Invalid detailed timing.')
            close(sum(event['wall_ms'] for event in row['events']),row['invocation_wall_ms_sum'])
            detailed += n
        else:
            require(not row['events'], 'Unexpected detailed trace outside selected trial.')
    recomputed = paired_summary(passes)
    require(set(recomputed)==set(summary['cases']), 'Missing case summary.')
    for name,values in recomputed.items():
        for key,value in values.items():
            if isinstance(value,(float,int)):
                close(summary['cases'][name][key],value)
            else:
                require(summary['cases'][name][key]==value, 'Summary note differs.')
    inputs = {(tuple(row['key']),row['q']) for plan in protocol['cases'].values() for row in plan}
    require(Counter((tuple(r['key']),r['q'],r['phase']) for r in correctness)==Counter({
        (key,q,phase):1 for key,q in inputs for phase in ('pre','post')}), 'Numerical coverage differs.')
    errors = [r['error'] for r in correctness]+[r['output_error'] for r in samples]
    require(all(all(math.isfinite(value) and value>=0 for value in error.values())
                and error['relative_l2']<=.02 and error['normalized_max_abs']<=.03 for error in errors),
            'Numerical acceptance limit exceeded.')
    out = {}
    for name,stats in recomputed.items():
        fixed,affine = stats['fixed_mean_execution_ms'],stats['affine_mean_execution_ms']
        inclusive = {policy:mean(r['setup_execution_teardown_wall_ms'] for r in passes
                     if r['case']==name and r['policy']==policy and not r['warmup'])
                     for policy in ('fixed','calibrated')}
        out[name] = dict(**stats,ratio_of_means_savings=1-affine/fixed,speedup_ratio_of_means=fixed/affine,
            fixed_counts=expected[(name,'fixed')][0],affine_counts=expected[(name,'calibrated')][0],
            setup_execution_teardown_means_ms=inclusive,
            conservative_calibration_payback_passes=math.ceil(summary['calibration_inclusive_wall_ms']/(fixed-affine))
                if fixed>affine else None)
    return dict(status='passed',run=str(folder.resolve()),seed=args['seed'],calls_per_pass=n,cases=out,
        checked_passes=len(passes),formal_calls=sum(r['counters']['calls'] for r in passes if not r['warmup']),
        detailed_events_checked=detailed,correctness_comparisons=len(errors),
        worst_relative_l2=max(e['relative_l2'] for e in errors),
        worst_normalized_max_abs=max(e['normalized_max_abs'] for e in errors),
        calibration_ms=summary['calibration_inclusive_wall_ms'],job_ms=summary['complete_job_wall_ms'],
        cost_model=asdict(costs),raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(folder.iterdir()) if p.is_file()},
        scope='Selected controlled real-kernel expert-cache experiment; not full KITE, WiSP, P/D or KV evidence.')


def report(result):
    names = {'mixed_q_pressure':'异构 q + 缓存压力','small_q_pressure':'小 q + 缓存压力',
             'stable_hot_control':'稳定热点对照'}
    main = result['cases']['mixed_q_pressure']
    lines = [f"# 专家成本选择配对实测核验（2026-10-16，seed {result['seed']}）",'',
             '同一 A100、同一真实 Qwen 专家内核、相同两个 GPU 权重槽。10 组随机交错正式配对，'
             '每个 pass 192 次调用，缓存从空开始，计时中无逐调用参考检查。', '',
             '| 场景 | 固定成本平均 ms | 线性成本平均 ms | 平均配对耗时降低 | 正收益配对 |',
             '|---|---:|---:|---:|---:|']
    for name,row in result['cases'].items():
        lines.append(f"| {names[name]} | {row['fixed_mean_execution_ms']:.3f} | "
                     f"{row['affine_mean_execution_ms']:.3f} | {row['paired_savings_mean']*100:.2f}% | "
                     f"{row['affine_faster_pairs']}/{row['pairs']} |")
    lines += ['', '## 为什么主场景有利', '',
              '每轮 A(q=32)、B(q=1)、C(q=1)，三专家竞争两个槽。固定成本模型在这次标定下选择 upload，'
              '导致循环淘汰，192 次调用均为 upload miss。线性模型仅首次上传 A，随后 A 命中 63 次，'
              'B/C 共 128 次走 CPU，不污染 GPU 缓存。它利用的是计算量异构与 CPU bypass 的共同收益，'
              '不能单独归因于拟合精度或 phase 修正。','',
              '小 q 场景线性模型所有调用都走 CPU，固定模型均 upload；稳定热点场景两种模型均首次上传两次、'
              '随后命中 190 次。稳定热点变化很小且配对有正有负，不作为加速结论。','',
              '## 开销和核验', '',
              f"启动 Expert-only 标定（含标定预热/数值检查/拟合）{result['calibration_ms']:.3f} ms；"
              f"完整比较程序 {result['job_ms']/1000:.3f} 秒。主表不含槽初始化/释放和启动标定，"
              '不表示工作负载含启动端到端时间。', '',
              '槽初始化、执行和释放均另有计时，详见 audit.json；若仅在线性方案单独计入本次全部标定开销，'
              f"主场景需约 {main['conservative_calibration_payback_passes']} 个同长度 pass 才能按本轮平均执行差额摊平，"
              '属于算术估算、不是新的实测端到端收益。','',
              f"核对 {result['checked_passes']} 个 pass，{result['formal_calls']} 次正式调用的汇总计数，"
              f"{result['detailed_events_checked']} 次详细状态转换与时间，{result['correctness_comparisons']} 次数值检查。"
              f"最坏 relative L2={result['worst_relative_l2']:.6f}，"
              f"normalized max={result['worst_normalized_max_abs']:.6f}。", '',
              'GPU cache 18 MiB，独立 scratch 9 MiB；CPU FP32 权重副本、I/O 和预置输入另计。'
              '数值比较仅为前后已选输入和标定检查，未对每次正式调用都重算输出。', '',
              '## 可用表述', '',
              '在构造的路由 token 数异构及专家缓存压力场景下，基于 routed-token 数的路径成本模型，'
              f"相较使用相同标定样本的固定成本模型，将每 {result['calls_per_pass']} 次专家调用的平均执行时间从 "
              f"{main['fixed_mean_execution_ms']:.3f} ms 降低至 {main['affine_mean_execution_ms']:.3f} ms；"
              f"{main['pairs']} 组配对测量的平均耗时降幅为 {main['paired_savings_mean']*100:.2f}%。该结果来自真实专家权重与算子执行，"
              '说明按计算规模选择 CPU 回传或权重上传路径可减少不必要的搬运和缓存污染。','',
              '上述可用表述须附硬件、CPU FP32/GPU BF16、受控输入构造、缓存容量、计时边界与标定开销。'
              '该场景依据此前结果主动选取，不是自然请求 trace 或盲测；单进程重复尚需独立会话复测。'
              '不能将该降幅写成完整模型 TTFT、KITE、phase-aware 或原始 WiSP 的提升。','']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    result = audit(args.folder)
    require(not args.output.exists(), 'Refusing to overwrite an existing audit.')
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.output/'report.md').write_text(report(result),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],cases=result['cases'],report=str(args.output/'report.md')),
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
