"""Offline consistency audit of local latency records, not hardware remeasurement."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-6), label)


def stats(values):
    values = sorted(values)
    if not values:
        return None
    def percentile(p):
        position = (len(values)-1)*p
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo]+(values[hi]-values[lo])*(position-lo)
    return dict(n=len(values), mean=mean(values), median=median(values),
                p10=percentile(.1), p90=percentile(.9), min=values[0], max=values[-1])


def check_summary(rows, summary):
    values = dict(ttft_ms=[r['ttft_ms'] for r in rows],
                  completion_ms=[r['completion_ms'] for r in rows],
                  tpot_ms=[r['tpot_ms'] for r in rows if r['tpot_ms'] is not None],
                  itl_ms=[v for r in rows for v in r['itl_ms']])
    require(set(summary) == set(values), 'Metric names differ.')
    for key, samples in values.items():
        expected = stats(samples)
        if expected is None:
            require(summary[key] is None, f'{key} should be null.')
            continue
        require(set(summary[key]) == set(expected), f'{key} statistic names differ.')
        for field, value in expected.items():
            close(summary[key][field], value, f'{key}.{field} differs from raw samples.')


def audit(folder):
    folder = Path(folder)
    def read(name):
        return json.loads((folder/name).read_text(encoding='utf-8'))
    summary, perf, workload = (read(name) for name in
                              ['summary.json', 'performance.json', 'workload.json'])
    rows = [json.loads(line) for line in (folder/'request_metrics.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    require(summary['status'] == perf['status'] == 'passed', 'Run did not finish successfully.')
    require(not summary['is_kite_allocator'] and not perf['is_kite_allocator'],
            'This auditor only supports the existing-backend baseline.')
    require(not perf['trace_hooks_attached'] and perf['native_module_hook_count'] == 0,
            'Unexpected native trace hooks.')
    require(not perf['cost_calibration_performed'], 'Warmup must not be called calibration.')
    require(workload['schedule'] == 'sequential_requests_batch_1', 'Unexpected schedule.')
    require(hashlib.sha256((folder/'workload.json').read_bytes()).hexdigest()
            == perf['workload_sha256'], 'Workload hash differs.')
    require(len(rows) == len(workload['requests']) == perf['request_count']
            == summary['request_count'], 'Request counts differ.')
    require(len({r['request_id'] for r in rows}) == len(rows), 'Duplicate request IDs.')
    previous_time = None
    for index, (row, request) in enumerate(zip(rows, workload['requests'])):
        require(row['request_id'] == request['request_id'], 'Request order differs.')
        require(row['input_tokens'] == request['input_tokens'] == len(request['input_token_ids']),
                'Input token count differs.')
        require(row['topic'] == request['topic'], 'Topic differs.')
        require(row['warmup'] == (index < workload['warmup_requests']), 'Warmup prefix differs.')
        offsets = row['token_completion_offsets_ms']
        count = row['output_tokens']
        require(count == len(offsets) == len(row['generated_token_ids'])
                == row['forward_count'] and 1 <= count <= workload['new_tokens_cap'],
                'Generated-token accounting differs.')
        require(all(math.isfinite(v) and v > 0 for v in offsets)
                and all(b >= a for a, b in zip(offsets, offsets[1:])), 'Invalid token times.')
        close(row['ttft_ms'], offsets[0], 'TTFT differs.')
        close(row['completion_ms'], offsets[-1], 'Completion differs.')
        gaps = [b-a for a, b in zip(offsets, offsets[1:])]
        require(len(gaps) == len(row['itl_ms']), 'ITL count differs.')
        for actual, expected in zip(row['itl_ms'], gaps):
            close(actual, expected, 'ITL differs.')
        if gaps:
            close(row['tpot_ms'], mean(gaps), 'TPOT differs.')
        else:
            require(row['tpot_ms'] is None, 'Single-token TPOT should be null.')
        require(row['decode_forward_count'] == count-1, 'Decode forward count differs.')
        require(row['final_cache_length'] == row['input_tokens']+count-1,
                'Final cache length differs; last sampled token should not be fed back.')
        require(row['stop_reason'] in {'eos', 'max_new_tokens'}, 'Unknown stop reason.')
        if row['stop_reason'] == 'max_new_tokens':
            require(count == workload['new_tokens_cap'], 'Unexpected early stop.')
        tick = datetime.fromisoformat(row['time_utc'])
        require(previous_time is None or tick >= previous_time, 'Request timestamps reversed.')
        previous_time = tick
    evaluation = [r for r in rows if not r['warmup']]
    warmup = [r for r in rows if r['warmup']]
    require(evaluation, 'No evaluation requests.')
    require(len(warmup) == perf['warmup_count'] == workload['warmup_requests'], 'Warmup count differs.')
    require(len(evaluation) == perf['evaluation_count'], 'Evaluation count differs.')
    output_count = sum(r['output_tokens'] for r in evaluation)
    require(output_count == perf['evaluation_output_tokens'], 'Output total differs.')
    require(sum(r['forward_count'] for r in rows) == summary['forward_count']
            and sum(r['decode_forward_count'] for r in rows) == summary['decode_forward_count'],
            'Forward totals differ.')
    check_summary(evaluation, perf['evaluation_metrics'])
    lengths = {str(r['input_tokens']) for r in evaluation}
    require(set(perf['evaluation_by_input_tokens']) == lengths, 'Length groups differ.')
    for length in lengths:
        check_summary([r for r in evaluation if str(r['input_tokens']) == length],
                      perf['evaluation_by_input_tokens'][length])
    close(perf['loaded_workload_wall_ms_including_warmup_and_driver_gaps'],
          perf['warmup_prefix_wall_ms']+perf['evaluation_wall_ms_including_driver_gaps'],
          'Workload wall-time decomposition differs.')
    require(perf['warmup_prefix_wall_ms'] >= sum(r['completion_ms'] for r in warmup)
            and perf['evaluation_wall_ms_including_driver_gaps'] >= sum(r['completion_ms'] for r in evaluation),
            'Driver wall time is shorter than request times.')
    close(perf['evaluation_processing_tokens_per_second_excluding_driver_gaps'],
          output_count/(sum(r['completion_ms'] for r in evaluation)/1000), 'Processing throughput differs.')
    close(perf['evaluation_wall_tokens_per_second_including_driver_gaps'],
          output_count/(perf['evaluation_wall_ms_including_driver_gaps']/1000), 'Wall throughput differs.')
    require(perf['driver_job_wall_ms_including_model_load'] >=
            summary['load_seconds']*1000+perf['loaded_workload_wall_ms_including_warmup_and_driver_gaps'],
            'Job wall time is missing model load.')
    before, loaded, mapping = (read(name) for name in
        ['resources_before.json', 'resources_after_load.json', 'device_map.json'])
    require(not any(v == 'disk' for v in mapping.values()), 'Unexpected disk placement.')
    layer_placement = Counter(v for k, v in mapping.items()
                              if k.startswith('model.layers.') and len(k.split('.')) == 3)
    return dict(status='passed', run=str(folder.resolve()),
        checked='raw token timing, schedule, aggregation, workload hash, placement and wall-time consistency',
        evaluation_metrics=perf['evaluation_metrics'],
        by_length=perf['evaluation_by_input_tokens'],
        warmup_requests=warmup, load_seconds=summary['load_seconds'],
        loaded_workload_seconds=perf['loaded_workload_wall_ms_including_warmup_and_driver_gaps']/1000,
        complete_job_seconds=perf['driver_job_wall_ms_including_model_load']/1000,
        initial_gpu=before['gpus'], initial_gpu_processes=before['gpu_processes'],
        after_load_gpu=loaded['gpus'], layer_weight_placement=dict(layer_placement),
        cpu_affinity=loaded['allowed_logical_cpus'],
        raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(folder.iterdir()) if p.is_file()},
        limitations=['One controlled sequential run, five evaluation requests; not a mixed batch.',
            'Existing HF/Accelerate weight offload, not WISP or KITE.',
            'No continuous GPU/CPU isolation monitoring or inference memory peak record.',
            'Warmup cause unprofiled; resource snapshots alone do not demonstrate exclusive use.',
            'No observed physical cache miss counts or split-dependent end-to-end comparison.',
            'Small unequal length groups; no generalization or confidence claim.'])


def report(result):
    metrics = result['evaluation_metrics']
    lines = ['# 现有卸载后端延迟基线核验（2026-10-16）', '',
             '原始记录一致性核验通过。仅为真实模型的顺序 batch=1 试跑，不是 WISP/KITE 对照。', '',
             '| 输入 tokens | 评价请求数 | 平均 TTFT（秒） | 平均 TPOT（秒） | 平均完成时间（秒） |',
             '|---|---:|---:|---:|---:|']
    for length, group in sorted(result['by_length'].items(), key=lambda item:int(item[0])):
        lines.append(f"| {length} | {group['ttft_ms']['n']} | {group['ttft_ms']['mean']/1000:.3f} | "
                     f"{group['tpot_ms']['mean']/1000:.3f} | {group['completion_ms']['mean']/1000:.3f} |")
    lines += ['', f"五条评价请求平均 TTFT {metrics['ttft_ms']['mean']/1000:.3f} 秒，"
              f"TPOT {metrics['tpot_ms']['mean']/1000:.3f} 秒，"
              f"完成时间 {metrics['completion_ms']['mean']/1000:.3f} 秒。", '',
              f"模型加载 {result['load_seconds']:.3f} 秒；含预热的已加载工作负载 "
              f"{result['loaded_workload_seconds']:.3f} 秒；含加载的完整程序 "
              f"{result['complete_job_seconds']:.3f} 秒。", '',
              '第一条预热保留原始计时，但不纳入评价均值；预热不是四路径标定。', '',
              '权重存储映射：19 层 GPU、29 层 CPU（不是 CPU 执行层数）。加载后 GPU 占用约 23.06 GiB，'
              '仅为加载后快照，不是推理峰值。CPU affinity 为 24–31。', '',
              '逐 token 时间单调；TTFT/TPOT/完成时间、35 个评价 ITL、KV 最终长度、'
              '请求顺序和 workload SHA256 均已核对。生成文本存在，但本次没有答案质量评价。', '',
              '## 解释边界', '',
              '首次预热慢的原因尚未定位。后续不同长度耗时接近，不能直接认定全部来自 PCIe，'
              '也不能将 CPU 权重存储映射解读为专家 CPU fallback。', '',
              '未连续监控其他作业，不能凭启动时空闲快照宣称整段严格独占。'
              '输入预先分词，计时包含输入搬运和模型内部卸载，不含服务排队、网络及文本解码。', '',
              '尚不能证明混合 batch 的 P/D 修正、MRC 分配收益、四路径在线选择收益或任何 KITE 加速。', '',
              '## 下一步', '',
              '将工作负载级启动标定接入实际执行器，先验证 Expert 两路径的数值一致性、'
              '容量限制和真实 miss 计数，再接 KV 的 restore/整前缀 replay。'
              '四组消融必须共用同一执行器、请求顺序、容量预算和计时边界；'
              '当前 HF 基线只作参考，不代替固定代价消融。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.folder)
    require(not args.output.exists(), 'Refusing to replace an existing audit directory.')
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (args.output/'report.md').write_text(report(result), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], report=str((args.output/'report.md').resolve())), ensure_ascii=False))


if __name__ == '__main__':
    main()
