"""Audit downloaded expert timing logs without torch/GPU use or source changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from expert_paths import PATHS, input_source, parse_int_list, summarize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def same_stats(actual, expected):
    require(set(actual) == set(expected), 'Timing statistic keys differ.')
    for key, value in expected.items():
        require(math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-12),
                f'Recomputed {key} differs: {actual[key]} vs {value}')


def audit(folder):
    folder = Path(folder).resolve()
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    require(summary['status'] == 'passed', 'Run failed or incomplete.')
    require(summary['is_end_to_end_inference'] is False, 'Unexpected scope.')
    rows = load_jsonl(folder/'trials.jsonl')
    points = load_jsonl(folder/'points.jsonl')
    sorted_points = json.loads((folder/'point_summary.json').read_text(encoding='utf-8'))
    require(sorted(points, key=lambda r: (r['layer'], r['q'])) == sorted_points,
            'Two point summaries differ.')
    samples = {(s['layer'],s['expert']):s for s in summary['samples']}
    require(len(samples) == len(summary['samples']), 'Duplicate expert samples.')
    counts = summary['args']['repeats']
    requested_q = parse_int_list(summary['args']['q'], 1, 128)
    expected_points = {(layer, expert, q) for (layer, expert), s in samples.items()
                       for q in set(requested_q+[s['native_routed_tokens']])}
    grouped = defaultdict(dict)
    max_l2 = max_normalized = checked = 0
    for s in samples.values():
        require(s['shape'] == [s['native_routed_tokens'],2048], 'Activation shape mismatch.')
        require(s['weight_bytes_bf16'] == 9*1024**2, 'Unexpected expert weight size.')
        require(len(s['weight_tensors']) == 3, 'Missing source weight metadata.')
        require(all(re.fullmatch('[0-9a-f]{64}', t['tensor_bf16_sha256']) for t in s['weight_tensors']),
                'Malformed tensor identity.')
    for row in rows:
        identity = (row['layer'],row['expert'],row['q'])
        require(identity in expected_points, 'Unexpected expert/q point.')
        native = samples[identity[:2]]['native_routed_tokens']
        require(row['native_q'] == native and row['input_source'] == input_source(row['q'],native),
                'Input provenance mismatch.')
        require(0 <= row['trial'] < counts and row['path'] in PATHS, 'Invalid trial or path.')
        require(0 <= row['order'] < 4, 'Invalid execution order.')
        require(math.isfinite(row['wall_ms']) and row['wall_ms'] > 0, 'Invalid timing.')
        key = (*identity, row['trial'])
        require(row['path'] not in grouped[key], 'Duplicate trial/path.')
        grouped[key][row['path']] = row
        require(row['correctness_pass'] is True, 'Failed correctness record.')
        error = row['output_error']
        if row['path'] == 'weight_h2d_only':
            require(error is None, 'Transfer-only operation has a compute result.')
        else:
            require(error is not None, 'Missing correctness metrics.')
            require(all(math.isfinite(v) and v >= 0 for v in error.values()), 'Invalid output errors.')
            require(error['relative_l2'] <= summary['args']['relative_l2_tolerance']
                    and error['normalized_max_abs'] <= summary['args']['normalized_max_tolerance'],
                    'Output error above declared tolerance.')
            max_l2 = max(max_l2, error['relative_l2'])
            max_normalized = max(max_normalized, error['normalized_max_abs'])
            checked += 1
    require(len(rows) == len(expected_points)*counts*4, 'Missing timing rows.')
    require(len(grouped) == len(expected_points)*counts, 'Missing trial groups.')
    require(len(points) == summary['point_count'] == len(expected_points), 'Point count mismatch.')
    point_map = {(p['layer'],p['expert'],p['q']):p for p in points}
    require(set(point_map) == expected_points, 'Missing or duplicate point summaries.')
    point_results = []
    for identity in sorted(expected_points):
        per_path = defaultdict(list)
        for trial in range(counts):
            pair = grouped[(*identity,trial)]
            require(set(pair) == set(PATHS), 'Trial lacks a measurement path.')
            require({r['order'] for r in pair.values()} == set(range(4)), 'Invalid path permutation.')
            for path, row in pair.items():
                per_path[path].append(row['wall_ms'])
            for path in ('upload_gpu','cpu_return'):
                per_path[path+'_minus_hit'].append(pair[path]['wall_ms']-pair['gpu_hit']['wall_ms'])
        stats = {path:summarize(values) for path,values in per_path.items()}
        for path, expected in stats.items():
            same_stats(point_map[identity]['timings'][path], expected)
        point_results.append({'layer':identity[0], 'expert':identity[1], 'q':identity[2],
                              'timings':stats, 'cpu_trials_ms_in_order':per_path['cpu_return'],
                              'faster_median_full_path':min(('upload_gpu','cpu_return'),
                                                           key=lambda path:stats[path]['median_ms'])})
    hashes = {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in
              ('summary.json','trials.jsonl','points.jsonl','point_summary.json')}
    return {'audit_status':'passed', 'input_folder':str(folder), 'sha256':hashes,
            'trial_rows':len(rows), 'point_count':len(points), 'output_comparisons':checked,
            'max_relative_l2':max_l2, 'max_normalized_max_abs':max_normalized,
            'warmup':summary['args']['warmup'], 'repeats':counts,
            'cpu_threads':summary['args']['cpu_threads'], 'cpu_affinity':summary['effective_cpu_affinity'],
            'peak_cuda_allocated_bytes':summary['peak_cuda_allocated_bytes'],
            'points':point_results,
            'evidence_limits':['Timing/output metrics audited from logs; source tensors not available locally to rehash.',
                               'Hot isolated-expert microbenchmark, not end-to-end inference.',
                               'CPU FP32 and GPU BF16 compute; output correctness is tolerance-based.',
                               'Resized q values use controlled prefix/cyclic activation rows.',
                               'Snapshots do not establish resource exclusivity throughout execution.']}


def report(result):
    formal = result['warmup'] >= 10 and result['repeats'] >= 30
    title = '# Expert 两路径正式测量会话核验' if formal else '# Expert 两路径试跑核验'
    lines = [title, '',
             f"数据目录：`{result['input_folder']}`。离线核验通过。", '',
             f"共 {result['point_count']} 个专家×q 点、{result['trial_rows']} 条计时、{result['output_comparisons']} 次计算输出对照。",
             f"最大 relative L2：{result['max_relative_l2']:.6f}；最大归一化最大绝对误差：{result['max_normalized_max_abs']:.6f}。",
             'GPU hit 与上传后计算均对照同输入的 BF16 GPU 输出；CPU FP32 路径按容限验证，不声明 bitwise 等价。', '',
             f"设置：CPU {result['cpu_threads']} 线程，逻辑 CPU {result['cpu_affinity']}；每点每项预热 {result['warmup']} 次、测量 {result['repeats']} 次。",
             f"PyTorch 分配显存峰值：{result['peak_cuda_allocated_bytes']/2**20:.3f} MiB（不包含全部 CUDA 上下文开销）。", '',
             '## 逐专家结果', '',
             '| Layer | Expert | q | 纯权重 H2D ms | 常驻 GPU ms | 上传+GPU ms | CPU+回传 ms | 中位数更快路径 |',
             '|---:|---:|---:|---:|---:|---:|---:|---|']
    for p in result['points']:
        times = [p['timings'][path]['median_ms'] for path in PATHS]
        lines.append(f"| {p['layer']} | {p['expert']} | {p['q']} | " + ' | '.join(f'{v:.6f}' for v in times) + f" | {p['faster_median_full_path']} |")
    lines.extend(['', '## 执行次序中的时间变化', '',
                  '保留全部测量点，不删除较慢 trial。以下逐次数据用于检查启动/次序影响，不能单凭下降趋势诊断具体原因。', ''])
    for p in result['points']:
        if p['layer'] == 0 and p['q'] in (32,128):
            lines.append(f"- L{p['layer']} E{p['expert']} q={p['q']} 的 CPU 路径依次为：" + ', '.join(f'{v:.6f}' for v in p['cpu_trials_ms_in_order']) + ' ms。')
    lines.extend(['', '## 结论范围', '',
                  '此次数据支持：在指定热数据、CPU FP32、GPU BF16、8 线程条件下，小 q 的 CPU 路径比上传路径快，大 q 时优劣反转。',
                  '不支持：CPU 路径严格线性、所有模型/工作负载都存在同一交叉点、真实 cache miss 数或 KITE 端到端提升。',
                  '不能把完整上传路径当成纯 H2D；两条完整路径都从 GPU 输入到 GPU 输出。',
                  '所有 point summary 已从逐次记录重算核对；本地未重新计算输出、未独立读取服务器原始权重和激活。',
                  '原始记录的 SHA256 见 audit.json；快照只反映运行前后状态，不保证期间独占资源。', ''])
    if formal:
        lines.extend(['## 后续会话', '',
                      '此文件核验一个独立进程会话，不能称已经完成三个会话。',
                      '其余会话沿用相同 q、精度、输入、10 次预热、30 次测量和误差容限，只改变顺序随机种子。',
                      '保持安全资源门禁；相邻进程启动间增加短等待，不将等待纳入路径计时。',
                      '不要合并旧试跑与正式数据来估算误差带；不要先强行假定 CPU 路径严格线性。', ''])
        return '\n'.join(lines)
    lines.extend([
                  '## 下一轮预先固定设置', '',
                  '使用相同三个专家和激活；q=1/2/4/8/16/24/32/64/128，外加每个专家的原生 q。',
                  '24 是为细化先前观察到的 q=8 到 q=32 之间的路径反转，不参与回填本次试跑。',
                  '为检查前几个 trial 偏慢，将预热从试跑的 2 次增加为 10 次；每点每项正式记录 30 次。',
                  '三个独立进程分别使用顺序随机种子 17/23/41，保持 8 线程、相同亲和性与误差阈值。',
                  '不承诺预热能消除波动，不因 proposed 是否获胜而删改试验；仍保留逐次记录检查趋势。', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in [('audit.json',json.dumps(result,ensure_ascii=False,indent=2)+'\n'),
                       ('report.md',report(result))]:
        with (args.output_dir/name).open('x',encoding='utf-8') as stream:
            stream.write(text)
    print(json.dumps({key:value for key,value in result.items() if key not in ('points','sha256')},
                     ensure_ascii=False,indent=2))
    print('Reports:', args.output_dir.resolve())


if __name__ == '__main__':
    main()
