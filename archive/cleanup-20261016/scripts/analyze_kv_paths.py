"""Offline audit of downloaded KV records; no model/GPU execution or input mutation."""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path

from expert_paths import parse_int_list, sha256_file, summarize
from kite_trace import GIB
from kv_paths import PATHS, object_spec, repeated_prefix, validate_loading
from qwen_smoke import EXPECTED


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def tensor_header(path, spec):
    """Validate BF16 tensor layout and payload length without torch/safetensors imports."""
    with Path(path).open('rb') as stream:
        size_bytes = stream.read(8)
        require(len(size_bytes) == 8, 'Truncated safetensors header.')
        header_bytes = struct.unpack('<Q', size_bytes)[0]
        require(2 <= header_bytes <= 1024**2, 'Unexpected safetensors header length.')
        header = json.loads(stream.read(header_bytes))
    require(set(header) - {'__metadata__'} == {'kv'}, 'Expected exactly one KV payload.')
    tensor = header['kv']
    require(tensor['dtype'] == 'BF16' and tensor['shape'] == spec['shape'], 'KV dtype/shape mismatch.')
    require(tensor['data_offsets'] == [0, spec['bytes']], 'KV payload offsets differ.')
    require(Path(path).stat().st_size == 8+header_bytes+spec['bytes'], 'KV payload is incomplete.')
    return tensor


def check_stats(actual, expected):
    require(set(actual) == set(expected), 'Timing statistic keys differ.')
    for key, value in expected.items():
        require(math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-12),
                f'Timing statistic {key} differs from raw records.')


def audit(folder):
    folder = Path(folder).resolve()
    summary = read_json(folder/'summary.json')
    args = summary['args']
    require(summary['status'] == 'passed', 'Failed or incomplete benchmark.')
    require(summary['is_allocator_benchmark'] is False
            and summary['is_end_to_end_serving_benchmark'] is False, 'Unexpected evidence scope.')
    require(summary['protocol'] == 'pinned_CPU_restore_vs_full_token_prefix_replay_to_same_tail_block',
            'Unexpected recovery prerequisites.')
    require(summary['model_class'] == 'Qwen3MoeModel', 'Unexpected transformer class.')
    require(re.fullmatch('[0-9a-f]{64}', summary['model_content_manifest_sha256']) is not None,
            'Malformed checkpoint identity.')
    config = read_json(folder/'model_config.json')
    require(all(config[key] == value for key,value in EXPECTED.items()), 'Qwen model configuration differs.')
    validate_loading(read_json(folder/'loading_info.json'))
    mapping = read_json(folder/'device_map.json')
    require(set(mapping.values()) <= {'0', 'cuda:0', 'cpu'}, 'Disk/other-device offload found.')
    spec = object_spec(args['block_tokens'])
    require(summary['object'] == spec, 'Recovery object specification differs.')
    contexts = parse_int_list(args['contexts'], args['block_tokens'], 4096)
    require(sorted(summary['context_execution_order']) == contexts, 'Invalid context execution order.')
    inputs = read_json(folder/'inputs.json')
    require(sha256_file(folder/'inputs.json') == summary['inputs_sha256'], 'Token input hash differs.')
    require(inputs['contexts'] == contexts and inputs['not_real_chat_requests'] is True,
            'Unexpected input provenance.')
    require(inputs['longest_input_ids'] == repeated_prefix(inputs['source_token_ids'], max(contexts)),
            'Controlled nested prefixes differ from recorded construction.')
    require(all(type(token) is int and 0 <= token < config['vocab_size']
                for token in inputs['longest_input_ids']), 'Invalid token IDs.')
    rows = [json.loads(line) for line in (folder/'raw.jsonl').read_text(encoding='utf-8').splitlines() if line]
    warmups, repeats = args['warmup'], args['repeats']
    require(len(rows) == len(contexts)*2*(warmups+repeats), 'Missing/extra timing records.')
    grouped = defaultdict(dict)
    max_errors = dict.fromkeys(('relative_l2','max_abs','normalized_max_abs'), 0.)
    for row in rows:
        require(row['context_tokens'] in contexts and row['path'] in PATHS, 'Unexpected context/path.')
        require(type(row['warmup']) is bool, 'Invalid warmup flag.')
        count = warmups if row['warmup'] else repeats
        require(type(row['trial']) is int and 0 <= row['trial'] < count, 'Invalid trial number.')
        require(math.isfinite(row['elapsed_ms']) and row['elapsed_ms'] > 0, 'Invalid duration.')
        require(row['object_bytes'] == spec['bytes'], 'Recovery object size changed.')
        require(sorted(row['order']) == sorted(PATHS) and row['order_position'] in (0,1),
                'Invalid path order.')
        require(row['order'][row['order_position']] == row['path'], 'Path/order mismatch.')
        key = (row['context_tokens'], row['warmup'], row['trial'])
        require(row['path'] not in grouped[key], 'Duplicate trial/path.')
        grouped[key][row['path']] = row
        errors = row['correctness']
        require(errors['pass'] is True, 'Recorded numerical failure.')
        for name in max_errors:
            value = errors[name]
            require(math.isfinite(value) and value >= 0, 'Invalid numerical error.')
            max_errors[name] = max(max_errors[name], value)
        require(errors['relative_l2'] <= .03 and errors['normalized_max_abs'] <= .05,
                'Numerical error above the fixed recompute threshold.')
        if row['path'] == 'restore':
            require(all(errors[name] == 0 for name in max_errors), 'Restore is not byte-exact in logged checks.')
    for pair in grouped.values():
        require(set(pair) == set(PATHS), 'Missing paired path.')
        require(pair['restore']['order'] == pair['recompute']['order'], 'Paired ordering records disagree.')
    require(len(grouped) == len(contexts)*(warmups+repeats), 'Missing trial groups.')
    points = {point['context_tokens']: point for point in summary['points']}
    require(len(points) == len(summary['points']) and sorted(points) == contexts, 'Missing/duplicate context summaries.')
    audited_points = []
    for length in contexts:
        point = points[length]
        require(point == read_json(folder/f'context-{length}-reference.json') | {'timings':point['timings']},
                'Reference metadata and summary disagree.')
        require(point['object'] == spec and point['block_range'] == [length-args['block_tokens'],length],
                'Different KV object or absolute token range.')
        source_name = point['source_file']
        require(source_name == f'context-{length}-source.safetensors', 'Unexpected source filename.')
        require(sha256_file(folder/source_name) == point['source_sha256'], 'Source KV hash differs.')
        tensor_header(folder/source_name, spec)
        stats = {}
        for path in PATHS:
            samples = [grouped[(length,False,trial)][path]['elapsed_ms'] for trial in range(repeats)]
            stats[path] = summarize(samples)
            check_stats(point['timings'][path], stats[path])
        audited_points.append({'context_tokens':length, 'timings':stats,
                               'reference_creation_ms_not_a_trial':point['reference_creation_ms_not_a_trial'],
                               'source_sha256':point['source_sha256']})
    before = read_json(folder/'resources_before.json')
    after = read_json(folder/'resources_after_load.json')
    affinity = parse_int_list(args['cpu_affinity'],0,4095)
    require(after['allowed_logical_cpus'] == affinity, 'CPU affinity differs from requested mask.')
    return {'audit_status':'passed', 'input_folder':str(folder),
            'file_sha256':{p.name:sha256_file(p) for p in sorted(folder.iterdir()) if p.is_file()},
            'model_content_manifest_sha256':summary['model_content_manifest_sha256'],
            'record_count':len(rows), 'measured_count':sum(not row['warmup'] for row in rows),
            'warmup_count':sum(row['warmup'] for row in rows), 'max_logged_errors':max_errors,
            'warmup_per_point':warmups, 'repeats_per_point':repeats, 'object':spec,
            'cpu_threads':args['cpu_threads'], 'cpu_affinity':affinity,
            'gpu_idle_at_start':all(gpu['used_bytes']==0 and gpu['util_percent']==0 for gpu in before['gpus']),
            'cuda_peak_allocated_gib':summary['cuda_peak_allocated_bytes']/GIB,
            'cuda_peak_reserved_gib':summary['cuda_peak_reserved_bytes']/GIB,
            'gpu_weight_layer_count':sum(mapping.get(f'layers.{layer}') in ('0','cuda:0') for layer in range(48)),
            'cpu_weight_layer_count':sum(mapping.get(f'layers.{layer}') == 'cpu' for layer in range(48)),
            'points':audited_points,
            'limits':['Correctness metrics are remotely logged checks, not local re-execution of the model.',
                      'Checkpoint weights are not downloaded; only recorded checkpoint identity is checked.',
                      'Two contexts in one session do not establish linearity or allocator gains.',
                      'Prefix replay cost is shared across rebuilt blocks, not additive for each miss.']}


def report(result):
    lines = ['# KV 两路径数据核验', '', '原始文件哈希、记录完整性、统计重算与 KV 对象检查全部通过。', '',
             f"- 原始目录：`{result['input_folder']}`",
             f"- {result['record_count']} 条原始计时：{result['measured_count']} 条正式样本、{result['warmup_count']} 条预热。",
             f"- 固定对象：{result['object']['block_tokens']} tokens，跨全部 48 层，BF16 K/V，共 {result['object']['bytes']/1024**2:g} MiB。",
             f"- 所有远端数值检查通过，最大 relative L2 / normalized max error 均为 {result['max_logged_errors']['relative_l2']:g} / {result['max_logged_errors']['normalized_max_abs']:g}。",
             f"- CPU {result['cpu_threads']} threads，亲和 CPU {result['cpu_affinity']}；{result['gpu_weight_layer_count']} 层权重位于 GPU，{result['cpu_weight_layer_count']} 层卸载到 CPU，执行仍在 GPU。",
             f"- PyTorch peak allocated / reserved：{result['cuda_peak_allocated_gib']:.3f} / {result['cuda_peak_reserved_gib']:.3f} GiB。", '',
             '| 上下文 tokens | Restore 中位数 ms | Recompute 中位数 ms |', '|---:|---:|---:|']
    for point in result['points']:
        lines.append(f"| {point['context_tokens']} | {point['timings']['restore']['median_ms']:.6f} | {point['timings']['recompute']['median_ms']:.6f} |")
    lines += ['', '## 计费解释与可用结论', '',
              'Restore 搬运已经存在的 pinned CPU 块。Recompute 没有可复用前缀 KV/隐藏状态，执行原始 token 的全前缀，再提取末尾同一块。',
              '它包括 CPU 卸载权重搬运与真实 Transformer 计算，不是单层 KV 投影，也不是已隔离出的 GPU 算术时间。',
              '实测中位数见上表；上下文变化的趋势、成本拟合与验证另行分析，不由本核验自动判定。',
              '不声称存在路径交叉、严格线性、完整分配器收益或端到端加速。前缀重放能共同恢复多个块，不能对每块重复加收相同费用。', '',
              '## 初次参考生成', '']
    for point in result['points']:
        lines.append(f"- context={point['context_tokens']}：参考生成 {point['reference_creation_ms_not_a_trial']/1000:.3f} 秒，按预先定义的协议单独留档，不纳入正式中位数。")
    lines += ['', '第一份参考显著慢于后续重算，具体原因未经隔离实验确认，不能直接断定是磁盘、CUDA 首次初始化或缓存；该数据没有删除。', '',
              '## 成本分析协议', '',
              '五长度扫描约定为 128/256/512/1024/2048 tokens，预热 2 次、正式 5 次，seed=23；此段不是运行状态声明。',
              '只改变长度、重复与随机顺序；使用默认相同权重预算及亲和性。预先保留全部长度，不按收益筛选。',
              '预先指定新会话 128/512/2048 三点作为常数/仿射模型标定，256/1024 留作长度插值检查；',
              '因输入同源且分析已看过旧试跑，这不是自然 workload 泛化或完全盲测。新会话全部点另报拟合残差。',
              '若重算在全范围始终更贵，保留此结果；Expert 两路径的交叉证据与 KV 数据分别陈述。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'report.md').write_text(report(result),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('audit_status','record_count','measured_count','max_logged_errors')},indent=2))
    print('Report:', args.output/'report.md')


if __name__ == '__main__':
    main()
