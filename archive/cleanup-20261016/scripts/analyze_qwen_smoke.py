"""Offline audit of actual smoke artifacts. No torch, model loading or GPU use."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def analyze(folder):
    folder = Path(folder)
    names = ['summary.json', 'request.json', 'model_config.json', 'device_map.json',
             'resources_before.json', 'resources_after_load.json', 'expert_trace.jsonl',
             'routing_checks.jsonl', 'kv_snapshots.jsonl', 'steps.jsonl']
    hashes = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}
    summary, request, config, mapping = [json.loads((folder/name).read_text(encoding='utf-8'))
                                        for name in names[:4]]
    require(summary['status'] == 'passed', 'Run did not pass.')
    require(summary['is_tiny_synthetic_test'] is False, 'Not a real-checkpoint run.')
    require(not summary['is_kite_allocator'] and not summary['is_fallback_benchmark'],
            'Unexpected experiment scope; needs another audit protocol.')
    require(all(v in ('cpu', '0') for v in mapping.values()), 'Unexpected storage placement.')
    steps = read_jsonl(folder/'steps.jsonl')
    checks = read_jsonl(folder/'routing_checks.jsonl')
    snapshots = read_jsonl(folder/'kv_snapshots.jsonl')
    events = read_jsonl(folder/'expert_trace.jsonl')
    n = summary['forward_count']
    layers, top_k = config['num_hidden_layers'], config['num_experts_per_tok']
    input_tokens = request['input_tokens']
    require(input_tokens == len(request['input_token_ids']), 'Input token count mismatch.')
    require(len(steps) == len(checks) == len(snapshots) == n, 'Incomplete forward records.')
    require(len(summary['generated_token_ids']) == n, 'Output token count mismatch.')
    require(summary['decode_forward_count'] == n-1, 'Decode count mismatch.')
    grouped = defaultdict(list)
    expert_bytes = 3 * config['hidden_size'] * config['moe_intermediate_size'] * 2
    kv_bytes_per_token = layers * 2 * config['num_key_value_heads'] * config['head_dim'] * 2
    for sequence, event in enumerate(events):
        require(event['sequence'] == sequence, 'Expert sequence gap.')
        require(event['request_id'] == request['request_id'], 'Request ID mismatch.')
        require(0 <= event['step'] < n, 'Expert step outside run.')
        require(0 <= event['layer'] < layers, 'Invalid layer.')
        require(0 <= event['expert'] < config['num_experts'], 'Invalid expert.')
        require(event['routed_tokens'] > 0, 'Trace contains empty expert call.')
        require(event['weight_bytes'] == expert_bytes, 'Expert weight size mismatch.')
        require(event['input_device'] == 'cuda:0', 'Expert input not on expected GPU.')
        require(event['semantics'] == 'nonempty_expert_module_invocation', 'Unknown trace meaning.')
        key = f"model.layers.{event['layer']}"
        require(event['weight_storage_placement'] == mapping[key], 'Placement mismatch.')
        grouped[event['step']].append(event)
    step_stats = []
    for i, (step, check, snapshot) in enumerate(zip(steps, checks, snapshots)):
        phase = 'prefill' if i == 0 else 'decode'
        query = input_tokens if i == 0 else 1
        before = 0 if i == 0 else input_tokens+i-1
        after = before+query
        for record in (step, check, snapshot):
            require(record['step'] == i and record['phase'] == phase, 'Step/phase mismatch.')
        for record in (step, check, *grouped[i]):
            require(record['query_tokens'] == query and record['context_before'] == before
                    and record['phase'] == phase, 'Forward metadata mismatch.')
        require(step['context_after'] == snapshot['length'] == after, 'KV length mismatch.')
        require(step['next_token_id'] == summary['generated_token_ids'][i], 'Generated token mismatch.')
        counts = Counter()
        for event in grouped[i]:
            counts[event['layer']] += event['routed_tokens']
        require(set(counts) == set(range(layers)), 'Missing expert layer.')
        require(all(counts[layer] == query*top_k for layer in range(layers)),
                'Independent routed-token conservation check failed.')
        require(check['pass'] is True and len(check['checks']) == layers, 'Stored check failed.')
        require({row['layer'] for row in check['checks']} == set(range(layers)), 'Check layer IDs mismatch.')
        require(all(row['pass'] and row['expected'] == row['observed'] == query*top_k
                    for row in check['checks']), 'Stored layer check mismatch.')
        distinct = {(event['layer'], event['expert']) for event in grouped[i]}
        require(len(distinct) == len(grouped[i]) == check['nonempty_calls'],
                'Expected one nonempty invocation per expert per forward.')
        require(check['ignored_empty_calls'] + len(distinct) == layers*config['num_experts'],
                'Pinned backend invocation count mismatch.')
        require(len(snapshot['layers']) == layers, 'Missing KV layer.')
        for layer, row in enumerate(snapshot['layers']):
            require(row['layer'] == layer, 'KV layer ID mismatch.')
            for kind in ('key', 'value'):
                require(row[f'{kind}_shape'] == [1, config['num_key_value_heads'], after, config['head_dim']],
                        'KV shape mismatch.')
                require(row[f'{kind}_dtype'] == 'torch.bfloat16' and row[f'{kind}_device'] == 'cuda:0',
                        'KV dtype/device mismatch.')
            require(row['logical_tensor_bytes'] == kv_bytes_per_token*after//layers,
                    'KV layer bytes mismatch.')
        require(snapshot['logical_tensor_bytes'] == kv_bytes_per_token*after, 'KV total bytes mismatch.')
        step_stats.append({'step': i, 'phase': phase, 'query_tokens': query,
                           'nonempty_expert_calls': len(distinct),
                           'routed_token_assignments': sum(counts.values()),
                           'mean_tokens_per_nonempty_expert': sum(counts.values())/len(distinct),
                           'max_tokens_per_nonempty_expert': max(e['routed_tokens'] for e in grouped[i]),
                           'ignored_empty_calls': check['ignored_empty_calls'],
                           'kv_tokens': after, 'kv_logical_bytes': snapshot['logical_tensor_bytes'],
                           'diagnostic_seconds': step['diagnostic_forward_seconds_with_hooks_and_offload']})
    require(summary['final_cache_length'] == input_tokens+n-1, 'Final KV length mismatch.')
    by_phase = {}
    for phase in ('prefill', 'decode'):
        rows = [r for r in step_stats if r['phase'] == phase]
        selected = [e for e in events if e['phase'] == phase]
        assignments = sum(e['routed_tokens'] for e in selected)
        by_phase[phase] = {'forwards': len(rows), 'expert_calls': len(selected),
                           'distinct_layer_expert_objects': len({(e['layer'], e['expert']) for e in selected}),
                           'mean_expert_calls_per_forward': mean(r['nonempty_expert_calls'] for r in rows),
                           'routed_token_assignments': assignments,
                           'mean_tokens_per_nonempty_expert': assignments/len(selected),
                           'routed_tokens_histogram': dict(sorted(Counter(e['routed_tokens'] for e in selected).items())),
                           'diagnostic_mean_seconds': mean(r['diagnostic_seconds'] for r in rows)}
    return {'audit_status': 'passed', 'input_folder': str(folder.resolve()), 'sha256': hashes,
            'model_source': summary['model_source'], 'model_revision': summary['model_revision'],
            'model_content_manifest_sha256': summary['model_content_manifest_sha256'],
            'software': summary['software'], 'input_tokens': input_tokens, 'output_tokens': n,
            'generated_text': summary['generated_text'], 'layers': layers, 'top_k': top_k,
            'layer_forward_checks_passed': layers*n, 'expert_event_count': len(events),
            'expert_object_bytes': expert_bytes, 'kv_bytes_per_token': kv_bytes_per_token,
            'cpu_weight_bytes_planned': summary['cpu_weight_bytes_planned'],
            'gpu_storage_layer_count': sum(mapping[f'model.layers.{i}'] == '0' for i in range(layers)),
            'cpu_storage_layer_count': sum(mapping[f'model.layers.{i}'] == 'cpu' for i in range(layers)),
            'max_cuda_allocated_bytes': max(r['cuda_peak_allocated_bytes'] for r in steps),
            'max_cuda_reserved_bytes': max(r['cuda_peak_reserved_bytes'] for r in steps),
            'steps': step_stats, 'phases': by_phase,
            'limitations': ['One handwritten request, not representative workload evaluation.',
                            'Expert invocations are not physical cache misses or transfers.',
                            'KV tensor metadata is not a KV block access trace.',
                            'CPU weight storage does not establish CPU expert execution.',
                            'Instrumented cold/offloaded diagnostics are not a controlled latency benchmark.',
                            'No KITE allocator or four-path benchmark executed.',
                            'Log hashes identify downloaded artifacts, not independent execution attestation.']}


def render_report(result):
    p, d = result['phases']['prefill'], result['phases']['decode']
    rows = ['# Qwen 真实推理首轮日志核验', '',
            f"原始记录：`{result['input_folder']}`。离线核验通过；没有重新运行模型。", '',
            f"Qwen3-30B-A3B，BF16；输入 {result['input_tokens']} tokens，输出 {result['output_tokens']} tokens。",
            f"生成内容：{result['generated_text']}。", '',
            '| 指标 | Prefill | Decode |', '|---|---:|---:|',
            f"| Forward 次数 | {p['forwards']} | {d['forwards']} |",
            f"| 每次 forward 的非空专家调用数 | {p['mean_expert_calls_per_forward']:.0f} | {d['mean_expert_calls_per_forward']:.0f} |",
            f"| 每次非空专家调用平均 routed tokens | {p['mean_tokens_per_nonempty_expert']:.3f} | {d['mean_tokens_per_nonempty_expert']:.3f} |",
            f"| 全阶段 routed-token assignments | {p['routed_token_assignments']} | {d['routed_token_assignments']} |", '',
            'Assignment 指一个 token 被分配给一个专家；top-8 下一个 token 有 8 个 assignments。',
            'Expert 标识是 (layer, expert)，跨层同编号不是同一个缓存对象。', '',
            f"共 {result['expert_event_count']} 条非空专家记录；{result['layer_forward_checks_passed']} 个层×forward 的路由守恒检查通过。",
            f"固定后端每轮还执行空输入专家调用，已单独计数，不将其记为逻辑专家需求；不能据此推算物理传输。", '',
            '## KV 与权重放置', '',
            f"单专家 BF16 权重 {result['expert_object_bytes']/2**20:.0f} MiB；全层 KV 每 token {result['kv_bytes_per_token']/1024:.0f} KiB。",
            f"KV 从 {result['steps'][0]['kv_tokens']} 增至 {result['steps'][-1]['kv_tokens']} tokens，",
            f"逻辑张量占用从 {result['steps'][0]['kv_logical_bytes']/2**20:.5f} 增至 {result['steps'][-1]['kv_logical_bytes']/2**20:.5f} MiB。",
            '全部 48 层 K/V 的形状、BF16 精度、GPU 位置与字节数通过核验。最后生成的 token 没有再次输入模型，故 KV 不包含它。', '',
            f"权重放置：{result['gpu_storage_layer_count']} 层常驻 GPU，{result['cpu_storage_layer_count']} 层存储在 CPU 并临时传入 GPU；不代表后者在 CPU 执行。",
            f"CPU 计划权重占用 {result['cpu_weight_bytes_planned']/2**30:.3f} GiB（非 RSS 实测）。",
            f"本次 PyTorch 已分配显存峰值 {result['max_cuda_allocated_bytes']/2**30:.3f} GiB；保留显存峰值 {result['max_cuda_reserved_bytes']/2**30:.3f} GiB。", '',
            '## 已支持与尚未支持的结论', '',
            '已支持：真实模型可以在当前 CPU/GPU 卸载环境生成输出；真实专家路由和 KV 观测链路可用。',
            '在这一个请求中，Prefill 聚合多个 token，非空专家调用覆盖更多对象，每次调用的 token 数存在差异；Decode 的单请求单 token 使非空调用的 routed tokens 为 1。',
            '这为按 routed-token 数标定专家 CPU 路径提供真实输入依据，但不能推出 Prefill 必然产生更多 miss，或某种显存划分已经更优。', '',
            '尚未测量：四条 fallback 的完成时间、真实缓存 miss、KITE 分配收益、端到端速度改善。',
            '计时含观测 hooks、首次执行和 CPU 权重卸载，只保留为诊断记录，不作为正式 TTFT/TPOT。', '',
            '## 逐轮记录', '',
            '| Step | Phase | Query tokens | 非空专家调用 | KV tokens | KV MiB | 诊断秒数 |',
            '|---:|---|---:|---:|---:|---:|---:|']
    for row in result['steps']:
        rows.append(f"| {row['step']} | {row['phase']} | {row['query_tokens']} | {row['nonempty_expert_calls']} | {row['kv_tokens']} | {row['kv_logical_bytes']/2**20:.5f} | {row['diagnostic_seconds']:.3f} |")
    rows.extend(['', '## 可追溯性', '',
                 f"模型来源 {result['model_source']}；分支 {result['model_revision']}（分支名本身不保证不可变性）。",
                 f"下载时冻结的文件清单指纹：`{result['model_content_manifest_sha256']}`。",
                 '本地核验 JSON 保存全部 10 个原始文件的 SHA256，用于追溯文件版本；不是第三方执行证明。', ''])
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in [('audit.json', json.dumps(result, ensure_ascii=False, indent=2)+'\n'),
                          ('report.md', render_report(result))]:
        with (args.output_dir/name).open('x', encoding='utf-8') as stream:
            stream.write(content)
    print(json.dumps({key: result[key] for key in ('audit_status', 'expert_event_count',
                     'layer_forward_checks_passed', 'phases')}, ensure_ascii=False, indent=2))
    print('Reports:', args.output_dir.resolve())


if __name__ == '__main__':
    main()
