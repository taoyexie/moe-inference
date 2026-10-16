"""Fit the predeclared KV calibration lengths, retaining unfavorable slopes and errors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from analyze_kv_paths import audit, require
from summarize_expert_models import linear_fit

CALIBRATION = (128, 512, 2048)
VALIDATION = (256, 1024)


def fit_and_evaluate(points):
    by_length = {p['context_tokens']:p for p in points}
    require(len(by_length) == len(points) and set(by_length) == set(CALIBRATION+VALIDATION),
            'Require exactly the five predeclared context lengths.')
    actual = lambda length,path: by_length[length]['timings'][path]['median_ms']
    model = {'restore_constant_ms':mean(actual(length,'restore') for length in CALIBRATION),
             'recompute_affine':linear_fit(list(CALIBRATION),[actual(length,'recompute') for length in CALIBRATION])}
    predictions = []
    for length in sorted(by_length):
        predicted = {'restore':model['restore_constant_ms'],
                     'recompute':model['recompute_affine']['intercept_ms']
                                 +model['recompute_affine']['slope_ms_per_token']*length}
        require(all(value > 0 for value in predicted.values()), 'Negative prediction in measured domain.')
        measured = {path:actual(length,path) for path in ('restore','recompute')}
        predictions.append({'context_tokens':length, 'role':'calibration' if length in CALIBRATION else 'validation',
                            'predicted_ms':predicted, 'measured_median_ms':measured,
                            'relative_abs_errors':{path:abs(predicted[path]-measured[path])/measured[path]
                                                   for path in predicted},
                            'model_choice':min(predicted,key=predicted.get),
                            'measured_faster_path':min(measured,key=measured.get)})
    validation = [p for p in predictions if p['role']=='validation']
    return {'calibration_lengths':list(CALIBRATION), 'validation_lengths':list(VALIDATION),
            'model':model, 'predictions':predictions,
            'validation_mape':{path:mean(p['relative_abs_errors'][path] for p in validation)
                               for path in ('restore','recompute')},
            'positive_context_growth_supported':model['recompute_affine']['slope_ms_per_token']>0,
            'scope':'One process session, controlled nested prefixes, not allocator or end-to-end performance.',
            'units':{'restore':'ms per one 128-token all-layer 12 MiB transfer',
                     'recompute':'ms per full-prefix replay task, NOT an independent per-block penalty'},
            'runtime_extrapolation_allowed':False,
            'selection':'Calibration/validation lengths specified before this scan; pilot lengths already inspected.',
            'warning':'Negative fitted slope is retained, not clipped or reinterpreted as a physical speedup law.'}


def write_report(result):
    model = result['model']
    fit = model['recompute_affine']
    lines = ['# KV 成本拟合与验证', '',
             '数据为已核验的 5 长度、每路径每长度 5 次正式计时；使用每点中位数，不混入预热或初次参考生成。',
             '按扫描前指定：128/512/2048 标定，256/1024 检查长度插值。所有点均保留。', '',
             f"- Restore 常数：{model['restore_constant_ms']:.6f} ms / 12 MiB 全层块。",
             f"- Recompute 仿射拟合：截距 {fit['intercept_ms']:.6f} ms，斜率 {fit['slope_ms_per_token']:.9f} ms/token。",
             f"- 留出两长度的平均绝对相对误差：restore {result['validation_mape']['restore']:.2%}，recompute {result['validation_mape']['recompute']:.2%}。", '',
             '| 长度 | 用途 | Restore 实测 ms | Restore 预测 ms | 重算实测 ms | 重算预测 ms |',
             '|---:|---|---:|---:|---:|---:|']
    for p in result['predictions']:
        a,b = p['measured_median_ms'],p['predicted_ms']
        lines.append(f"| {p['context_tokens']} | {p['role']} | {a['restore']:.6f} | {b['restore']:.6f} | {a['recompute']:.3f} | {b['recompute']:.3f} |")
    lines += ['', '## 解释', '',
              'Restore 的常数近似在本次扫描上误差较小；重算的拟合斜率为负，不能作为“上下文越长，重算越快”的物理规律，',
              '也不支持“本轮测出了随上下文正线性增长”。线性形式和拟合误差仍按原设计如实保留，不外推、不删点。',
              '权重卸载传输、启动与执行状态等均可能影响时间，但没有 profiling 证据来判定具体主因。',
              '五种长度上，模型选择与实测较快路径均为 restore；不能宣称本轮证明了 KV 路径选择有额外收益。', '',
              '## 冻结的计费前提', '',
              '不保存每层 hidden states，缺失 KV 若需重算则从原始 token 重放。',
              'Restore 按实际传回的固定全层块计费。KV 重算按一次真正执行的前缀重放任务计费，',
              '同次重放产生的多块共享计算，不对每个缺失块再收费。不能将这项重算成本直接乘普通块 MRC 的 miss 数。',
              '重放内已有 Expert 权重搬运与运算，不把这些内部访问再次作为独立 Expert 恢复收费。',
              '重放之外正常推理的 Expert misses 单独统计。恢复成本不是整个请求的总推理时间。', '',
              '## 下一步', '',
              '准备多请求的真实路由采集，而非继续扩大相同 KV 计时扫描。',
              '先 6 条受控请求：128/512/1024 tokens 各 2 条，每条最多生成 8 tokens；两种手写内容，保存精确 token IDs。',
              '采集器尽量每进程只加载一次模型；每个请求使用独立 KV；记录非空 Expert 调用、routed tokens、阶段与真实 KV 长度。',
              '这些是受控顺序请求，不称自然工作负载或真实 mixed-batch 服务。',
              '之后以明确构造的交错顺序做 trace 重放，冻结同样的缓存/预算/路径与非前视决策，进行 2×2 消融。',
              '宏重算的分组、内部 Expert 去重、临时 KV 显存和准入规则须先实现并测试；未完成前不输出联合收益数字。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    audited = audit(args.folder)
    result = fit_and_evaluate(audited['points'])
    result['source_file_sha256'] = audited['file_sha256']
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'report.md').write_text(write_report(result),encoding='utf-8')
    print(json.dumps({'model':result['model'],'validation_mape':result['validation_mape']},indent=2))


if __name__ == '__main__':
    main()
