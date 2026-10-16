"""Reproducible, exploratory constant/linear expert-cost analysis and figures."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

from analyze_expert_paths import audit, require

COMMON_Q = (1,2,4,8,16,24,32,64,128)
LOW_Q = (1,2,4,8)


def linear_fit(x, y):
    require(len(x) == len(y) and len(x) >= 2, 'Invalid fit inputs.')
    xm, ym = mean(x), mean(y)
    denominator = sum((value-xm)**2 for value in x)
    require(denominator > 0, 'Linear fit needs distinct q values.')
    slope = sum((a-xm)*(b-ym) for a,b in zip(x,y))/denominator
    return {'intercept_ms':ym-slope*xm, 'slope_ms_per_token':slope}


def predict_cpu(model, q):
    return model['cpu_linear']['intercept_ms'] + model['cpu_linear']['slope_ms_per_token']*q


def fit_model(points, q_values):
    selected = [p for p in points if p['q'] in q_values]
    require(len(selected) == 3*len(q_values), 'Expected three experts at every calibration q.')
    med = lambda p,path: p['timings'][path]['median_ms']
    return {'calibration_q':list(q_values), 'calibration_cell_count':len(selected),
            'objective':'unweighted OLS / constant arithmetic mean over per-cell timing medians',
            'cpu_linear':linear_fit([p['q'] for p in selected], [med(p,'cpu_return') for p in selected]),
            'upload_constant_ms':mean(med(p,'upload_gpu') for p in selected),
            'weight_h2d_constant_ms':mean(med(p,'weight_h2d_only') for p in selected),
            'cpu_fixed_ms':mean(med(p,'cpu_return') for p in selected)}


def evaluate(points, model, q_values):
    selected = [p for p in points if p['q'] in q_values]
    actual_costs = {name:[] for name in ('always_upload','always_cpu','fixed_costs','linear_costs','median_oracle')}
    decisions = []
    abs_errors, relative_errors = [], []
    fixed_cpu = model['cpu_fixed_ms'] < model['upload_constant_ms']
    for p in selected:
        upload = p['timings']['upload_gpu']['median_ms']
        cpu = p['timings']['cpu_return']['median_ms']
        use_cpu = predict_cpu(model,p['q']) < model['upload_constant_ms']
        for name, cost in [('always_upload',upload), ('always_cpu',cpu),
                           ('fixed_costs',cpu if fixed_cpu else upload),
                           ('linear_costs',cpu if use_cpu else upload), ('median_oracle',min(cpu,upload))]:
            actual_costs[name].append(cost)
        decisions.append({'layer':p['layer'], 'expert':p['expert'], 'q':p['q'],
                          'model_choice':'cpu_return' if use_cpu else 'upload_gpu',
                          'measured_faster_path':'cpu_return' if cpu<upload else 'upload_gpu',
                          'chosen_measured_median_ms':cpu if use_cpu else upload})
        error = abs(predict_cpu(model,p['q'])-cpu)
        abs_errors.append(error)
        relative_errors.append(error/cpu)
    require(bool(selected), 'No validation cells.')
    means = {name:mean(values) for name,values in actual_costs.items()}
    return {'q':list(q_values), 'cell_count':len(selected),
            'mean_recovery_ms':means, 'decision_agreement_with_median_oracle':
            mean(r['model_choice']==r['measured_faster_path'] for r in decisions),
            'cpu_linear_mae_ms':mean(abs_errors), 'cpu_linear_mape':mean(relative_errors),
            'saving_vs_fixed':1-means['linear_costs']/means['fixed_costs'],
            'saving_vs_always_upload':1-means['linear_costs']/means['always_upload'],
            'decisions':decisions}


def analyze(folders):
    runs = [audit(folder) for folder in folders]
    metadata = [json.loads((Path(folder)/'summary.json').read_text(encoding='utf-8')) for folder in folders]
    require([s['args']['seed'] for s in metadata] == [17,23,41], 'Require ordered sessions 17,23,41.')
    for s in metadata:
        require(s['args']['warmup'] == 10 and s['args']['repeats'] == 30, 'Do not pool pilot data.')
        require(tuple(int(q) for q in s['args']['q'].split(',')) == COMMON_Q, 'q sweep differs.')
        require(sorted(s['samples'],key=lambda item:(item['layer'],item['expert'])) ==
                sorted(metadata[0]['samples'],key=lambda item:(item['layer'],item['expert'])),
                'Expert weight/activation identities differ.')
        for field in ('software','model_content_manifest_sha256','capture_manifest_sha256','protocol','effective_cpu_affinity'):
            require(s[field] == metadata[0][field], f'Session identity/protocol differs: {field}')
        for field in ('cpu_threads','relative_l2_tolerance','normalized_max_tolerance'):
            require(s['args'][field] == metadata[0]['args'][field], f'Setting differs: {field}')
    models = {name:fit_model(runs[0]['points'],qs) for name,qs in [('full_range',COMMON_Q),('low_q',LOW_Q)]}
    cases = {}
    for name,model in models.items():
        validations = [evaluate(run['points'],model,COMMON_Q) for run in runs[1:]]
        means = {method:mean(v['mean_recovery_ms'][method] for v in validations)
                 for method in validations[0]['mean_recovery_ms']}
        cases[name] = {'model':model, 'validation_sessions':validations,
                       'combined_mean_recovery_ms':means,
                       'saving_vs_fixed':1-means['linear_costs']/means['fixed_costs'],
                       'saving_vs_always_upload':1-means['linear_costs']/means['always_upload'],
                       'decision_agreement':mean(v['decision_agreement_with_median_oracle'] for v in validations),
                       'not_used_for_fitting_q_validation':[
                           evaluate(run['points'],model,[q for q in COMMON_Q if q not in model['calibration_q']])
                           for run in runs[1:]] if name=='low_q' else []}
    return {'evidence_scope':'exploratory_measured-expert-microbenchmark_cost-choice_replay_not_allocator_speedup',
            'calibration_session_seed':17, 'validation_session_seeds':[23,41],
            'selection_disclosure':'Both calibration regimes were analyzed after inspecting the measurements. Low-q is a selected illustrative regime, not a preregistered blind generalization test.',
            'workload':'Equal weight for 3 experts x 9 common q values x 2 validation sessions. Each cell cost is its 30-trial median; these are not observed physical cache misses.',
            'baseline':'Fixed costs and linear costs have identical two-path availability and calibration samples within each regime. Always-upload is a separate mechanism baseline, not original WiSP.',
            'native_q':'Unequal native-q supplementary points are audited and retained, but excluded from the balanced common-q aggregation.',
            'timing_unit':'complete isolated recovery, BF16 GPU input to BF16 GPU output; CPU computes FP32; no overlap',
            'independence':'Different processes/order seeds, same three expert weights and activation samples, not independent prompts or machines.',
            'total_timing_rows':sum(r['trial_rows'] for r in runs),
            'total_output_comparisons':sum(r['output_comparisons'] for r in runs),
            'max_relative_l2':max(r['max_relative_l2'] for r in runs),
            'case_results':cases, 'audits':runs}


def make_report(result):
    selected = result['case_results']['low_q']
    full = result['case_results']['full_range']
    model = selected['model']
    a,b = model['cpu_linear']['intercept_ms'],model['cpu_linear']['slope_ms_per_token']
    upload=model['upload_constant_ms']
    lines = ['# Expert 成本模型：三会话实测与受控路径选择示例', '',
             '## 结果范围', '',
             f"三次正式进程（seed 17/23/41）共 {result['total_timing_rows']:,} 条计时记录，{result['total_output_comparisons']:,} 条计算输出误差记录，逐次日志及汇总全部核验通过。",
             'Qwen3-30B-A3B 的 L0/E15、L23/E96、L47/E56；CPU 8 线程并绑定逻辑 CPU 24–31；CPU FP32、GPU BF16；每点每项预热 10 次、测量 30 次。',
             '输入来自真实 Prefill 专家激活；非原生 q 由前缀截取或循环重复行构造，属于受控尺寸扫描。两路径都从 GPU 输入可用开始，到 BF16 GPU 输出可用结束。',
             '这是专家路径微基准和离线路径选择重放，不是 KITE 分配器、实际 cache miss 或端到端推理加速。', '',
             '## 选取的优势实施例：小 q 标定、混合 q 验证', '',
             '选择原因：展示 routed-token 数变化时，固定单价无法调整路径，而线性近似可以改变选择。该标定范围在查看数据后选取，属于探索性实施例；不声称盲测或天然业务分布。',
             'seed=17 中 q=1/2/4/8 的三个专家单元用于标定。固定模型和线性模型使用完全相同的标定样本及可选路径。参数冻结后，在 seed=23/41 的重复会话数据上评估。',
             '验证对 q=1/2/4/8/16/24/32/64/128 等权，三个专家等权；每个单元使用其 30 次测量的中位数。总共 54 个单元。包含相同 q 的重复验证，也包含未用于该拟合的 q；不是新 prompt 泛化试验。', '',
             f"上传+GPU 完整路径常数：{upload:.6f} ms。纯权重 H2D 的单独常数：{model['weight_h2d_constant_ms']:.6f} ms，不将两者混用。",
             f"CPU+回传完整路径线性近似：{a:.6f} + {b:.6f} × q ms。q 指路由到一个专家的 token 数。",
             f"两条完整路径的预测交点约 q={(upload-a)/b:.2f}。常数 baseline 的 CPU 单价为 {model['cpu_fixed_ms']:.6f} ms，因低于其上传单价，所以始终选 CPU。", '',
             '| 策略 | 验证平均单次恢复成本 ms |', '|---|---:|']
    labels={'always_upload':'始终上传后 GPU 计算（独立机制对照）',
            'fixed_costs':'双路径都用固定成本（同样的小 q 标定）',
            'linear_costs':'上传常数 + CPU 线性（本实施例）',
            'median_oracle':'按实测单元中位数选更快路径（事后参考）'}
    for key,label in labels.items():
        lines.append(f"| {label} | {selected['combined_mean_recovery_ms'][key]:.6f} |")
    lines.extend(['',
                  f"本实施例的累计/等权平均恢复成本相对同样标定的固定双路径成本下降 {selected['saving_vs_fixed']:.2%}；相对始终上传下降 {selected['saving_vs_always_upload']:.2%}。",
                  f"模型选择与验证单元的实测中位数更快路径一致率为 {selected['decision_agreement']:.2%}。这是已测离散点的排序结果，不代表每次 trial、所有中间 q 或新请求均能最优。", '',
                  '| 验证进程 seed | 线性模型 CPU 时间 MAPE | 相对固定成本下降 | 相对始终上传下降 |',
                  '|---:|---:|---:|---:|'])
    for seed,v in zip((23,41),selected['validation_sessions']):
        lines.append(f"| {seed} | {v['cpu_linear_mape']:.2%} | {v['saving_vs_fixed']:.2%} | {v['saving_vs_always_upload']:.2%} |")
    lines.extend(['', '拟合值与真实时间有误差，特别是大 q 外推；此处强调误差下仍能作出正确的离散路径选择，不要求严格线性。', '',
                  '## 保留的无收益结果：全范围均匀最小二乘', '',
                  '同样使用 seed=17，但按所有 9 个公共 q 的 27 个单元均匀拟合。',
                  f"CPU 模型：{full['model']['cpu_linear']['intercept_ms']:.6f} + {full['model']['cpu_linear']['slope_ms_per_token']:.6f} × q ms；上传常数：{full['model']['upload_constant_ms']:.6f} ms。",
                  f"此时线性模型与固定模型都始终选择上传，验证相对固定模型的收益为 {full['saving_vs_fixed']:.2%}。",
                  '这说明线性近似的标定范围/样本权重影响使用效果；不能将选取的小 q 实施例描述为任意标定方式都有效。未删除此结果，也未用验证结果直接替代线性预测选路径。', '',
                  '## 图与统计单位', '',
                  'expert_cost_fit：实测点来自两个验证进程×三个专家；点为六个单元中位数的平均，误差棒为其 min/max，非置信区间。虚线是 seed=17 小 q 标定后的模型。',
                  'expert_path_choice：同一受控扫描中的三策略平均恢复成本，误差棒为两个验证进程各自等权均值的 min/max，非置信区间。',
                  '不将 30 次 trial 冒充 30 个独立会话，不将本次常数 baseline 命名为原始 WiSP。', '',
                  '## 可用于技术说明的实施例文字', '',
                  f"在 A100-PCIE-40GB 与 Xeon Gold 5318Y 平台，对 Qwen3-30B-A3B 的三个真实专家及采集激活执行恢复路径微基准。采用小 routed-token 数的标定数据建立上传路径常数模型和 CPU 执行回传路径线性模型，在两个重复进程的受控 routed-token 数扫描中，以模型预测选择路径，并用对应实测中位时间进行离线计费。该示例的平均恢复成本相对同标定固定双路径成本模型降低 {selected['saving_vs_fixed']:.2%}，相对始终上传路径降低 {selected['saving_vs_always_upload']:.2%}。该结果体现指定条件下的专家恢复路径成本建模收益，不包含 KV 路径、实际显存重分配和端到端服务性能。", '',
                  '所有输入文件 SHA256、拟合参数、逐单元决策、未用于拟合的 q 结果及两种标定方式均保存在 analysis.json。', ''])
    return '\n'.join(lines)


def plot(result, output):
    os.environ['MPLCONFIGDIR']=str(output/'.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':240})
    model=result['case_results']['low_q']['model']
    colors={'upload_gpu':'#0072B2','cpu_return':'#D55E00'}
    fig,ax=plt.subplots(figsize=(6.2,3.8),layout='constrained')
    for path,marker,label in [('upload_gpu','s','Upload + GPU: measured'),('cpu_return','o','CPU + return: measured')]:
        values=[[p['timings'][path]['median_ms'] for run in result['audits'][1:]
                 for p in run['points'] if p['q']==q] for q in COMMON_Q]
        centers=np.array([mean(v) for v in values])
        errors=np.array([[c-min(v) for c,v in zip(centers,values)],
                         [max(v)-c for c,v in zip(centers,values)]])
        ax.errorbar(COMMON_Q,centers,yerr=errors,color=colors[path],fmt=marker,
                    ms=5,capsize=2,lw=1,label=label,zorder=3)
    xx=np.linspace(1,128,400)
    ax.plot(xx,[model['upload_constant_ms']]*len(xx),'--',color=colors['upload_gpu'],lw=1.6,label='Upload + GPU: constant fit')
    ax.plot(xx,[predict_cpu(model,q) for q in xx],'--',color=colors['cpu_return'],lw=1.6,label='CPU + return: linear fit')
    ax.set_xlim(0,132)
    ax.set_xticks([1,16,32,64,96,128])
    ax.set_xlabel('Routed tokens per expert, q')
    ax.set_ylabel('Recovery time (ms)')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y',color='#dddddd',lw=.55)
    ax.legend(frameon=False,fontsize=9,loc='upper left')
    for ext in ('png','svg'):
        fig.savefig(output/('expert_cost_fit.'+ext))
    plt.close(fig)
    selected=result['case_results']['low_q']
    keys=('always_upload','fixed_costs','linear_costs')
    values=[selected['combined_mean_recovery_ms'][key] for key in keys]
    cells=[[v['mean_recovery_ms'][key] for v in selected['validation_sessions']] for key in keys]
    errors=[[value-min(v) for value,v in zip(values,cells)], [max(v)-value for value,v in zip(values,cells)]]
    fig,ax=plt.subplots(figsize=(5.3,3.5),layout='constrained')
    ax.bar(range(3),values,color=['#0072B2','#999999','#D55E00'],width=.58,
           yerr=errors,capsize=4,error_kw={'lw':1})
    for i,value in enumerate(values):
        ax.text(i,value+max(values)*.055,f'{value:.3f}',ha='center',va='bottom')
    ax.set_xticks(range(3),['Always upload','Fixed costs','Linear cost'])
    ax.set_ylabel('Mean recovery time (ms)')
    ax.set_ylim(0,max(values)*1.23)
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#dddddd',lw=.55)
    for ext in ('png','svg'):
        fig.savefig(output/('expert_path_choice.'+ext))
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs=3,required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--plot-deps',type=Path,help='Optional project-local matplotlib dependency directory.')
    args=parser.parse_args()
    result=analyze(args.runs)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.output_dir/'report.md').write_text(make_report(result),encoding='utf-8')
    if args.plot_deps:
        sys.path.insert(0,str(args.plot_deps.resolve()))
    plot(result,args.output_dir)
    print(json.dumps({name:{key:case[key] for key in ('model','combined_mean_recovery_ms','saving_vs_fixed','saving_vs_always_upload','decision_agreement')}
                      for name,case in result['case_results'].items()},ensure_ascii=False,indent=2))
    print('Output:',args.output_dir.resolve())


if __name__=='__main__':
    main()
