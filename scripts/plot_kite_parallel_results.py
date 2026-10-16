"""Three minimal scientific figures from audited actual E2E records; no GPU use."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile,ZIP_DEFLATED

from audit_kite_e2e import read,require,same
from audit_kite_parallel_decode import audit


def figure_data(run):
    run = Path(run)
    verified = audit(run)
    rows = []
    for case in verified['cases']:
        for mode in ('baseline','kite'):
            folder = run/f'{case["case_name"]}-{case["budget_mib"]}-{mode}'
            perf = read(folder/'performance.json')
            clocks = perf['task_clock_ns']
            e2e = (clocks['finished']-clocks['start'])/1e9
            same(e2e,perf['workload_total_ms']/1000)
            requests = read(folder/'requests.json')
            tpots = [mean(b-a for a,b in zip(r['token_task_offsets_ms'],r['token_task_offsets_ms'][1:]))/1000
                for r in requests]
            tpot = mean(tpots)
            same(tpot,perf['request_metrics']['tpot_ms']['mean']/1000)
            recoveries = read(folder/'recoveries.json')
            restores = sum(r['kind']=='restore' for r in recoveries)
            require(restores==perf['kv_counts']['kv_restore_pages'],'Actual restore counts differ.')
            rows.append(dict(budget_gib=case['budget_mib']/1024,mode=mode,e2e_s=e2e,tpot_s=tpot,
                kv_restores=restores,request_count=len(requests),output_tokens=sum(r['output_tokens'] for r in requests),
                source=str(folder.resolve())))
    return dict(scope='single completed paired exploratory workload per budget, no independent repetitions or error bars',
        baseline='fixed path costs plus historical curves; not original WiSP',
        e2e_scope='pool initialization, startup calibration and complete generation including decisions/transfers/writes; model loading separate',
        tpot_scope='mean per-request time between first and last output divided by 15, within one four-request workload; not independent trials',
        source_run=str(run.resolve()),audited=True,rows=rows,raw_sha256=verified['raw_sha256'])


def plot(data,output,style='bars'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':12,'axes.labelsize':13,'xtick.labelsize':12,'ytick.labelsize':12,
        'axes.linewidth':1.,'axes.spines.top':False,'axes.spines.right':False,
        'axes.axisbelow':True,'savefig.facecolor':'white','svg.fonttype':'path',
        'hatch.linewidth':.5})
    styles = [('baseline','Fixed-cost baseline','#327CB7'),('kite','KITE','#E87732')]
    budgets = sorted({r['budget_gib'] for r in data['rows']})
    specs = [dict(name='01_e2e_latency',metric='e2e_s',label='End-to-end time (s)',
                  top=100,ticks=[0,20,40,60,80,100],format=lambda v:f'{v:.2f}',offset=1.7),
             dict(name='02_tpot',metric='tpot_s',label='Time per output token (s)',
                  top=1.9,ticks=[0,.4,.8,1.2,1.6],format=lambda v:f'{v:.3f}',offset=.032),
             dict(name='03_kv_restores',metric='kv_restores',label='KV restores (count)',
                  top=6000,ticks=[0,1000,2000,3000,4000,5000,6000],format=lambda v:f'{v:,d}',offset=105)]
    for spec in specs:
        fig,ax = plt.subplots(figsize=(6.2,4.25))
        fig.subplots_adjust(left=.16,right=.98,bottom=.19,top=.84)
        width = .29
        zoom = style=='paired-zoom' and spec['metric']!='kv_restores'
        if zoom:
            limits,ticks = ((74,84),[74,76,78,80,82,84]) if spec['metric']=='e2e_s' else ((1.36,1.58),[1.36,1.40,1.44,1.48,1.52,1.56])
            for i,budget in enumerate(budgets):
                values = {m:next(r[spec['metric']] for r in data['rows'] if r['budget_gib']==budget and r['mode']==m)
                    for m,_,_ in styles}
                # Paired points, not truncated bars: vertical distance is an absolute
                # difference. Exact percentage reductions remain numerically explicit.
                ax.annotate('',xy=(i,values['kite']),xytext=(i,values['baseline']),
                    arrowprops=dict(arrowstyle='->',color='#9AA6B1',lw=1.5,shrinkA=8,shrinkB=8),zorder=2)
                reduction=100*(1-values['kite']/values['baseline'])
                ax.text(i+.12,(values['kite']+values['baseline'])/2,f'\N{MINUS SIGN}{reduction:.2f}%',
                    ha='left',va='center',fontsize=13,color=styles[1][2],fontweight='bold')
            for index,(mode,label,color) in enumerate(styles):
                values = [next(r[spec['metric']] for r in data['rows'] if r['budget_gib']==b and r['mode']==mode) for b in budgets]
                ax.plot(list(range(len(budgets))),values,linestyle='none',marker='o' if index==0 else 's',
                    markersize=10,label=label,color=color,markeredgecolor='white',markeredgewidth=.8,zorder=3)
                for i,value in enumerate(values):
                    ax.text(i-.075,value,spec['format'](value),ha='right',va='center',fontsize=12,color='#202832')
            ax.set_ylim(*limits)
            ax.set_yticks(ticks)
        else:
            for index,(mode,label,color) in enumerate(styles):
                xs = [i+(index-.5)*.34 for i in range(len(budgets))]
                values = [next(r[spec['metric']] for r in data['rows'] if r['budget_gib']==b and r['mode']==mode) for b in budgets]
                bars = ax.bar(xs,values,width=width,label=label,color=color,edgecolor='#263547',linewidth=.65,zorder=3)
                for bar,value in zip(bars,values):
                    ax.text(bar.get_x()+bar.get_width()/2,value+spec['offset'],spec['format'](value),
                        ha='center',va='bottom',fontsize=12,color='#202832',zorder=4)
            ax.set_ylim(0,spec['top'])
            ax.set_yticks(spec['ticks'])
        ax.set_xlim(-.6,len(budgets)-.4)
        ax.set_xticks(list(range(len(budgets))),[f'{b:g}' for b in budgets])
        ax.set_xlabel('Pageable memory budget (GiB)',labelpad=10)
        ax.set_ylabel(spec['label'],labelpad=10)
        ax.tick_params(axis='both',length=4,pad=6)
        if spec['metric']=='kv_restores':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))
        elif zoom and spec['metric']=='tpot_s':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.2f}'))
        ax.grid(axis='y',color='#DCE1E6',linewidth=.7,zorder=0)
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.06),ncol=2,frameon=False,
            handlelength=1.5,handletextpad=.55,columnspacing=1.8,borderaxespad=0,fontsize=12)
        fig.savefig(output/(spec['name']+'.png'),dpi=400)
        fig.savefig(output/(spec['name']+'.svg'))
        plt.close(fig)


def captions(data,style='bars'):
    lines = ['# KITE 实测实验图','','图形只展示核验后的原始实测值，PNG 为 400 dpi，SVG 为可缩放矢量图。',
        '蓝色：固定成本基线（历史曲线分配）；橙色：完整 KITE。不是原始 WiSP。','',
        '## 图 1：完整工作负载完成时间','',
        '在 6/7 GiB 可分页 Expert–KV 显存池下，对比完整 workload 的实际完成时间。',
        '包含池初始化、本 workload 启动标定、全部 prefill/decode、在线决策及实际搬运和写回；模型加载单列。','',
        '## 图 2：输出 token 间隔（TPOT）','',
        '按每条请求首个至末个输出的时间差除以 15，随后对同一 workload 的四条请求取均值。',
        '不是四次独立实验；不把该指标称为包含 prefill 的完整 E2E。','',
        '## 图 3：实际 KV 恢复次数','',
        '统计完整 workload 中实际执行的 KV restore 页数，不是 MRC 的参考 miss，也不是拟合成本。','',
        '## 统一实验设置','',
        'Qwen3-30B-A3B、BF16、A100 PCIe 40GB；四条请求同时进入，输入为 8192/11264/11264/11264 tokens，',
        '每条生成 16 tokens，共 64 个真实输出 tokens。每组独立标定；两组共用执行器、fallback 集合和预算。',
        '6 GiB 基线先跑，7 GiB KITE 先跑。每个预算每种配置目前只有一次完整测量，因此不画误差条，',
        '不声称多次独立复测平均值或统计显著性。结果适用于此受控长输入、短输出场景。','',
        '## 数据与效果','',
        '| 预算 GiB | 配置 | E2E s | TPOT s/token | KV restores |','|---:|---|---:|---:|---:|']
    for r in data['rows']:
        lines.append(f'| {r["budget_gib"]:g} | {r["mode"]} | {r["e2e_s"]:.6f} | {r["tpot_s"]:.6f} | {r["kv_restores"]} |')
    lines += ['','可用的简短效果描述：', '',
        '在上述受控四请求实验中，与固定成本的历史曲线基线相比，完整 KITE 在 6 GiB 池预算下，',
        '完整工作负载耗时由 82.174 秒降至 75.348 秒，下降 8.31%；实际 KV restore 由 4934 次降至 200 次。',
        '7 GiB 下完整耗时由 78.091 秒降至 75.634 秒，下降 3.15%。以上为各配置一次配对实测结果。','',
        '数据溯源见 figure_data.json；原始记录未修改。']
    if style=='paired-zoom':
        lines += ['','图 1/2 使用配对点图而不是截断柱状图，纵轴分别为 74–84 秒、1.36–1.58 秒。',
            '箭头连接同一预算的基线与 KITE，数字为原始实测值；橙色百分数为耗时下降比例，按 1−KITE/基线计算。',
            '点距表达绝对差值，不可把点距或图高当作比例。图 3 的柱状图仍从零开始。']
    else:
        lines += ['图中坐标从零开始，不使用截断坐标夸大差异。']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--style',choices=['bars','paired-zoom'],default='bars')
    args = parser.parse_args()
    data = figure_data(args.run)
    data['visualization'] = dict(style=args.style,nonzero_y_axes=args.style=='paired-zoom',
        y_ranges=dict(e2e_s=[74,84],tpot_s=[1.36,1.58],kv_restores=[0,6000]) if args.style=='paired-zoom' else None)
    require(not args.output.exists(),'Refusing to overwrite figures.')
    args.output.mkdir(parents=True)
    (args.output/'figure_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plot(data,args.output,args.style)
    (args.output/'README.md').write_text(captions(data,args.style),encoding='utf-8')
    zip_path = args.output.parent/('kite_experiment_figures_zoom_20261016.zip' if args.style=='paired-zoom' else 'kite_experiment_figures_20261016.zip')
    with ZipFile(zip_path,'x',ZIP_DEFLATED) as archive:
        for path in sorted(args.output.iterdir()):
            archive.write(path,arcname=path.name)
    print(json.dumps(dict(figures=str(args.output.resolve()),zip=str(zip_path.resolve()),
        zip_sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest()),ensure_ascii=False))


if __name__=='__main__':
    main()
