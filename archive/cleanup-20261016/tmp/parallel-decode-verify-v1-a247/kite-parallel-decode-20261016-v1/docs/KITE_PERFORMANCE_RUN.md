# 回到端到端收益：v3 直接性能试跑

用户已明确停止新增精度探针，当前目标仅为找到完整 KITE 占优的受控场景并取得实测数据。
不再运行 `first_layer_probe.py` 或 `numerical_probe.py`，也不增加消融要求。

## 本版变化与不变项

- 不运行跨执行形态 packed/serial logits 门禁，不将未执行的精度验证标记为通过。
  `validation.json` 明确保存 `not_run_for_performance_trial`；已有数值诊断记录保留。
- 启动阶段只测本任务四路径成本，不再比较 CPU/GPU 或 replay/全前缀的相对数值误差。
  保留 NaN/Inf 停止、KV 拷贝完整性、物理缓存预算、路由守恒、GPU 干扰和 OOM 保护。
- 两组使用完全相同的上述口径。正式模型仍为原来的 BF16；不采用 FP32 投影探针配置。
- 8 GiB 共享池、每 16 批刷新、当前权重 .65、P/D 先验 .8/.2、8 GiB CPU FP32 上限均不变。
- 相同的 10 请求受控场景：8192-token 长背景与第 17/49/81 轮的三条 11264-token 突发。
  两组各运行一次，真实 router、真实自回归，不指定专家、不给基线强制不利分配。
- 主指标仍为包含本工作负载启动标定、分配调整、推理与搬运的实际完成时间；模型加载另列。
  成本拟合和 miss 仅用于做决策，不能拿它们计算一个假端到端耗时。

## 操作：每条命令的含义

Windows PowerShell：上传 v3 性能包，不覆盖旧包、旧结果或模型。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:/moe inference/dist/kite-advantage-20261016-v3.zip" xty@10.214.243.19:/data2/xty/epkv/
```

已有容器 `/workspace`：解压到独立 v3 目录；`-e` 表示解压，使用现有虚拟环境，不安装依赖。

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-advantage-20261016-v3.zip /workspace
```

同一容器、卡空闲：执行实际基线与完整 KITE 两组。`--checkpoint` 为已下载权重；seed17 固定执行顺序。
自动检查资源，加载模型一次，随后直接进入每组启动标定与完整推理，不再进入精度诊断。

```bash
/workspace/.venv/bin/python /workspace/kite-advantage-20261016-v3/scripts/kite_advantage.py search --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --seed 17
```

结果为 `/workspace/results/kite_advantage/kv-shock-search-...`。
运行完下载整个结果目录，核对实际时钟和缓存/请求记录，得到真实加速比。
如果整体占优，按同版冻结设置换预留主题确认，再把基线耗时、KITE 耗时和适用场景写入交底书实施例。
只展示整体效果，不声称已经证明 phase 修正的独立收益或无精度损失。

本版仍不能承诺一定胜出；若没有净收益，先看这一次完整运行的瓶颈，不用探针或旧局部微基准代替。
