# Expert 优势场景与对照：配对重复测量 v1

## 目标与选择依据

用户希望突出设计的适用优势。本实验主动选择“token 数异构、专家缓存紧张、存在低复用访问”的场景，
验证按 routed q 建模的路径选择是否优于固定路径成本。此选择依据此前路径实测，不是盲测；
协议在本轮运行前固定，不能事后将它描述成随机工作负载的普适结论。
已有缓存验收显示固定与线性政策会产生不同准入和 miss；本轮测量这种差异的实际时间效果。

本轮不是完整 KITE，不测试 P/D 修正或 KV，也不是复刻/复现 WiSP。
两种政策共用同一个物理缓存执行器、两个 GPU 专家槽、同一输入和同一启动样本，
唯一变化是 miss 时的固定/仿射成本预测。CPU 路径不入缓存，upload 路径入缓存。

## 预定义场景

A/B/C 是原先捕获的三个真实专家（按层/专家排序）。每种模式重复 64 轮，每次测量 192 次调用。

| 场景 | 每轮调用 | 想验证的条件 |
|---|---|---|
| mixed_q_pressure（主优势场景） | A(q=32), B(q=1), C(q=1) | 缓存只能容纳两个专家；低 q CPU bypass 可能避免污染高 q 热点 |
| small_q_pressure（辅助优势场景） | A(q=1), B(q=1), C(q=1) | 三专家轮转使两个槽不足，低 q CPU 路径可能更便宜 |
| stable_hot_control（边界对照） | A(q=32), B(q=32), A(q=32) | 两个热点均可放入缓存，预计不需要靠 CPU bypass 获益 |

这些访问顺序和 q 是人为构造，输入来自真实捕获激活的前缀或循环重复，不是新完整模型路由的 trace。
不能将 q=32 写成真实 prefill、q=1 写成真实 decode，也不能据此声称 phase-aware 有收益。

## 测量协议

- 仅加载三个真实 Expert，GPU BF16、CPU FP32，与已通过验收的内核相同。
- 每个进程仅启动标定一次：第一专家 q=1/8/32，各路径 3 次预热、5 次测量。
  全部正式样本参与拟合：upload 常数、CPU 仿射；固定政策用相同样本的 CPU 常数均值。
- 不根据本轮计时重调阈值、不选择性删点；超范围预测默认 upload，日志标记。
- 正式计时前验证所有专家/q 的 CPU 与 GPU 输出；计时后再检查一次。
  正式 pass 内不做逐调用参考上传，不插入校验工作。
- 每场景每政策 1 次预热 pass 保留，10 次正式 pass，随机交错场景和政策顺序。
  seed=17 为首轮，后续 23/41 可作为独立进程重复，但仍不是独立硬件或自然用户请求。
- 每个 pass 都从空的两槽缓存开始；输入在 GPU 预备好，槽分配和释放另行记录。
  主指标 execution_wall_ms 包含整个专家调用循环、路径选择和缓存管理；不含输入准备、槽 setup/teardown。
- 同时保存 setup+execution+teardown、启动标定、预热及完整 job 时间，不能用短循环收益掩盖启动开销。
- 每个 case/trial 的两政策构成一对，报告所有配对有符号收益；负收益和 stable control 不删除。
  单进程重复不当作独立用户，先复测再下结论，不保证任何场景出现正收益。
- 所有场景严格两个 GPU 槽 18 MiB，另有 9 MiB scratch、输入/输出工作区；CPU 副本和峰值另留档。

## 上传（Windows PowerShell）

只上传代码，不修改环境/模型。`-i` 为原密钥，`-P` 为 SSH 端口。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:/moe inference/dist/kite-expert-cache-bench-20261016-v1.zip" xty@10.214.243.19:/data2/xty/epkv/
```

## 解压（服务器容器）

Python 自带 zipfile 解压至新目录，不覆盖旧版本和旧结果。

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-expert-cache-bench-20261016-v1.zip /workspace
```

## 运行（同一容器，GPU 空闲时）

`--checkpoint` 为现有权重，`--capture-dir` 为旧输入；`--cycles 64` 每次 192 调用；
`--repeats 10` 为 10 组正式配对；`--seed 17` 固定交错顺序。
不会再跑完整 Qwen，也不需要安装新依赖。资源门禁照常启用，不抢其他作业。

```bash
/workspace/.venv/bin/python /workspace/kite-expert-cache-bench-20261016-v1/scripts/expert_cache_bench.py --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --capture-dir /workspace/results/expert_inputs/capture-20261015T021708Z-8cc0e95d --cycles 64 --repeats 10 --seed 17
```

结束后返回 Results 路径。passes.jsonl 保留全部 66 个 pass（6 个预热，60 个正式）；
每场景第一组正式 pass 保存详细事件，其他 pass 保存真实路径与命中计数。
protocol.json 固定完整受控访问流；calibration.jsonl 为原始标定；correctness.json 为前后检查。
summary.json status=passed 表示结束。下载核验后再决定怎样表述结果。

## 可用于文书的结论边界

若复测确有收益，可写“在所构造的专家缓存压力与 token 数异构场景下，相同容量和执行器条件下，
基于 routed-token 数的代价选择相比固定代价选择降低了专家序列执行时间”。填写具体硬件、输入构造、
预算、计时边界和实测数字。不能写成完整模型 TTFT 提升、完整 KITE 提升或优于原始 WiSP。

后续阶段修正的优势场景是可预测的当前 P/D 组成变化与滞后的历史窗口之间存在偏差；
必须等混合 batch 和 Expert/KV 实际预算接通后再实测，不用本轮专家数字代替。
