# 真实 TTFT / TPOT / 请求完成时间：现有后端基线

这是第一次专门用于性能计时的入口，不是前面的带专家 hooks 的 trace 采集。
沿用真实 Qwen3-30B-A3B、BF16、SDPA、Accelerate CPU weight offload，模型只加载一次。
**它还不是 KITE**，不改变显存分配，不执行四路径成本标定。
若之前多请求 trace 正在跑，先等结束，避免自己的两个进程争用 GPU。

## 边界与设置

- 6 条受控顺序请求：输入 128/512/1024 tokens 各两条，系统和旅行主题，长度含 chat 模板。
  使用与多请求 trace 相同的 token 构造。不是自然请求集，也没有真实混合 batch。
- 每条最多输出 8 tokens，遇到 EOS 如实结束；不是固定输出 8 tokens。
- 默认第一条 128-token systems 请求作为 warmup，不纳入稳定期分布，但保留计时，并计入完整 workload wall time。
  Warmup 不是成本标定，日志固定 `cost_calibration_performed=False`。
- 模型加载、输入分词与请求列表准备发生在请求计时前。输入 token IDs 预先在 CPU；GPU 输入搬运计入请求。
- 请求起点：GPU 同步完成后，准备将 CPU input IDs 搬上 GPU。首次采样 token ID 到达 CPU 时为首 token 完成；
  最后采样 token ID 到达 CPU 时为请求完成。每次显式同步，包含模型内部真实权重搬运与 greedy 采样。
- TTFT = 起点到第一个 token 可用；TPOT = 第二个及以后 token 间隔的均值；ITL 列出每个间隔。
  只有一个 token 时 TPOT 为 null，不伪造 decode 延迟。
- 不包含 HTTP/队列/网络、tokenization、输出文本 decode 或生成后的 cache 清理。
  是本地预分词 batch=1 延迟，不能直接等同线上服务端 TTFT。
- 不安装专家采集 hooks，也不计时内写逐层 JSON；保留 Accelerate 正常卸载所需的执行机制。
  检测到额外注册的模块 forward hooks 会拒绝声称这是干净基线。
- 每条请求的 metadata/结果写入、进度打印与 GC 在请求延迟计时之外，但包含在 workload driver wall time。
- 分别记录：模型加载时间、带 warmup 的 loaded workload wall time、warmup 前缀时间、评价段 wall time、包含模型加载的 driver job wall time。
  另报扣除 driver 间隙的处理吞吐与包含间隙的顺序运行吞吐，两者都不是并发服务吞吐。
- 同时按输入长度分组，避免将五条不均匀长度的评价请求混成一个“普适平均值”。
- 24 GiB GPU / 48 GiB CPU 权重预算；8 CPU threads；NUMA1 的 CPU 24..31；GPU 忙时门禁拒绝。

## 1. 上传：Windows PowerShell

以下命令将小代码包上传到个人目录。`-i` 使用已有密钥，`-P 2022` 指定端口，
`IdentitiesOnly=yes` 限定 SSH 身份。不会传输模型或启动 GPU。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:/moe inference/dist/kite-workload-baseline-20261015-v1.zip" xty@10.214.243.19:/data2/xty/epkv/
```

## 2. 解压：服务器 Docker 容器内

以下命令用已有虚拟环境的 Python 解压（`-m zipfile -e`），只新增版本目录，
不更新 PyTorch、不动权重与旧代码、不覆盖结果。

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-workload-baseline-20261015-v1.zip /workspace
```

## 3. 性能试跑：同一 Docker 容器内

`--checkpoint` 指定已有模型；`--lengths` 自动为每个长度生成两条内容；`--new-tokens` 为输出上限；
`--warmup-requests 1` 表示第一条为预热，`--repeats 1` 表示本次运行一遍六条请求。
模型加载约数分钟，整个顺序运行可能需十几分钟，以实际输出为准；不保证每条请求相同耗时。

```bash
/workspace/.venv/bin/python /workspace/kite-workload-baseline-20261015-v1/scripts/workload_baseline.py --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --lengths 128,512,1024 --new-tokens 8 --warmup-requests 1 --repeats 1
```

预期每条请求输出 `TTFT=...ms completion=...ms output=...`，最终输出 `Results: ...`。
根目录在 `/workspace/results/workload_baseline/qwen-时间-编号`，对应数据盘
`/data2/xty/epkv/results/workload_baseline/...`。不会覆盖旧实验。
`request_metrics.jsonl` 是逐请求原始计时、token 时间点与输出，`performance.json` 是统计，
`workload.json` 保存精确输入及顺序；另有 checkpoint 身份、环境、placement 与 root summary。
只有 `summary.json` 与 `performance.json` 都正常结束才算本轮完成。

完成后发输出和 `Results:`，再按真实目录下载；若失败请保留完整报错与已有文件，不修改门禁或伪造样本。

## 下一项尚未完成

工作负载级的 CostCalibration、MRC、周期刷新、P/D 投影与宏计费核心已经有本机测试，
但未接通自动实测标定和实际缓存执行器。因此此入口只用于 baseline，绝不输出 KITE 加速。
后续 KITE 对照必须沿用相同输入、硬件预算与计时边界，并额外报告其真实启动标定开销。
