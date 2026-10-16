# 多请求真实路由采集

旧的 20-token Qwen 请求已经是真实模型 trace，并不是随机模拟。
随后 Expert/KV 的局部路径测试也真实执行搬运和计算，但控制了 q 或上下文长度，不能当成完整多请求访问轨迹。
本入口只是扩大真实请求采集范围：不是重新证明模型能运行，也不是已经运行 KITE 分配器。

## 本轮设置

- 同一已核验的 Qwen3-30B-A3B BF16 checkpoint，CPU weight offload，SDPA，禁止磁盘卸载。
- 沿用 24 GiB GPU / 48 GiB CPU 权重预算、CPU 8 threads、CPU affinity 24..31。
- 128/512/1024 输入 tokens 各 2 条（系统与旅行主题），每条最多输出 8 tokens，greedy、thinking 关闭。
- 输入为手写内容在 token 层重复/截断，保留 chat 开头和 assistant 后缀；保存精确 token IDs，长度包含模板。
  不是自然请求数据集，不能据此声称生产分布泛化或回答质量。
- 整个进程只加载一次模型；6 条请求顺序执行，每条独立 DynamicCache，无跨请求 KV prefix reuse。
- EOS 提前停止如实记录，不强行生成 8 tokens，不丢弃短输出。
- 专家模块 hooks 记录实际非空调用、层/专家 ID、routed tokens、请求 ID、Prefill/Decode。
  每层计数必须等于该轮 query tokens × top-8。
- KV 是每层真实 K/V 张量的长度、形状、精度、设备与 bytes，不是 GPU 物理 block read/miss 日志。
- hooks 与 CPU weight offload 下的 forward 时间只用于诊断，不能当无插桩 TTFT/TPOT 或端到端加速。
- 本轮是顺序请求，未实现真正混合 batch；未来交错重放必须明确标为构造调度。
- 不新增 hidden-state cache，也不执行 KV restore/recompute 微基准；用于后续 miss 分析。

## 上传（Windows PowerShell）

以下命令使用已有 SSH 密钥（`-i`）和服务器端口（`-P 2022`），上传新的小代码包。
`IdentitiesOnly=yes` 限定身份。只新增代码 zip，不复制权重、不运行 GPU、不删除旧文件。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:/moe inference/dist/kite-multi-trace-20261015-v1.zip" xty@10.214.243.19:/data2/xty/epkv/
```

## 解压（服务器 Docker 容器内）

调用已有虚拟环境的 Python，`-m zipfile -e` 表示使用内置模块解压。
只创建新版本目录，不更新环境或旧版本脚本，模型和结果仍保留。

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-multi-trace-20261015-v1.zip /workspace
```

## 采集（同一 Docker 容器内）

`--checkpoint` 指向已下载模型；`--lengths` 指定包含模板的输入长度，每个长度自动跑两个主题；
`--new-tokens` 是输出上限，不是保证输出数量。模型只加载一次。
启动前自动做资源门禁；GPU 忙时不要绕过。大致可能需十几分钟，按实际输出判断，不保证时长。

```bash
/workspace/.venv/bin/python /workspace/kite-multi-trace-20261015-v1/scripts/multi_trace.py --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --lengths 128,512,1024 --new-tokens 8
```

原始结果写入新的 `/workspace/results/multi_trace/qwen-时间-编号`，宿主机对应
`/data2/xty/epkv/results/multi_trace/qwen-时间-编号`。每条请求有独立子目录和唯一请求 ID。
根目录 `requests.json` 为请求索引与执行语义，`summary.json` 为总状态；每个子目录保存
`request.json`、`expert_trace.jsonl`、`routing_checks.jsonl`、`kv_snapshots.jsonl`、`steps.jsonl`、`summary.json`。
根目录用于加载的短 smoke prompt 不会额外执行；实际执行的只有索引里的 6 条请求。

完成后复制最后的 `Results: ...` 和输出给助手，再按具体目录下载结果；不猜目录名，不下载模型。

## 本机验证边界

本机无 CUDA/模型运行：单元测试验证 token 长度、模板保留、唯一请求 ID 与原有采集记账兼容性。
不把单元测试通过冒充新采集器已在 GPU 执行。旧单请求和路径微基准数据完整保留。
