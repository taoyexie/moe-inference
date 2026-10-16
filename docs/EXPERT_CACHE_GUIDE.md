# Expert 物理缓存验收 v1（2026-10-16）

这是接回完整模型前的执行器验收，不是新的端到端性能实验。
它不加载完整 Qwen，只读取已经下载的三个真实 Expert 权重，以及旧 capture 中的真实输入。
沿用既有 Python / Torch / Transformers；不下载模型、不升级环境、不修改旧结果。

## 本次要验证什么

- GPU 缓存为真正预分配的 BF16 专家权重槽，默认两个槽，每个 9 MiB；不只是维护一个 LRU 字典。
- upload miss 把真实权重复制到空闲槽，必要时覆盖 LRU，然后在该槽运行 Qwen3MoeMLP。
- CPU miss 把 GPU 输入传到 CPU，执行 FP32 专家，结果转 BF16 传回；不准入 GPU 缓存。
- hit 直接使用已有 GPU 槽，不重传权重，并更新 LRU。
- 缩容释放相应权重张量引用、淘汰 LRU；扩容分配空槽。容量为零时，upload 使用单独声明的 scratch，绝不准入。
- 单独保留 9 MiB GPU scratch，供零容量上传和数值参考使用。不得把 scratch 隐藏在缓存预算内。
- 主存另持有三份 BF16 权重共 27 MiB、对应 FP32 CPU 副本共 54 MiB；I/O 与激活工作区另计。
- CUDA reserved bytes 是分配器保留内存，不等于真实专家驻留；日志分别记录预算、驻留 bytes、已分配槽 bytes 与 CUDA 峰值。

每次调用之后都与 GPU BF16 参考输出比较，限值与旧路径实验一致（relative L2 <= 0.02、normalized max <= 0.03）。
GPU 路径应一致，CPU FP32 路径可能有 BF16 舍入误差。出现超限立即失败并保留日志，不放宽阈值。

## 标定与受控场景

本次小验收任务只在启动时标定一次 Expert 的两条路径：第一个捕获专家，q=1/8/32，
每点每路径 3 次预热、5 次测量，路径顺序交错，保存全部数据。
upload 用常数、CPU 用含截距线性拟合；不借用旧任务参数，不挑选低 q 样本提高收益。
q 超出标定范围或线性预测无效时，保守走 upload 并注明原因，不做外推。
这是 Expert-only 标定，不是完整四路径工作负载标定；接入完整模型后要重新标定。
标定预热、参考检查和拟合的总时间另行保存。

三个专家对应捕获的第 0/23/47 层。调用顺序人为指定，每场景 12 次：
A、B、A、C、B、A、A 使用原生捕获 q，随后 C(q=1)、A(q=8)、B(q=32)、B(q=32)、C(q=8)。
更小 q 取前缀、更大 q 循环输入行，逐条注明来源。**不是真实模型新路由出来的访问流。**
场景包括 upload LRU、CPU bypass、固定成本选择、线性成本选择，以及显式 2→1→2→0→2 缩扩容验收。
固定与线性使用同一批启动样本，但这些仍只是验收场景，不是 WISP 或完整 KITE 消融。

为数值验收，每次调用后均上传参考权重，发生在调用计时外；它会改变硬件热状态。
所以本次计时只能用于实现诊断，不能用它宣称自然工作负载性能提升。
每次 invocation 从 GPU 输入就绪开始，到 GPU 输出就绪结束，显式同步；
包括查表、LRU、miss 路径操作，不包括输入准备、参考检查和落盘。
完整 job / scenario wall time 保留这些开销。不是 TTFT，也不是 exposed latency。

## 1. Windows PowerShell：上传

作用：只上传小源码包。`-i` 选择 SSH 密钥；`-P 2022` 为端口；不会传模型或启动 GPU。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:/moe inference/dist/kite-expert-cache-20261016-v1.zip" xty@10.214.243.19:/data2/xty/epkv/
```

## 2. 服务器容器：解压

作用：用现有虚拟环境 Python 的 zipfile 模块解压到新的版本目录；不改旧脚本、模型或环境。

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-expert-cache-20261016-v1.zip /workspace
```

## 3. 同一容器：运行

确认没有其他 GPU 作业，再执行。`--checkpoint` 为原权重；`--capture-dir` 为旧的三个真实专家输入；
`--capacity-experts 2` 为 GPU 缓存两个专家（18 MiB），另加已声明的 9 MiB scratch 与工作区。
程序会检查身份、环境和 GPU 忙闲，拒绝抢占他人的 GPU 作业。

```bash
/workspace/.venv/bin/python /workspace/kite-expert-cache-20261016-v1/scripts/expert_cache_smoke.py --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --capture-dir /workspace/results/expert_inputs/capture-20261015T021708Z-8cc0e95d --capacity-experts 2
```

预期：五行场景计数与 correctness PASS，随后 Results 目录。
首次访问权重可能需要等待磁盘读入，不能承诺固定耗时。若失败保留完整错误与目录。
`summary.json` 的 status=passed 才表示完成；events.jsonl 记录 60 次真实专家调用，
calibration.jsonl 记录 48 次启动计时（含预热），calibration.json 保存拟合及时间。
结果写入 `/workspace/results/expert_cache/expert-cache-...`，不会覆盖旧实验。

## 尚未完成

与完整 Qwen 稀疏层的连接、联合 Expert/KV 分配、KV 恢复/重放、真实混合 batch，
以及四组相同执行器的端到端消融仍未完成。不要把这个小验收写成这些功能的实测结果。
本机只有无 CUDA 的接口/状态测试；GPU 数值与字节实测需以上述服务器结果为准。
