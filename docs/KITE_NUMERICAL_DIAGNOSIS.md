# 第二轮 packed/serial 数值失败：短诊断

2026-10-16 服务器 v2 在正式标定/测量前失败：
`upload,restore`，relative L2=0.048165880143642426，normalized max abs=0.04583333432674408。
原阈值仍为 .03/.05。不能将其当作性能负结果，也不能声称只是可忽略的 BF16 波动。
原结果目录：`/workspace/results/kite_advantage/kv-shock-search-20261016T110603Z-9c70f214`。

## 为什么需要这个诊断

四维 mask、position 与逻辑 KV 布局的本地检查不能保证实际 CUDA 数值一致。
不同批次形状、默认 causal 与显式 mask、缓存与全前缀之间都可能产生差异；
MoE top-k 路由还可能放大初始数值差异。以上是待区分的假设，不是已证实根因。
固定版本参考：[Qwen 模型](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)、
[SDPA 接口](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/integrations/sdpa_attention.py)。

诊断包不修改正式执行器、原阈值、分配器、模型精度或请求场景。
只用一次完整模型加载，运行极短的五种对照；不启动标定或 104 批主工作负载：

| 对照 | 作用 |
|---|---|
| 原生逐请求全前缀 | 原始参考 |
| packed 入口逐请求全前缀 | 区分显式 mask/补齐/缓存适配入口 |
| 原生 DynamicCache 逐请求增量 | 观察不经过分页器的增量误差 |
| packed 逐请求增量 restore | 检查分页读写与恢复 |
| packed 混合增量 restore | 检查混合后额外误差 |

所有对照使用相同的真实输入与由 seed forward 产生的同一个待消费 token；不是性能评测中的指定答案。
记录每层输入归一化、Q/K/V 投影、attention、router、MoE 和 layer output 的误差，
以及 top-8 专家集合改变的 token 数。第一层用相同实际 Q/K/V 与 mask 额外算短 FP32 MATH oracle，
只记录误差，不替换 EFFICIENT_ATTENTION 的实际输出。不对长上下文使用 math fallback。
诊断 hooks/同步开销明确存在，任何诊断耗时都不得作为加速证据。
逐对照写出部分 JSON，方便异常时保留已完成的证据。

## 命令：每条的含义

Windows PowerShell：上传独立诊断源码包，不修改模型和旧结果。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:/moe inference/dist/kite-numerics-20261016-v1.zip" xty@10.214.243.19:/data2/xty/epkv/
```

服务器已有容器 `/workspace`：解压诊断包，`-e` 表示解压；创建独立目录，不覆盖已运行的 v2。

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-numerics-20261016-v1.zip /workspace
```

同一容器且 GPU 空闲：运行一次短诊断；绝对 Python 路径使用现有环境，`--checkpoint` 指现有权重。
仍需要加载完整模型与 CPU masters，不能承诺比先前加载更快。内置资源检查和监控。

```bash
/workspace/.venv/bin/python /workspace/kite-numerics-20261016-v1/scripts/numerical_probe.py --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2
```

结束时 `diagnostic_complete` 仅表示诊断执行完成，不表示数值已通过。
读取 `within_original_tolerances` 和 `numerical_diagnostic.json` 才能判断差异。
Results 目录位于 `/workspace/results/kite_numerics/packed-diagnostic-...`。
之后依据误差首次出现的位置修实现，不能直接重跑主实验或放宽误差门禁。
