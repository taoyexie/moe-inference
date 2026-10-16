# 下载数据复核：首层投影差异与路由放大

原始目录：`results/packed-diagnostic-20261016T112746Z-c8b95ded`，五个 JSON，完整下载后离线复核。
这不是性能实验；`within_original_tolerances=false` 原样保留。

## 文件与资源证据

readiness 的源码清单 SHA256 与已交付诊断 zip 的 bundle_manifest 完全匹配：
`9f040c07cd6321b6e89830fe72279db3c173cc183606f7dbf13aea1e1d7140ca`，31 个源文件。
模型加载无 missing/unexpected/mismatched keys 或 error messages。
资源监控 605 次采样，预期 GPU PID 8328，无干扰错误。模型加载 608.372 秒，
诊断程序总时间 636.613 秒；这是诊断开销，不作为推理性能指标。

## 数值记录

以下相对 L2 均来自原始 JSON，百分数仅为可读展示。

| 对全前缀参考的比较 | 合并 logits L2 | Decode logits L2 | Prefill logits L2 |
|---|---:|---:|---:|
| packed 全前缀逐请求 | 0 | 0 | 0 |
| 原生增量逐请求 | 3.133541% | 4.469847% | 0 |
| 分页增量逐请求 | 3.133541% | 4.469847% | 0 |
| 分页混合增量 | 4.816465% | 6.521819% | 2.124416% |

混合相对分页串行的直接比较为合并 5.793891%、decode 8.030231%、prefill 2.124416%，
不能将这组直接差异误写为 4.816465%-3.133541%。
五种变体的 argmax 均为 `[107684,106285]`；只说明这两个被检验 token 相同，不证明整段生成相同。

原生增量和分页增量相对全前缀的**整个比较记录**一致，包含 772 个阶段误差和 96 个路由统计。
这强烈支持该短例未出现额外分页误差，但日志只保留比较统计，未保存全部原始张量，
不能以两个相同 L2 标量或这些统计代替任意输入的缓存逐位正确性证明。

## 最有定位价值的边界

混合相对全前缀的第一层 decode：

- Q/K/V（送入 SDPA 的实际张量）及 SDPA 原始输出，误差均为零。
- 日志 `L00/attention/decode` 是含 o_proj 的整个 self_attn 模块输出，不是 SDPA 原始输出。
  此处 relative L2=0.0014226904604583979，max abs=0.0009765625。
- 固定 Transformers 4.51.3 源码在 SDPA 输出后执行 reshape/contiguous 与 o_proj，
  因此该短例的首个模块输出差异定位到投影计算边界，支持批矩阵形状数值因素。
- 第 2 层开始改变 decode 专家集合，48 层中 17 层该 token 的专家集合不同。

原生增量的最早差异则在第一层 v_proj，max abs 仅 `2.384185791015625e-07`；
第 11 层首次改变 decode 专家集合，最终 logits 差异明显更大。
这与初始微小数值差异经过 BF16 网络/离散路由传播放大的解释一致，
不证明每一层误差只由路由导致。

第一层 BF16 EFFICIENT_ATTENTION 对相同输入的 FP32 MATH oracle 相对 L2 为约 0.159%–0.175%，
混合为 0.166%。这是整个短 attention 的合并误差，不是输出质量或所有层的保证。

## 对执行与验证的影响

原生增量自己已超过旧的跨执行形态 .03 门禁，不能仅凭此门禁失败判定分页实现错误。
但混合的额外差异仍需处理或用明确的同形态独立参考验证，不能仅因 argmax 相同或“BF16 正常”跳过。
目前不放宽阈值、不启动性能重跑、不宣称 KITE 正确性或加速已经完成。

下一步轻量第一层探针：默认 BF16、关闭 BF16 reduced reduction、FP32 投影控制，
同一真实 token 和同一首层权重；不加载完整 Qwen。先确认最小设置是否改善投影差异，
再决定完整模型阶段的修复或验证方案。正式代码和实验条件暂时不改。

参考：[固定 Qwen attention 源码](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)、
[PyTorch 数值说明](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html)。
