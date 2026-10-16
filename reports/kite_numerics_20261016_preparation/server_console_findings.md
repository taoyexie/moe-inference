# 短数值诊断：服务器终端初步结果

来源：用户 2026-10-16 粘贴的输出；完整 JSON 尚未下载，以下不代替原始文件审计。
目录：`/workspace/results/kite_numerics/packed-diagnostic-20261016T112746Z-c8b95ded`。
运行约 11:27:50 至 11:38:22 UTC，诊断执行完成；没有性能测量。

下表均以原生逐请求全前缀推理为参考。数值由终端六位小数输出抄录，不冒充完整精度：

| 变体 | relative L2 | 最早非零阶段 | 最早专家集合变化 |
|---|---:|---|---|
| packed_full_serial | 0.000000 | None | None |
| stock_cached_serial | 0.031335 | L00/v_proj/decode | L11/router/decode |
| packed_cached_serial | 0.031335 | L00/v_proj/decode | L11/router/decode |
| packed_cached_mixed | 0.048165 | L00/attention/decode | L02/router/decode |

现有证据：显式入口的逐请求全前缀与原生逐请求全前缀在这次短例完全一致。
原生增量与分页增量的汇总误差相同，支持进一步检查批形状/数值因素，
但相同的 L2 标量不能证明两个输出张量完全相同，也不能排除全部分页错误。
原生增量自身也超过旧 .03 全前缀阈值，说明不能只用跨执行形态最终 logits 差异归因分页。
混合额外差异在第零层 attention 输出出现，路由更早变化；不能仅凭日志证明注意力内核缺陷或可忽略舍入。

下一步只下载现有 JSON：检查逐请求误差/argmax、混合相对串行的直接比较、
第一层实际 Q/K/V/attention output 差异，以及对相同输入执行的 FP32 MATH oracle 误差。
在这些证据到齐前不改正式执行器、不放宽阈值、不追加 GPU 重跑。

参考：[PyTorch 2.7 numerical accuracy](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html)。
该文档说明批量与切片浮点运算不保证逐位一致；这只能支持数值差异假设，
不能用来证明本实验已正确或 4.8% 最终相对 L2 可接受。
