# Qwen 真实推理首轮日志核验

原始记录：`D:\moe inference\results\qwen-20261015T011335Z-1993559b`。离线核验通过；没有重新运行模型。

Qwen3-30B-A3B，BF16；输入 20 tokens，输出 8 tokens。
生成内容：缓存是用于临时存储常用数据。

| 指标 | Prefill | Decode |
|---|---:|---:|
| Forward 次数 | 1 | 7 |
| 每次 forward 的非空专家调用数 | 2565 | 384 |
| 每次非空专家调用平均 routed tokens | 2.994 | 1.000 |
| 全阶段 routed-token assignments | 7680 | 2688 |

Assignment 指一个 token 被分配给一个专家；top-8 下一个 token 有 8 个 assignments。
Expert 标识是 (layer, expert)，跨层同编号不是同一个缓存对象。

共 5253 条非空专家记录；384 个层×forward 的路由守恒检查通过。
固定后端每轮还执行空输入专家调用，已单独计数，不将其记为逻辑专家需求；不能据此推算物理传输。

## KV 与权重放置

单专家 BF16 权重 9 MiB；全层 KV 每 token 96 KiB。
KV 从 20 增至 27 tokens，
逻辑张量占用从 1.87500 增至 2.53125 MiB。
全部 48 层 K/V 的形状、BF16 精度、GPU 位置与字节数通过核验。最后生成的 token 没有再次输入模型，故 KV 不包含它。

权重放置：19 层常驻 GPU，29 层存储在 CPU 并临时传入 GPU；不代表后者在 CPU 执行。
CPU 计划权重占用 34.239 GiB（非 RSS 实测）。
本次 PyTorch 已分配显存峰值 23.223 GiB；保留显存峰值 23.227 GiB。

## 已支持与尚未支持的结论

已支持：真实模型可以在当前 CPU/GPU 卸载环境生成输出；真实专家路由和 KV 观测链路可用。
在这一个请求中，Prefill 聚合多个 token，非空专家调用覆盖更多对象，每次调用的 token 数存在差异；Decode 的单请求单 token 使非空调用的 routed tokens 为 1。
这为按 routed-token 数标定专家 CPU 路径提供真实输入依据，但不能推出 Prefill 必然产生更多 miss，或某种显存划分已经更优。

尚未测量：四条 fallback 的完成时间、真实缓存 miss、KITE 分配收益、端到端速度改善。
计时含观测 hooks、首次执行和 CPU 权重卸载，只保留为诊断记录，不作为正式 TTFT/TPOT。

## 逐轮记录

| Step | Phase | Query tokens | 非空专家调用 | KV tokens | KV MiB | 诊断秒数 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | prefill | 20 | 2565 | 20 | 1.87500 | 106.225 |
| 1 | decode | 1 | 384 | 21 | 1.96875 | 9.170 |
| 2 | decode | 1 | 384 | 22 | 2.06250 | 8.571 |
| 3 | decode | 1 | 384 | 23 | 2.15625 | 8.465 |
| 4 | decode | 1 | 384 | 24 | 2.25000 | 8.298 |
| 5 | decode | 1 | 384 | 25 | 2.34375 | 9.265 |
| 6 | decode | 1 | 384 | 26 | 2.43750 | 9.306 |
| 7 | decode | 1 | 384 | 27 | 2.53125 | 9.602 |

## 可追溯性

模型来源 modelscope；分支 master（分支名本身不保证不可变性）。
下载时冻结的文件清单指纹：`1208a85873f530e25a62558a182fd28d8f95e594b24c0ae3cb3b609ddffcf6d8`。
本地核验 JSON 保存全部 10 个原始文件的 SHA256，用于追溯文件版本；不是第三方执行证明。
