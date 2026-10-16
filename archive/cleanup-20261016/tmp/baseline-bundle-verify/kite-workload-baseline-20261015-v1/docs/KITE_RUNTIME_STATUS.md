# KITE 工作负载级运行框架：当前实现与下一步

## 已实现（本机代码与测试）

`scripts/kite_controller.py` 为不依赖 Torch 的决策核心：

1. 每个工作负载新建一个 Controller；CostCalibration 接收本次启动阶段四条路径的实际计时样本，常数/仿射拟合后只冻结一次。
   它是采样接入接口，本身不会启动 GPU 测量；不偷偷用旧任务的成本冒充本任务标定。
2. 每个 batch 执行前只输入已排程的 P/D 请求数；执行后才能输入实际专家 ID、routed q 和 KV 对象需求。
   因而当前 batch 的实际路由不会泄漏到该 batch 的分配中。
3. 通过 Fenwick tree 维护完整访问 recency，持续更新复用距离。一次访问摊销 O(log N)，初版内存 O(N)；
   没有实现采样或压缩，长任务的内存控制仍为待办。
4. 周期性扫描候选 split，默认每 16 batches；重置统计窗口不重置访问栈，期间所有访问都保留。
5. Prefill/Decode 共用各资源的访问历史，不按 phase 分成两个缓存。
   直方图附带 routed q 标记，历史成本能够保留“大 q 与 miss 相关”的信息，不先将所有 q 平均。
6. 默认当前信号对 expert 的预算比例为 prefill 0.8、decode 0.2，融合当前信号权重 0.65，
   最后投影到最近的可行字节分配。均为显式可配置初值，未经性能调参验证，不称最优设置。
7. 候选按照 Expert 9 MiB、KV 全层 128-token 块 12 MiB 取整，记录 slack，预算内至少各保留一个对象。
8. KV 需求须按 request/forward 归为一个恢复组：restore 对该组的 miss 块逐块计费，replay 对同组整个前缀只计一次。
   RecoveryLedger 可将显式标注为 replay 内部的 Expert 操作吸收进宏操作总时间，防止再次收费。

历史 miss 是等尺寸对象、按访问准入、固定容量的 LRU reference stream 统计。
**不是**真实 CPU bypass/non-admission、请求结束失效和 resize 后状态的精确模拟。
`physical_split_applied=False` 固定写入决策结果；调用者不能将这些分配当作已实际改变 GPU 缓存。
KV 分组中的输入必须是正常推理需求，不能把 replay 内部的全套专家访问再重复加入同一个恢复账单。

本机对照：复用距离 MRC 与独立 OrderedDict LRU 在随机流、17 个容量和多个历史窗口逐项比较；
另测字节可行性、周期刷新、任务级标定冻结、非前视输入顺序、路径成本相关性与宏操作去重。
全部属于合成代码测试，不是模型或 GPU 性能证据。

## 已准备的真实性能入口

`scripts/workload_baseline.py` 测现有 Hugging Face + Accelerate CPU weight offload 后端。
加载一次模型，运行 6 条顺序请求，不安装专家 trace hooks，不在模型 forward 内写采集日志。
它不调用 Controller，不声称执行 KITE，不执行四路径启动标定。
首次真实运行由用户在服务器完成；本机仅通过计时算术和 fake-backend 生成控制测试。

详细操作及 TTFT/TPOT 边界见 `docs/WORKLOAD_BASELINE_GUIDE.md`。

## 未实现，不能声称已完成

- 同一加载进程里真实执行启动四路径标定，并将本任务样本送入 CostCalibration。
- 真正由 Controller 的 split 控制 Expert/KV 驻留字节的缓存执行器。
- CPU 计算绕过准入、KV 恢复/重放、扩缩容、请求结束释放、附带 KV 准入的实际执行语义。
- CPU/GPU 工作区、CPU FP32 专家副本、KV replay 临时完整前缀等的统一显存/主存预算。
- 原型下完整配对 KITE/baseline 端到端结果；当前不输出联合加速数字。

## 接下来的实施顺序

先拿到独立的真实性能基线，确认延迟统计与资源占用能正确运行；若当前 trace 采集仍在进行，应等它结束再启动。
并行意义上的开发顺序（不是启动第二个 GPU 任务）：接启动真实标定适配器，再接 Expert 有界缓存与两路径执行，
随后接 KV 块存储与宏重算执行器。每阶段都有真实 bytes、实际计时、数值一致性与失败日志。
最后同一请求序列、同一资源预算配对运行四组消融，报告启动标定、稳定期和包括启动的总耗时。
不把本机模拟、自身标定接口、LRU reference 成本代理或现有后端基线描述成完整 KITE 性能。
