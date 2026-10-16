# KITE 工作负载级运行框架：当前实现与下一步

## 当前执行优先级：v3 直接性能试跑

用户已明确停止新增精度探针，目标仅为完整 KITE 受控场景端到端收益。
v3 跳过跨形态精度门禁和额外诊断；启动四路径标定改为有限值检查，KV 拷贝仍须完整。
保留真实物理缓存、模型/router、同预算对照、资源隔离、OOM 和计时审计。
既有数值诊断完整保留，不据此声称质量已验证。141 项本地逻辑回归通过；服务器 v3 尚待运行。
步骤见 `docs/KITE_PERFORMANCE_RUN.md`，下述精度探针安排均为历史，不再执行。

## 最新状态：v2 完整端到端已运行并核验，当前三预算均无收益

2026-10-16 新入口 `scripts/kite_e2e.py`，操作见 `docs/KITE_E2E_GUIDE.md`。
正式交付已收敛为 baseline 与完整 KITE 两组，消融不再是必交付项。
完整模型混合 P/D、统一真实 frame 池、实际 Expert/KV 四路径、启动标定、历史＋当前批次决策均已接入代码。
独立 Controller 仍返回 decision-only 标志；Runtime 真正修改 frame 配额后才标记 physical_split_applied=True。
本机逻辑、NumPy 掩码/复制与审计检查不等于 GPU 数值验证。此前 45.57% 仍只是专家局部结果。
服务器 v1 短集成检查通过，但启动标定 replay relative L2=0.030144 超过 0.03，正式 workload 未开始。
v2 统一正常 forward / replay 的显式因果 mask、位置与缓存布局入口，仍保留独立串行参考和原误差阈值。
v2 已在服务器跑完六组 workload，全部启动 replay 检查误差为 0，原失败未再出现；没有 profiler，不能断言具体内核是唯一根因。
完整结果经源码哈希、时钟、请求/路由/缓存和资源监控核验，KITE 在三个预算下分别慢 15.98%/16.99%/23.96%。
见 `reports/kite_e2e_20261016_v2_seed17_audit/report.md` 与 `reports/kite_e2e_20261016_v2_seed17_diagnosis/report.md`。
诊断确认热标定/冷执行条件不匹配（CPU 调用约 93% 需转换）；Expert LRU reference 在当前预算范围零命中；
1/2 GiB 下 KV 全命中。原数据和负结果完整保留，尚未修改执行器、参数或启动新的 GPU 实验。
尚无完整 KITE 正加速可填写。下面历史记录中“未实现”的描述已由本段及新指南更新。

用户授权继续寻找有利场景后，新入口 `scripts/kite_advantage.py` 已准备：8 GiB 单预算、两组各一次，
长上下文突发使 KV 峰值约 3.89 GiB，运行指南 `docs/KITE_ADVANTAGE_GUIDE.md`。
这是第二轮探索，不删除上轮负结果。新执行器接入真实 CPU 冷状态固定项、每 x 中位数稳健标定，
两组共用同样规则；长上下文使用统一 EFFICIENT_ATTENTION，禁止静默 math fallback。
v1 已在 A100 通过小型高效注意力检查并完成模型加载，但启动资源监控时失败，未进入正式测量。
本次复核提供独立 `kite-advantage-20261016-v2`：CUDA 前只读 `check`、模型加载前监控、
稳健拟合顺序修复、KV 临时工作区生命周期修复；129 项本地回归通过。
场景与算法参数不变，新版 GPU 数值、峰值和收益仍待验证，见 `docs/KITE_READY_TO_RUN.md`。

后续服务器 v2 记录：资源与模型加载通过，进入短数值检查后 `upload,restore` relative L2=0.048166，
超过 .03；尚未启动正式标定和两组主测量。现提供 `numerical_probe.py` 独立短诊断，
详见 `docs/KITE_NUMERICAL_DIAGNOSIS.md`。不放宽原阈值，不修改正式策略，不重跑主负载；
先获取逐层证据再决定修复。当前本地回归为 133 项。

下载复核后：原生/分页增量的全部比较记录一致，混合 decode 的第一层 SDPA 原始输出对全前缀仍精确一致，
但经过 o_proj 后出现差异并传播到路由；五种变体的两个 next-token argmax 相同。
详见 `reports/kite_numerics_20261016_preparation/downloaded_findings.md`。
新增只加载首层真实权重与少量 embedding 行的精度探针，先检验 BF16 累加开关和 FP32 投影控制，
不修改正式执行器或阈值，不重跑完整性能负载；本地回归为 136 项，首层探针尚待 GPU 执行。

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
2026-10-16 用户已在服务器完成首次真实运行，六条请求全部成功，其中一条预热、五条评价。
下载目录 `results/qwen-20261016T003757Z-6ad1fee8` 已离线核验：逐 token 时间、请求顺序、
历史聚合和 workload SHA256 一致。平均 TTFT 9.834 秒、TPOT 9.293 秒、完成时间 74.886 秒。
第一条预热 TTFT 165.513 秒，原样保留；原因尚未定位。含模型加载的完整运行约 786.699 秒。
报告为 `reports/workload_baseline_20261016_audit/report.md`。
这是现有后端的单次顺序试跑，不是 WISP、不是真实 KITE 执行，也没有混合 batch 或连续资源隔离监控。
本机计时逻辑仍只有算术和 fake-backend 测试，真实性能来自上述服务器记录。

详细操作及 TTFT/TPOT 边界见 `docs/WORKLOAD_BASELINE_GUIDE.md`。

## 未实现，不能声称已完成

更新（2026-10-16）：`expert_cache.py` 与 `expert_cache_smoke.py` 已准备独立物理 Expert 缓存验收。
支持真实 GPU 槽、LRU upload、CPU bypass、缩扩容和两路径启动标定；本机通过状态/接口测试。
服务器已执行 2026-10-16 验收：60 次调用和 48 次启动计时数值检查均通过，
独立离线 LRU 重放与日志一致，见 `reports/expert_cache_20261016_audit/report.md`。
使用旧捕获输入和人为指定顺序，不是完整模型真实 trace；
参考检查会改变硬件热状态，不能拿该验收计时宣称性能收益。见 `docs/EXPERT_CACHE_GUIDE.md`。
完整模型执行器和下面列出的联合功能仍未完成。

为单独验证专家成本选择收益，已准备去掉逐调用参考干扰的配对场景，
见 `docs/EXPERT_CACHE_BENCH_GUIDE.md`：两个有利压力场景和一个稳定热点对照。
依据旧数据选择适用场景，协议在运行前固定。
2026-10-16 seed17 首轮已运行并核验：主异构 q 压力场景 85.948→46.775 ms，
平均配对耗时下降 45.57%；小 q 压力场景下降 29.54%；稳定热点约持平。
各 10 组配对、每 pass 192 次受控专家调用；详见 `reports/expert_cache_bench_20261016_seed17_audit/report.md`。
结果不含启动标定和槽初始化/释放，相关开销另外保存；不是 KITE/phase/WiSP/完整模型性能。
尚需独立进程复测，不能将单进程重复称作独立用户负载。

- 新端到端原型以上述功能的代码接通为当前状态；GPU 首跑数值、资源峰值及性能仍未验证。
- 原型完整配对 KITE/baseline 端到端结果尚缺；不得输出联合加速数字。

## 接下来的实施顺序

直接运行新端到端 search 入口：一次必要短完整模型检查，通过后自动完成三个预定义预算的两组比较。
保留所有预算结果，以包含启动标定/调整的实测工作负载总时间选择优势预算。
有正收益才冻结设置并更换提示词，做三个独立确认会话；否则停止，依据已有完整运行日志定位。
不继续扩充独立微基准，不要求每个组件单独都有收益，不将未测的阶段独立贡献写入文书。
不把本机模拟、自身标定接口、LRU reference 成本代理或现有后端基线描述成完整 KITE 性能。
