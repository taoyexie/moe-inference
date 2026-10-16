# 完整 KITE 受控端到端原型 v1（2026-10-16）

本版交付真正接通的代码路径：完整 Qwen、实际混合 P/D forward、统一 GPU Expert/KV 池、
四条恢复路径、工作负载启动标定及在线分配。**代码已实现不等于 GPU 验证或性能已经通过。**
本机无 Torch/CUDA，只有逻辑、调度、NumPy 张量复制/掩码和审计检查；首次服务器运行是必要的集成验证。
不存在预填的 KITE 加速数字。旧专家 45.57% 不作为本版端到端结果。

适配依据为固定版本 [Qwen 模型实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)
与 [Cache 接口](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/cache_utils.py)，不修改已安装的 Transformers 文件。

## 正式对照和完成标准

仅 baseline（固定成本、历史分配）与 kite（路径成本、历史＋P/D 修正）两组。
同执行器、相同容量、路径集合、精度、初始状态、请求和生成规则。
不以每个组件都有独立收益作为完成条件，不强制消融；没有组件证据就不作独立收益表述。
基线不是原始 WiSP。无 HTTP 服务、无真实线上到达声明。

## 模型与真实执行

- 固定已有 Qwen3-30B-A3B、checkpoint identity、Torch 2.7.1+cu118、Transformers 4.51.3。
- 完整 BF16 checkpoint 先在 CPU 加载，Expert masters 逐张量克隆为主存实体，不依赖 checkpoint demand paging。
  固定 attention/router/embedding/norm/head 移到 GPU，不使用旧 Accelerate 整层卸载 hooks。
- Packed batch 保留每请求自己的位置与 KV，四维因果 mask 阻断跨请求注意力。
  只对每请求当前最后 query token 计算 logits；router、非空专家与实际 q 均来自真实模型。
- GPU 池预分配 3 MiB frames：专家三个、128-token 全层 KV 四个。
  调整容量真实释放所有权、淘汰并复用 frames，不重复初始化权重、不伪造准入。
- GPU Expert hit 直接计算；upload miss H2D 真正权重并准入；CPU miss FP32 计算回传但不准入。
  CPU FP32 专家副本最多 8 GiB，冷转换计入真实路径，并记录命中/转换/淘汰。
- KV CPU 副本是带版本的主存真值。restore 按实际 12 MiB 页复制；replay 用原始及已生成 token
  重建整个已消费前缀。重放最多 2080 tokens，同请求/前缀只恢复一次；内部专家统计单列，主时间不重复相加。
- KV 新写入及恢复统一更新版本；请求完成后释放 KV。工作区仅服务当前 forward，不跨步复用未准入副本。
  原始 CPU token 前缀不包含尚未送回模型的最后采样 token。
- 历史 MRC 为 admit-on-access LRU reference，保留真实 routed q；不宣称精确模拟 CPU bypass、失效及扩缩容。
  实际 hit/miss、P/D 比例和真正应用的容量独立记录。

## 内存和安全

主存要求：完整 BF16 checkpoint 大小＋8 GiB CPU 副本上限＋9 GiB KV/加载/安全余量。
当前约需 74 GiB MemAvailable；以启动时实际检查为准。不要强行绕过失败门禁。
CPU affinity 24–31、8 threads。模型加载和 masters 主存实体化可能需要几分钟。

GPU 可分页预算只搜索 512/1024/2048 MiB，是人为设置的缓存压力，**不是声称 A100 40 GiB 被自然用满**。
实际池容量向下取整至 3 MiB。池外明确保留：固定模型权重、9 MiB Expert scratch、12 MiB KV scratch、
当前批次 KV/注意力工作区、全前缀 replay 临时结果；记录 CUDA allocated/reserved 峰值。
主存还包括 9 MiB pinned upload staging、至多 1 GiB CPU KV 及临时输入/输出。
GPU 峰值必须保留至少 4 GiB 余量。CPU 缓存的逻辑上限不包含瞬时算子工作区。

启动前要求 GPU 空闲；加载后每秒监控 GPU PID 和主存余量，其他 GPU 作业进入或主存不足则终止自己的程序，
从不结束他人进程。轮询不能证明亚秒级严格独占，原始监控留档。

## 工作负载与端到端计时

主请求第 0 轮进入，2048 输入、32 输出；第 9/17/25 轮各加入三条 512 输入、8 输出请求。
最多四个活动请求，无 chunked prefill；总计 10 请求、104 输出、33 批次。
混合批次必须在实际模型执行里出现，而不是串行执行后贴 P/D 标签。
原始 prompt 来自留档手写文本，经重复截断到长度，非自然数据集。
同一规则抑制 EOS，真实 greedy 自回归生成固定数量 token，无 teacher forcing。
CPU FP32 和 GPU BF16 输出允许声明范围内的数值差异；不承诺各策略生成 token bit-identical。

主计时从模型已加载后的工作负载提交开始，包含池初始化、启动标定、分配调整、
完整推理、KV 写回和结束释放。第 0 轮请求 TTFT 包含启动；后续请求按逻辑到达轮次计时。
另保存纯评价段、TTFT、TPOT、实际 token 时间点和包含模型加载的整个比较程序时间。
分词在计时前，文件序列化在任务结束后；每轮 Python 控制及在线历史开销不能扣除。
各模式之前清空前一模式的缓存是实验 harness reset，单独记录且不计入下一模式，
避免把前一策略留下的 CPU 副本释放成本不公平地归给后一策略。新任务自身的池初始化照常计入。

每个 workload 独立进行实际启动标定：Expert q=1/8/32，每路径一次预热、三次正式；
KV context=128/512，每路径一次预热、两次正式。保存全部 36 次计时及 24 次误差检查。
固定成本取本次正式样本均值；路径模型用常数/仿射。不复用旧任务系数，不裁负斜率；
越界或无效预测保守走 upload/restore。MRC 每 16 批刷新，current 权重 .65，P/D Expert 先验 .8/.2。

## 一次操作直接进入完整端到端搜索

### 1. Windows PowerShell：上传

作用：上传源码包，不传模型、不安装依赖。`-i` 选择现有密钥，`-P 2022` 指定 SSH 端口。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:/moe inference/dist/kite-e2e-20261016-v1.zip" xty@10.214.243.19:/data2/xty/epkv/
```

### 2. 服务器容器：解压

作用：用现有虚拟环境 Python 解压至新版本目录；不覆盖旧实验、模型或环境。

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-e2e-20261016-v1.zip /workspace
```

### 3. 同一容器：执行搜索

作用：完整模型只加载一次；先自动通过一次必要的短混合批次/串行参考与恢复检查，
成功后自动测三个预定义预算、每预算两种配置，共六个完整 workload。
检查失败就停止，不继续生成性能表。`search` 不运行以前的独立专家微基准。
`--checkpoint` 是已有模型；`--seed 17` 固定两种配置及预算的随机执行顺序。
这会占用真实 GPU 和较多主存，请先与其他使用者协调空闲窗口。无法承诺固定耗时。

```bash
/workspace/.venv/bin/python /workspace/kite-e2e-20261016-v1/scripts/kite_e2e.py search --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --seed 17
```

输出每组 `E2E=...s (startup included)`，最终 Results 路径。
只有根 summary.json status=passed、两个模式的 performance.json 均 passed 才能使用结果。
失败保留 error.txt、已完成任务和 partial_* 记录，不修改门禁、不以成本估计填补失败项。

## 搜索之后

所有三个预算（含负收益）保留。若无净收益，selection.json 为 no_positive_result，停止并分析完整日志。
有收益则 selection.json 冻结最优预算、模型与源码 hash，不再改变协议。
确认阶段换为科学/数学/烹饪/体育主题，用同一新版本、原选择文件分别启动 seed17/23/41 三个进程。
命令的 selection 路径须使用本次真实 Results，不能提前猜测：

```bash
/workspace/.venv/bin/python /workspace/kite-e2e-20261016-v1/scripts/kite_e2e.py confirm --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --selection /workspace/results/kite_e2e/SEARCH_RESULT/selection.json --seed 17
```

含义：`confirm` 使用冻结预算和未用于搜索的提示词；`--selection` 必须替换成真实选择文件；
`--seed` 后续改成 23、41。复用同源码/模型已经通过的集成检查，不再重复独立功能实验。
每个确认进程仍从本工作负载启动标定开始真实计时。独立进程不等于独立用户或硬件。

下载原始目录后，audit_kite_e2e.py 核对原始时钟、请求、路由、容量、标定和净加速。
若只整体有效，交底书仅表述整体实测效果及条件，不主张阶段修正独立有效。

## 结果文件

根目录 summary.json / workload.json / validation.json / resource_monitor.json / selection.json（搜索时）。
每任务 performance.json、requests.json、calibration.json、batches.jsonl、recoveries.json；
失败任务保留 partial_*。主时间直接来自 perf_counter_ns，不由 miss×成本合成。
原始专家局部收益和任何历史模拟数据都不混入本版端到端结论。
