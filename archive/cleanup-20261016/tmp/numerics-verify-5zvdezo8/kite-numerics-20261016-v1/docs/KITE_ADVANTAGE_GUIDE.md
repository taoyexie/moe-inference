# 第二轮有限探索：8 GiB 长上下文 KV 突发

目的：继续寻找完整 KITE 的真实端到端优势，不要求消融，不宣称已经得到收益。
旧 v2 三预算负结果及其审计保留。本版不伪造专家访问、不重放预定答案、不锁死基线分配。
只先测一个预算（8 GiB）、两组各一次，模型加载一次；结果不支持就分析日志，而非自动无限扫参。

## 为什么这个场景比上轮更有检验价值

旧真实访问流的最小有限专家复用距离为 380。8 GiB 池内专家的实际候选容量从 91 到 819 个，
覆盖了曲线开始下降的区间，不再所有候选都处于零命中段。
新场景的 KV 页数由提示词/输出长度和排程精确确定，不假装已经采集了新模型路由：

| 设置 | 数值 |
|---|---|
| 背景请求 | 第 0 轮进入，8192 输入，104 输出 |
| 突发到达 | 第 17、49、81 轮，各三条请求 |
| 每条突发请求 | 11264 输入，16 输出 |
| 突发内主题 | 三个不同主题；每请求独立位置、KV 和因果注意力 |
| 活跃上限 | 四条，仍不使用 chunked prefill |
| 工作负载 | 10 请求、104 批次、248 输出、109568 输入 tokens |
| KV CPU 页峰值 | 332 页，3984 MiB（约 3.89 GiB） |
| 同时存在的最大 dense KV payload | 约 3.86 GiB |
| 最大四维 BF16 mask payload | 约 2.65 GiB |

突发安排在历史每 16 批刷新后，主动构造“历史是较小 KV 工作集，下一批变为较大 KV 工作集”的情形。
如果历史选择 90% 专家份额，基线只留 68 个 KV 页（816 MiB）；同样历史提案下，
KITE 在纯 decode 批次投影到 50% 专家候选，留 341 个 KV 页（4092 MiB），可容纳这次峰值。
这是**条件性的容量推导，不是实测决策或加速预测**。历史策略也可能选择其他候选；代码不限制它。
GPU 内存实际峰值仍由服务器运行验证，上面三个 payload 不是完整峰值估计。

长上下文统一启用 PyTorch EFFICIENT_ATTENTION，并关闭静默 math fallback。
模型加载前执行一次很小的 BF16/mask 能力检查，不作为性能结果；后端不支持则立即停止。
实际 K/V 临时工作存储补齐到 8 列，补齐位置在注意力 mask 中全部屏蔽，不计为有效 tokens 或持久 KV。
正常 forward 与原始 token replay 使用同一规则，仍保留独立 serial 参考检查。
后端选择依据固定版本 [PyTorch 2.7.1 接口](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/nn/attention/__init__.py)
及 [SDPA 选择逻辑](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)。
两组都用这一后端，不把旧 v2 基线与新版 KITE 跨版本比较。

代价与风险：KITE 给 KV 更多容量会减少 Expert 容量，可能损失专家命中；历史也可能及时适应。
因此容量差异不等于一定加速，必须比较包含启动开销的真实时钟。

## 实现修正，不改四路径和 P/D 融合框架

- CPU 路径仍为仿射函数，但冷状态额外计入一个固定 FP32 权重准备项。
  实际选择前查询对应 CPU 副本是否存在；历史 q-only 评分保守按冷状态计算。
- 启动阶段从实际 128-token 模型 forward 的非空专家中选六个不同的、尚无 FP32 副本的专家，
  第一个作为预热，后五个测实际准备耗时，取最大值作为保守固定项。不是重复转换同一块已热主存权重。
  采样产生的 FP32 副本也占用 8 GiB 上限，所有工作计入本 workload 启动。
- Expert q=1/8/32，每路径一次预热、五次正式；KV context=128/512 仍每路径一次预热、两次正式。
  分别保留 48 条路径计时、30 次数值检查，以及六条 CPU 冷准备记录。
- 对每个 x 的正式样本先取中位数，再做常数均值或仿射拟合；两组相同，不事后删除某个长尾样本。
- 基线仍是固定 CPU/重算常数＋历史曲线；KITE 仍是仿射 CPU/重算＋历史/当前信号融合。
  两者都检查实际 CPU 冷状态、都支付冷准备代价，不拿实现遗漏为一方制造劣势。
- 历史每 16 批刷新、当前权重 .65、P/D Expert 先验 .8/.2、LRU reference 与实际 bypass 的区别都不变。
- 不对校准域外做外推。长上下文域外保守走 restore；整个前缀 replay 仍在支持的域内从原始 tokens 真算，
  并在启动阶段执行。本场景不预先声称正式执行中四条路径都被选择。

## 资源与公平性

同一 A100、BF16、完整 Qwen、CPU affinity 24–31 / 8 threads。
Expert/KV 共用真实 8 GiB 池，按 3 MiB frame 向下取整到 8190 MiB。
固定模型权重、临时 Expert/KV scratch、当前 batch dense KV 和注意力工作区另记账。
CPU Expert FP32 上限不变，为 8 GiB；CPU KV 页上限显式提高到 5 GiB，以容纳本次真实工作集。
模型加载前执行主存余量、GPU 空闲和源码完整性门禁。CUDA 小检查后、加载模型前就开始监控
GPU PID 和主存余量；一旦检测到干扰，在下一个安全检查点停止，不能强制绕过忙卡或 OOM。
轮询不是资源预约；模型分片加载中的阻塞操作不能保证立即中断。
最大 GPU allocated 必须保留 4 GiB 余量；不能把 payload 推导当作保证不 OOM 的证据。

两组在任务入口都清空缓存，各自执行同一启动流程；从启动产生的 CPU 副本不是免费预加载。
E2E 包含自己的启动标定、显存池初始化、调整、实际生成、搬运和写回；模型加载另列，程序总耗时保留。
主指标不是 miss×拟合成本，而是 perf_counter_ns 的真实时间差。

## 上传与运行

以下每条命令都只做所述操作，不安装依赖或重下模型。
本次就绪复核包是 `kite-advantage-20261016-v2`，不是旧的 `kite-e2e-20261016-v2`。
不改变第二轮的 8 GiB 场景、排程和算法参数。复核内容见 [就绪检查表](KITE_READY_TO_RUN.md)。

### 1. Windows PowerShell：上传代码

`scp` 复制源码包；`-i` 选已有密钥，`-P 2022` 选 SSH 端口。不会触碰模型和旧结果。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:/moe inference/dist/kite-advantage-20261016-v2.zip" xty@10.214.243.19:/data2/xty/epkv/
```

### 2. 服务器宿主机：进入已有容器

在 `xty@yq2:~$` 后执行。`docker exec` 进入正在运行的 `xty-epkv`；`--user` 使用自己的 UID/GID，
`--workdir` 将位置设为 `/workspace`。这条命令不创建新容器，也不运行 GPU 计算。

```bash
sudo docker exec -it --user "$(id -u):$(id -g)" --workdir /workspace xty-epkv bash
```

若提示容器未运行，先在宿主机执行下面这一条；它只启动已有容器，不启动模型。

```bash
sudo docker start xty-epkv
```

然后重新执行上面的进入命令。已经在 `/workspace` 容器内时，跳过这一步。

### 3. 服务器容器：解压到独立目录

使用现有虚拟环境 Python 的 zipfile 模块；`-e` 表示解压，最后的 `/workspace` 是目标父目录。
首次解压创建 `kite-advantage-20261016-v2`，不覆盖旧版本目录。不要手工修改包内源码。

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-advantage-20261016-v2.zip /workspace
```

### 4. 同一容器：只读预检（现在有人用卡也能执行）

`check` 核对包内文件哈希、软件版本、已核验模型文件记录、主存、数据盘、CPU affinity 和 GPU 进程。
不导入 PyTorch、不创建 CUDA 上下文、不加载模型、不占 GPU 显存；不重新哈希全部 57 GiB 权重。
使用虚拟环境 Python 的绝对路径，不需要先 `activate`。

```bash
/workspace/.venv/bin/python /workspace/kite-advantage-20261016-v2/scripts/kite_advantage.py check --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2
```

输出 `ready` 才表示该时刻静态条件通过；`blocked` 并列出他人 PID 是正常保护，等空闲再查。
`failed` 表示版本、文件或查询异常，需要先排查。退出码分别为 0、2、1。
这不是资源预约，也还没有验证长上下文 CUDA 正确性和峰值。

### 5. 同一容器：卡空闲后，只运行这一个场景的两组

`search` 执行当前冻结的探索场景；`--checkpoint` 指现有模型；`--seed 17` 固定两组随机执行顺序。
模型加载一次，必要短验证一次，然后两组各自完整标定和推理。不启动旧三预算扫描。
长提示词会增加显存、主存和运行时间；不能承诺固定耗时或保证胜出。

```bash
/workspace/.venv/bin/python /workspace/kite-advantage-20261016-v2/scripts/kite_advantage.py search --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --seed 17
```

输出 Results 在 `/workspace/results/kite_advantage/kv-shock-search-...`。
仅根 summary 与两组 performance 均 passed，且 audit_kite_advantage.py 通过后，才使用实际结果。
校准失败、干扰、OOM、负收益全部保留，不能用旧 Expert 局部结果代替。

加载期间即监控资源；失败记录保留 `readiness.json`、当前 `summary.json` 的 `stage`，
CUDA 后的监控启动记录另存 `gpu_monitor_startup.json`。若启动检查发现多个 PID，保留原始查询结果，
不再仅给出笼统的“单进程不符”。不要使用旧版本或关闭保护重试。

## 如果整体有效，再做确认

selection.json 必须为 selected；预算、代码与排程冻结，确认换未参与搜索的提示词。
随后独立会话 seed17/23/41，只比较这两组，不追加消融要求。

```bash
/workspace/.venv/bin/python /workspace/kite-advantage-20261016-v2/scripts/kite_advantage.py confirm --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --selection /workspace/results/kite_advantage/ACTUAL_SEARCH/selection.json --seed 17
```

含义：`confirm` 使用冻结选择和新主题；`--selection` 的 ACTUAL_SEARCH 必须替换为真实搜索目录。
仅有正探索结果时才运行，后两次把 seed 改为 23、41。每次仍包含自己的完整启动开销。

原 v2 的复核请使用保留的 v2 包源码。新版本源文件哈希已变，不应拿新执行器代码去通过旧结果的源码一致性检查。
