# 继续找整体优势：短提示重复突发

## 已完成：容量窗口假设未成立

两场完整 GPU 对照已下载并通过离线核验。32 GiB E2E 为 40.943852 → 38.340083 秒（1.067912×），
但主执行仅 14.041260 → 13.916896 秒；28 GiB 整体反而慢约 1.58%。
实际两组全程都选约 90% Expert：32 GiB 同为 3276 / 273 个 Expert / KV 槽位，28 GiB 同为 2867 / 239。
启动历史已经选中大 Expert 池，没有产生下文假设的容量差。停止重复这一方向，不进入该场景确认。
详见 `reports/kite_burst_search_20261016/scene_diagnosis.md`；下文保留为运行前协议和假设，不改写事前记录。

## 已有数据给出的方向

v3 正结果已下载并审计：E2E 204.422930 → 200.604298 秒，1.019036×。
主执行 173.928910 → 172.442572 秒，只降低约 0.85%；启动/初始化贡献另约 2.33 秒。
从 background 请求的真实逐 token 时间点重建各批耗时：四请求 decode 共省约 6.32 秒，
单请求 decode 却多花约 5.48 秒，抵消收益；长 prefill 本身耗时约 76 秒，进一步稀释。
不能把 KV miss 降幅当成相同幅度的端到端加速。

现有另一条真实 Qwen trace 的 20-token prefill 访问了 2565 个不同专家。
对应约 22.54 GiB BF16 权重。它提示一个容量拐点，并不保证换执行器后的路由数完全相同。
在 32 GiB 总池、历史提案为一半时，基线分给专家 1820 个槽，而 KITE 当前全 P 信号投影到 2730 个槽。
如果重复请求的工作集落在两者之间，KITE 可以驻留，基线则在曲线刷新前反复换入。
代码不强制基线选择一半，实际提案、miss 与分配都留档；猜测不成立也保留完整结果。

## 固定两组、最多两个场景

| 场景 | 共享池 | 重复短提示 |
|---|---:|---|
| repeat-20 | 32 GiB | 请用一句话解释什么是缓存。 |
| repeat-short | 28 GiB | 解释缓存。 |

每场景为一次完整独立 workload：第 0 轮一条 512-token 实际请求，随后第 1–15 轮各一条短提示请求。
每条生成一个真实 token，完整运行模型/router；每请求位置与 KV 独立，不使用答案缓存或前缀缓存。
512-token 首请求的正文重复后置于正常 chat 边界内，保证既有 128/512 启动标定有本任务输入可用。
重复短提示使用完整真实 chat template，不截断，不指定专家。实际 token IDs 和输出都记录。

这是 **prefill 主导、首 token 完成、重复短请求突发**，不是混合 decode 场景，不能声称覆盖所有负载。
选择短突发是为了检验当前 phase 信号在历史首次刷新前的响应，不人为延缓或禁用基线刷新。
仍为每 16 批刷新、.65 当前权重、P/D .8/.2，CPU cache 8 GiB，四路径与执行器不变。
只调整请求负载和可分页预算，不修改 KITE 算法，也不增加消融/精度探针。
这个同一 phase 的 workload 内分配可以保持不变，审计仍检查每批真实应用的分配公式和物理容量。

首场景完整两组 E2E 达到 1.10× 时按预定规则停止搜索；否则再跑第二场景两组一次。
模型只加载一次，最多四次实际 workload。所有结果保留，不用 trace 估计时间代替实测。
每组启动标定、池初始化、推理、搬运与在线统计照常计入 E2E。GPU 最大 allocated 保留 4 GiB 余量。
较大的池只用于短输入的小工作区，不把 32 GiB 池套回上一轮长上下文大工作区。
OOM、忙卡或资源异常停止，不强行继续。

## 命令及含义

Windows PowerShell：上传新场景包；不重装环境、不重下载模型、不覆盖旧实验。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:/moe inference/dist/kite-burst-search-20261016-v1.zip" xty@10.214.243.19:/data2/xty/epkv/
```

服务器已有容器 `/workspace`：用现有 Python 解压到独立目录，`-e` 表示解压。

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-burst-search-20261016-v1.zip /workspace
```

同一容器、GPU 空闲：`search` 按上面的有限规则执行完整基线/KITE 对照；checkpoint 为已有权重。
首次 baseline 在前，若执行第二场则交换顺序。不运行此前的长上下文场景或任何精度探针。

```bash
/workspace/.venv/bin/python /workspace/kite-burst-search-20261016-v1/scripts/kite_burst_search.py search --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --seed 17
```

结果在 `/workspace/results/kite_burst_search/prefill-burst-search-...`。
先下载并审计；整体占优后冻结所选场景，换预留的“内存”提示进行独立整体确认。
确认种子 17/23/41，23 与 17/41 的两组顺序相反；三次不是完全平衡的两种顺序，必须如实记录。
不单独声称 fallback/phase 各自贡献，不将首 token 结果称为长生成加速。
