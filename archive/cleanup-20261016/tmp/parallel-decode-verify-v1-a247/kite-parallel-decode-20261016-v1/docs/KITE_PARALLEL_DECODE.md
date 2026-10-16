# 下一轮：四请求短输出的 KV 容量窗口

## 为什么不继续上一轮

短提示突发的两组实际配额完全相同，基线也驻留了全部热点，无法体现当前信号的分配收益。
更早的 8 GiB 长上下文实测中，四请求 decode 的批间区间合计少约 6.32 秒，
单请求 decode 却多花约 5.48 秒，抵消了大部分收益。下一轮不再保留这些单请求空档。

先在本机对既有两组各 45 批四请求 decode 的真实访问做了**条件性参考 LRU 重放**：
若历史提案为一半，6/7 GiB 时当前修正跨过 KV 容量临界点；两个来源均少 13818 次参考 KV miss，
额外参考 Expert miss 为零。8 GiB 及以上在该条件下参考 KV miss 差为零。
证据为 `reports/kite_parallel_decode_20261016/capacity_screen.json`，生成入口为 `screen_kite_parallel_decode.py`。
这是固定候选容量、每段空 reference cache、不模拟 CPU bypass 的容量敏感性分析，
不是新场景的实际 miss、路由预测、硬件计时或端到端加速，不能填入实测性能表。

## 固定下一轮实验

沿用同一执行器、Qwen BF16、四路径、8 GiB CPU FP32 cache、每 16 批刷新、.65 融合和 .8/.2 P/D 先验。
不改模型、成本模型、fallback、分配公式或基线策略。只改请求到达和可分页预算。

- 四条请求同时到达，输入长度为 8192 / 11264 / 11264 / 11264，每条生成 16 个真实 tokens。
- 第 0 批四条 prefill 合并运行，后续 15 批均为四请求 decode，没有单请求 decode。
- 每条请求独立位置、KV 和注意力范围；使用留档正文重复和截断到精确长度，不指定专家或生成 token。
- 不 chunk prefill、不重用答案/前缀缓存、不删除 prefill 时间。请求总数 4，输出总数 64，KV 峰值 332 页（3984 MiB）。
- 最大显式 mask 约 3.28 GiB；主机 KV 上限仍为 5 GiB。保留高效注意力、4 GiB GPU 安全余量、资源/OOM 保护。
- 仅运行 6 GiB、7 GiB 两个预算，每个预算固定成本基线和完整 KITE 各一次，模型只加载一次，最多四次完整 workload。
- 6 GiB 若历史为一半，baseline KV 256 页、KITE decode KV 384 页；7 GiB 对应 298 / 448 页。
  332 页介于两者之间，但**真实历史提案不被固定**。若启动历史不满足条件，照常保留实际运行结果。
- seed17/41：6 GiB 基线先，7 GiB KITE 先；seed23 全部反向。不是把同一组永远放在后面。

本场景刻意检验一个历史刷新区间内的短输出突发响应，不声称长期稳态或所有上下文的效果。
无限延长 decode 可能让基线追上，不能把前 16 批收益外推为无限收益。
两个预算都完整运行，不根据中间速度继续扫参，也不追加消融或精度探针。

主指标仍为每个完整 workload 的实际完成时间，包含该组标定、初始化、分配、完整推理、拷贝、写回和过期释放。
加载另列，包含加载的整个程序时间仍保存。正式生成不插逐调用参考重算。
选择确认场景时，同时要求 E2E 和主执行均为正收益、实际 KV restore 页数减少；不能只因启动更快就选中。
所有预算、负结果、时钟、真实分配、路由及 miss 都保留。不是原始 WiSP，不能给各模块单独归因。
占优后冻结场景，用预留的新主题运行 seed17/23/41 三个独立会话；探索不能冒充确认结果。

## 如何运行，每条命令的含义

1. **Windows PowerShell**：把新测试 zip 上传至自己的服务器数据目录。不重装、不覆盖旧包。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:/moe inference/dist/kite-parallel-decode-20261016-v1.zip" xty@10.214.243.19:/data2/xty/epkv/
```

2. **现有 Docker 容器 `/workspace`**：用已有 Python 解压，`-e` 是解压到后面的目录，生成独立版本文件夹。

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-parallel-decode-20261016-v1.zip /workspace
```

3. **同一容器，GPU 空闲**：先做不加载模型、不初始化 CUDA 的源包、资源和环境检查；忙卡会拒绝开跑。

```bash
/workspace/.venv/bin/python /workspace/kite-parallel-decode-20261016-v1/scripts/kite_parallel_decode.py check --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2
```

4. **同一容器**：`search` 加载已有真实模型一次，执行上面的两预算完整配对；`seed17` 固定初始种子与对照顺序。
每组自动重新标定并计入 E2E，不运行旧场景、不添加精度探针。加载还要约十分钟，不要重复启动第二个进程。

```bash
/workspace/.venv/bin/python /workspace/kite-parallel-decode-20261016-v1/scripts/kite_parallel_decode.py search --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --seed 17
```

输出目录 `/workspace/results/kite_parallel_decode/parallel-decode-search-...`。
终端会并列打印每组 E2E、主执行和实际 KV restore 页数，不再只看一个加速比。
完成后下载整个结果文件夹，再用 `audit_kite_parallel_decode.py` 核验，不填预测耗时代替实测。
