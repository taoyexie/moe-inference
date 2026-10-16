# Qwen 真实 KV 路径微基准（先小试跑）

本轮补齐 restore / recompute 两条路径；不是 KITE 完整分配器，不是服务吞吐测试。
沿用已下载并核验的 Qwen3-30B-A3B BF16 权重、PyTorch cu118 与现有依赖，不重新下载模型。

## 对比对象与计费边界

固定一个逻辑 KV 块：128 tokens，跨全部 48 层，batch=1，4 KV heads，head_dim=128，K/V 都用 BF16。
因此对象为 **12 MiB**，不是长度 L 的整段上下文。物理布局为
`[48, 2, 1, 4, 128, 128]`，连续存储。每个上下文 L 都恢复末尾 `[L-128, L)` 的同一块真实数据。

- Restore：预先存在的 pinned CPU KV 副本 → 已分配的 GPU 目标块，一次连续 H2D。
  不包含初次驱逐 D2H、CPU 内存申请、磁盘读取；这要求 CPU 副本已经存在。
- Recompute：从原始 token IDs 重放长度 L 的整个前缀，执行真实 Transformer，提取相同末尾块，
  丢弃临时前缀 KV 与隐藏状态。每次建立新 DynamicCache，不复用其他 KV 或隐藏状态。
  包含模型内部 CPU 卸载权重的上传、attention/MoE 计算、临时分配与尾块整理；没有单独加 Expert miss 费用。

两条路径终点都是同布局、同绝对位置的 GPU 块，不包含后续 decode。每次计时前将目标填成 NaN，
避免误用上次结果。恢复必须逐元素完全一致；重算固定阈值 relative L2 ≤ 0.03 且 normalized max error ≤ 0.05。
失败立即停止，保留日志，不放宽阈值凑通过。正确性检查不在计时区间。

**重要边界**：整前缀重放可能同时恢复多个缺失块，不能把这项开销乘上缺失块数；
一块 restore 也不能单独恢复其他已缺失的块。因此这是明确先决条件下的单对象恢复成本测量，
不是已经验证可直接相加的全局 KV miss 代价。本轮结果不足以证明重算严格线性、存在路径交叉或分配器有效。
所有上下文都使用同一份受控嵌套 token 前缀（中文文本分词后重复、截断），不是自然请求集合。
模型执行真实，但不将受控输入称为自然 workload。

## 实施选择

使用 `AutoModel` 加载同一 checkpoint 的真实 Qwen Transformer，不加载无需参与 KV 恢复的词表输出头。
检查所有 Transformer 参数完整加载，只有 `lm_head.weight` 允许未使用；禁止缺权重时随机补齐。
沿用 SDPA、BF16、24 GiB GPU 权重预算、48 GiB CPU 权重预算、8 CPU threads、NUMA1 的 CPU 24..31。
CPU 权重卸载到 GPU 执行，不是 CPU 上运行整个 Transformer。

模型每个进程只加载一次。时钟用同步 CPU wall-clock，包含提交开销；不讨论 exposed latency 或重叠。
输入 IDs/mask/positions 与目标块均在两条路径计时前放在 GPU。参考生成、打包 CPU 副本与写文件不计入 trial。
恢复用 pinned 连续块，是可达到的传输路径，不代表已有完整生产 paged-cache 管理器。

首次试跑：contexts=256,512，block=128，warmup=1，repeats=3，seed=17。
每种长度有参考生成 1 次、预热重算 1 次、正式重算 3 次，合计 10 次全前缀执行。
先看真实耗时与正确性再扩大到更多长度/重复；两点不足以验证线性拟合。
GPU 忙时会拒绝开始，不停止其他用户进程；快照不等于独占预约。

## 1. 上传：Windows PowerShell

下面命令把小代码包复制到你自己的 `/data2/xty/epkv`，不复制权重、不运行实验。
`-i` 指定已有 SSH 私钥文件，`IdentitiesOnly=yes` 只使用该身份，`-P 2022` 是 SSH 端口。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:\moe inference\dist\kite-kv-paths-20261015-v1.zip" xty@10.214.243.19:/data2/xty/epkv/
```

## 2. 解压：已进入的服务器 Docker 容器

下面命令调用现有虚拟环境的 Python，将 zip 解压到 `/workspace`，得到一个新的版本目录。
`-m zipfile` 使用 Python 自带解压工具，`-e` 表示解压。不要在 Windows 执行它。
仅新增该代码包文件，不安装依赖、不启动 GPU、不修改旧模型或结果。

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-kv-paths-20261015-v1.zip /workspace
```

## 3. 小试跑：同一 Docker 容器

下面命令实际加载一次模型并运行两种长度的 KV 微基准。模型加载约需数分钟，重算还可能需数分钟；
终端会逐步打印 reference / warmup / trial，尚未打印结束不代表卡死。
`--checkpoint` 只读取已核验的本地模型；`--contexts` 是前缀长度；`--block-tokens` 是每次缺失对象大小，
不是要生成的 token 数；`--warmup` 不纳入统计，`--repeats` 才纳入统计；`--seed` 固定路径与长度顺序的随机化。
其余资源参数沿用上文默认值，无需重新安装任何库。
结果写入 `/workspace/results/kv_paths/kv-paths-时间-随机编号`，即宿主机数据盘上的
`/data2/xty/epkv/results/kv_paths/...`；不会覆盖上次结果。

```bash
/workspace/.venv/bin/python /workspace/kite-kv-paths-20261015-v1/scripts/kv_paths.py --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --contexts 256,512 --block-tokens 128 --warmup 1 --repeats 3 --seed 17
```

正常结束打印 `Benchmark directory: ...`，`summary.json` 状态为 `passed`。
每个 context/path 正式样本 3 条，共 12 条正式计时，另留 4 条预热。
原始 `raw.jsonl` 保存顺序、计时、正确性；`summary.json` 保存统计、参数与模型身份；
`inputs.json` 保存精确 token IDs；每个长度保存真实 CPU KV 块 safetensors 及 SHA256（约 12 MiB/份）。
加载、门禁、KV shape 或数值检查异常时，请发完整输出，不手工修改门禁或删除失败记录。

## 4. 下载：试跑完成后在 Windows PowerShell

此处先不要复制占位符执行。把 `<本轮目录名>` 换成最终输出中的 `kv-paths-...`。
该命令 `-r` 递归复制本轮小结果目录到本机 `results`，不下载模型权重，也不删除服务器文件。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 -r xty@10.214.243.19:/data2/xty/epkv/results/kv_paths/<本轮目录名> "D:\moe inference\results\"
```

收到真实结果后再离线核验、决定正式扫描和拟合；当前本机单元测试仅验证协议/记账，绝不替代 GPU 实测。

## 固定版本实现依据

- [Transformers 4.51.3 Qwen3 MoE 实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)：真实 Transformer 与 DynamicCache。
- [同版本权重加载代码](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/modeling_utils.py)：从带 LM head 的 checkpoint 加载 base model 时处理 `model.` 前缀与未使用的 head。
