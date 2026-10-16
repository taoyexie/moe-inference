# Qwen 首次真实访问采集

这一轮只验证真实模型可以生成、Expert 调用可观测、KV 可以从 Prefill 延续到 Decode。
不是 KITE allocator，也不是四路径性能实验。

## 执行边界

- 本机 Windows：准备和上传代码，不安装 CUDA 或模型权重。
- 服务器宿主机 `xty@yq2`：检查硬件、进入现有容器。
- 容器 `/workspace`：使用原 `.venv` 安装额外依赖、下载模型、运行脚本。
- 新试跑包解压后为 `/workspace/kite-qwen-smoke-20261015-v3`，不覆盖 v1/v2 或旧实验代码。
- 模型和结果显式放 `/workspace/models`、`/workspace/results`，对应数据盘。
- 脚本不删除文件、不终止其他任务、不更改系统驱动、不启动服务。

## 一 进入现有容器和环境

在服务器宿主机执行：

```bash
sudo docker exec -it --user "$(id -u):$(id -g)" --workdir /workspace xty-epkv bash
```

含义：进入已经运行的 `xty-epkv` 容器；`-it` 提供交互终端，`--user` 用当前用户的
数字 UID/GID，避免在数据盘创建 root 所有的文件；`--workdir` 设置起始目录。
不新建、不重启容器。`I have no name!` 是容器缺少用户名映射，不代表失败。
若提示容器未运行，先联系本次指导，不要删除或重建它。

在容器执行：

```bash
source /workspace/.venv/bin/activate
```

含义：让当前终端的 `python` 和 `pip` 指向原虚拟环境。不安装软件、不使用 GPU。
提示符应出现 `(.venv)`。

```bash
cd /workspace/kite-qwen-smoke-20261015-v3
```

含义：进入已上传的新代码目录，仅改变当前终端所在位置。

## 二 先检查 再安装

```bash
python scripts/qwen_smoke.py check --path /workspace
```

含义：读取 GPU、主存、容器可见内存限制、CPU 配额和数据盘空间，打印 JSON；
`--path` 指定检查哪块磁盘。不导入 torch、不分配 GPU 显存、不下载模型。
这只是资源快照，不是资源预留；主机上不可见的 cgroup 限制仍可能存在。

v3 修复了首次实际加载时，在 CUDA 尚未显式初始化的情况下重置显存峰值统计的问题。
使用 v3 代码不需要重新安装依赖或重新下载 checkpoint。可以先在新代码目录执行：

```bash
python scripts/qwen_smoke.py cuda-check --path /workspace
```

含义：检查资源后初始化 CUDA，验证显存统计接口，并创建一个只有一个数值的 GPU 张量。
不加载模型、不下载文件；会短暂创建 CUDA 上下文，有少量显存占用，进程结束后释放。
预期 JSON 中 `status` 为 `passed`、`cuda_initialized` 为 true、`probe_value` 为 1.0。
这是设备接口检查，不是模型推理或性能实验。

```bash
python -m pip install -r requirements-qwen.txt
```

含义：将文件内的固定版 Transformers、Accelerate、Hub 等依赖装到当前 `.venv`。
`-r` 表示按文件安装；不指定升级 PyTorch，不安装 CUDA 驱动。
实际依赖下载仍需网络。已验证环境应保留 `torch==2.7.1+cu118`；脚本会严格核对。
模型权重尚未下载，GPU 尚未使用。

```bash
python -m unittest discover -s tests -v
```

含义：执行轻量记账测试；`-s tests` 指定测试目录，`-v` 显示每项结果。
没有模型、没有 GPU、没有网络。检测漏层、路由计数、KV 字节数和资源门禁。

```bash
python scripts/qwen_smoke.py run --tiny --new-tokens 3 --results-dir /workspace/results/qwen_smoke
```

含义：用随机初始化的极小 Qwen 架构在 CPU 验证实际 hook 与缓存接口，生成最多 3 个
token，不下载任何权重、不使用 GPU。结果带 `is_tiny_synthetic_test=true`，
不能作为真实模型或专利实测数据。通过后再进行大模型下载。

## 三 下载模型 不进行推理

### 国内官方 ModelScope 渠道

Hugging Face 无法连通时，可使用 Qwen 官方提供的 ModelScope 仓库。
在新代码目录中先安装可选下载器：

```bash
/workspace/.venv/bin/python -m pip install -r requirements-modelscope.txt
```

含义：在原虚拟环境安装 ModelScope 1.25.0；不安装框架 extras，不指定升级 PyTorch，
不下载模型、不使用 GPU。`-r` 表示读取依赖清单。

```bash
/workspace/.venv/bin/python scripts/qwen_smoke.py prepare --source modelscope --models-dir /workspace/models
```

含义：访问官方 ModelScope API，下载少量配置、tokenizer 和索引，校验 SHA256。
`--source modelscope` 切换下载来源；`--models-dir` 指定数据盘位置。
不联系 Hugging Face、不下载完整权重、不使用 GPU。
返回的路径含 `modelscope-` 和文件清单指纹，不能把它误写为 HF commit SHA。

上一步成功并确认所需空间后执行：

```bash
/workspace/.venv/bin/python scripts/qwen_smoke.py prepare --source modelscope --models-dir /workspace/models --weights
```

含义：`--weights` 才下载约 57 GiB 的 BF16 权重。下载后逐文件校验，检查仓库是否
中途更新；未通过不标记 ready。下载和哈希校验均不使用 GPU，也不创建第二份完整权重。
期间进度从下载切换到 `Checking SHA256` 属正常检查，读取大文件可能需要数分钟。
SDK 会在数据盘 `.modelscope-sdk` 下写匿名会话信息，不修改 HOME、DNS 或系统代理。

运行命令的 `--checkpoint` 使用该入口打印的真实路径，其他运行参数不变。
正式复测使用已验证的本地目录，不重新从 mutable master 下载并声称版本未变。

### Hugging Face 原入口

```bash
python scripts/qwen_smoke.py prepare --models-dir /workspace/models
```

含义：只访问官方模型库并下载配置、tokenizer 和分片索引，解析完整 commit SHA。
`--models-dir` 将文件放到数据盘。打印不可变 checkpoint 路径、权重所需空间。
不加载模型、不用 GPU、不下载几十 GB 的权重。若网络失败，保留原错误供排查，
不自动切换来源不明的镜像。

确认路径和下载空间后执行以下模板，将 `<SHA>` 替换为上一步打印的 40 位 revision：

```bash
python scripts/qwen_smoke.py prepare --models-dir /workspace/models --revision <SHA> --weights
```

含义：`--revision` 冻结刚确认的版本，`--weights` 才允许下载约 57 GiB 权重。
两个下载 worker，文件保存在同一 checkpoint 目录；下载过程仍不使用 GPU。
下载中断后可用相同命令继续，不需要删除部分文件。
不要把尖括号占位符直接粘贴到 Bash。

## 四 第一条真实请求

仅在最新资源检查通过、实验使用窗口已确认后执行；将 `<SHA>` 替换为实际 revision：

```bash
python scripts/qwen_smoke.py run --checkpoint /workspace/models/Qwen3-30B-A3B/<SHA> --gpu-memory-gib 24 --cpu-memory-gib 48 --new-tokens 8 --results-dir /workspace/results/qwen_smoke
```

含义：加载本地 BF16 权重，GPU 权重映射预算 24 GiB、CPU 权重映射预算 48 GiB；
CPU 卸载仅解决加载，执行在单 GPU 上。预算不是显存总占用上限，也不是 KITE 的
Expert–KV 可分页预算。还需为 KV、激活、加载及临时传入权重保留空间。
`--new-tokens 8` 是最多生成 8 个 token，默认输入是一句手写短提示；不静默截断。
`--results-dir` 指定数据盘上的结果目录，每次创建唯一子目录，不覆盖之前的数据。
该命令才会使用 GPU。默认拒绝繁忙或显存不足的 GPU，不停止别人的任务。
不得把诊断 forward 时长当成 TTFT、TPOT 或加速比。

需要终止自己当前前台试跑时，在该终端按 Ctrl+C。不要使用广泛 kill 命令。

## 输出与验收

- `summary.json`：状态、依赖、revision、生成 token IDs、文本和证据范围。
- `request.json`：原输入、chat template 文本及精确 token IDs；关闭 thinking。
- `device_map.json`：权重存储映射，不代表 CPU 部分实际在 CPU 执行。
- `expert_trace.jsonl`：真实非空专家调用顺序、层/专家 ID、routed token 数、权重大小。
- `routing_checks.jsonl`：每层每次 forward 的 token 数总和等于 query tokens × top-k。
- `kv_snapshots.jsonl`：真实 K/V 张量形状、精度、所在设备和逻辑 bytes。
- `steps.jsonl`：加 hooks 的诊断计时及 PyTorch 显存指标，不是性能对照。
- 资源快照和模型配置文件：复核运行环境与加载边界。

生成 n 个 token 通常执行一次 Prefill 加 n−1 次 Decode；最后采样出的 token 没有再
输入模型，因此最后 KV 长度为 prompt+n−1。这不是漏采一轮。
EOS 提前结束照实记录；若在第一次就结束，不能说已验证 Decode。
第一版只处理单请求、无 padding，不宣称已有真实混合批次调度器。

固定执行器可能调用 routed-token 数为零的专家模块；脚本将其单独计数，不记为逻辑
专家访问。记录的调用顺序属于该固定 Python 执行器，不等于实际搬运顺序。
CPU offload 可能仍对空模块搬运权重，这是基础执行器行为，不应称为 KITE miss。
KV 只有张量元数据，没有 KV block trace 或硬件 cache miss。

## 核对的官方来源

- 模型与 thinking 开关：https://huggingface.co/Qwen/Qwen3-30B-A3B
- Qwen 官方国内下载说明：https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html
- Qwen 的 ModelScope 仓库：https://modelscope.cn/models/Qwen/Qwen3-30B-A3B
- 固定下载器：https://github.com/modelscope/modelscope/blob/v1.25.0/modelscope/hub/snapshot_download.py
- 固定执行器：https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
- CPU offload 语义：https://huggingface.co/docs/accelerate/v1.6.0/en/package_reference/big_modeling
- 固定 Cache 接口：https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/cache_utils.py

本机单元测试通过不等于远程真实模型已运行。只有服务器对应日志通过，才记实测完成。
