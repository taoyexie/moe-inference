# 真实 Expert 两路径实测 v1

这一步只做专家路径微基准，不是 KITE 整体运行，不测 KV，也不证明分配器速度提升。
旧 smoke 权重和环境继续使用，不安装或升级任何依赖。

## 数据来源和边界

旧 smoke 没有保存激活。首先重跑同一短请求，默认第 0/23/47 层各选择
旧 Prefill trace 中 routed-token 数最多的一个专家（平局选较小 ID），
从实际专家调用前抓取 BF16 输入。仍使用完整模型和相同 CPU offload；
一次 Prefill 和一次 Decode，最多生成两个 token，正常退出后释放模型。
输入只复制到主存并保存为小型 safetensors，不会下载新权重。
捕获复制影响诊断计时，此次不能拿来做端到端性能对比。

然后用权重索引定位所选专家的 gate/up/down 三个真实矩阵，
从已有 safetensors 文件局部加载；不重新加载全部 30B 权重。
使用 Transformers 4.51.3 的实际 Qwen3MoeMLP 实现，不拿随机权重代替。

q 是一次调用路由给**一个专家**的 token 数，不是 request 数、batch 大小或总 prompt 长度。
默认扫描 q=1/2/4/8/16/32/64/128，另保留捕获的原生 q。
小于原生 q 取输入前 q 行；大于它循环重复输入行。日志逐点标注，
属于真实激活上的受控尺寸扫描，不声称这些 q 都来自新的真实路由。

GPU 使用 BF16；CPU 实现采用 FP32 算子，权重由原 BF16 数值提升。
CPU FP32 权重预先准备，18 MiB/专家，GPU BF16 权重 9 MiB/专家。
输入输出均 BF16，CPU 路径内的格式转换计时。
它是指定 CPU FP32 实现的实测，不等于最优 CPU BF16/量化内核性能。

## 四种测量项

两条完整路径从 GPU 输入可用开始，到输出在同一预分配 GPU 缓冲区可用结束：

- `upload_gpu`：三组权重 H2D + GPU 专家计算 + 写入输出缓冲区。
- `cpu_return`：输入 D2H + BF16→FP32 + CPU 专家计算 + FP32→BF16 + 输出 H2D。
- `gpu_hit`：相同任务的常驻权重 GPU 计算 + 写入输出缓冲区，作为 hit 基线。
- `weight_h2d_only`：固定 9 MiB 三组权重的纯搬运，仅作诊断，不是完整恢复路径。

每次开始前同步，结束后同步，使用 CPU wall time；不测重叠或 exposed latency。
主存权重 pinned，GPU 权重目标和输入输出暂存区预分配；模块内部临时张量分配
在两条计算路径的计时中。GPU 输入在计时外准备。
每个 trial 随机交错四项，保存完整时间及完整路径减该 trial 的 gpu_hit 时间。
差值即使负数也原样保留，不截断或挑选最好结果。

默认 5 次预热、30 次测量。独立会话应分别运行三个进程，不把一个进程的
30 次 trial 当作 30 个独立会话。首次只用 2 次预热、5 次测量试跑。
重复同一专家会热 CPU cache，明确属于热数据微基准；还没有评估大专家工作集的
冷 LLC/DRAM、跨 NUMA 和调度争用影响。

正确性逐次对照同输入 BF16 GPU 常驻专家输出，要求 relative L2 ≤0.03，
最大绝对误差除以参考输出最大绝对值 ≤0.05；不通过立即失败留档。
CPU/GPU 有不同计算精度，不声明逐 bit 一致，不为获得漂亮图而放宽容限。
这只验证独立专家输出，不等于完整生成结果的等价性测试。

## 第一步：上传代码包

在 **Windows PowerShell** 执行：

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:\moe inference\dist\kite-expert-paths-20261015-v1.zip" xty@10.214.243.19:/data2/xty/epkv/
```

作用：将本地小型代码包上传到你的服务器数据目录，不传权重、不占 GPU。
`-i` 指定你已有的私钥文件，`IdentitiesOnly=yes` 只尝试该身份，`-P 2022`
是 SSH 端口；它不会输出私钥内容。目标文件名首次使用，不覆盖旧 smoke 包。
预期上传进度到 100%。

## 第二步：解压新代码

在 **容器的 `/workspace`** 执行：

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-expert-paths-20261015-v1.zip /workspace
```

作用：`-m zipfile` 调用 Python 自带 ZIP 工具，`-e` 解压，最后一项是解压目标。
首次创建 `/workspace/kite-expert-paths-20261015-v1`，不修改模型、环境或旧结果。
成功通常没有输出。相同包重复解压会覆盖这个代码目录里的同名文件，运行期间不要重解压。

## 第三步：采集真实输入

仍在 **容器** 执行：

```bash
/workspace/.venv/bin/python /workspace/kite-expert-paths-20261015-v1/scripts/expert_paths.py capture --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --reference-run /workspace/results/qwen_smoke/qwen-20261015T011335Z-1993559b
```

作用：`capture` 重跑短请求并抓取所选专家实际输入。
`--checkpoint` 指向已下载、已核验的权重；`--reference-run` 提供原始请求与专家选择依据。
默认权重预算 GPU 24 GiB、CPU 48 GiB、CPU 8 线程；将当前进程绑到 GPU 邻近的
逻辑 CPU 24–31，不修改他人进程或服务器全局配置。仍需空闲 GPU 和足够主存，
脚本先检查，不符合条件就停止。此步是完整加载和推理，会占用 GPU 数分钟。
每次创建唯一结果目录，不覆盖旧 trace。成功输出 `Capture directory: /workspace/results/expert_inputs/capture-...`。

## 第四步：小规模路径试跑（拿到 capture 目录后）

下方 `<capture目录>` 是占位符，必须替换成第三步打印的完整真实路径，不能原样运行。
在 **容器** 执行：

```bash
/workspace/.venv/bin/python /workspace/kite-expert-paths-20261015-v1/scripts/expert_paths.py bench --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --capture-dir <capture目录> --q 1,8,32,128 --warmup 2 --repeats 5
```

作用：`bench` 只加载三个专家，`--capture-dir` 指向刚才真实输入，
`--q` 指定扫描点（另外自动加入原生 q），`--warmup 2` 每点每测量项预热两次，
`--repeats 5` 每点五个随机交错 trial。它不重新运行完整模型、不下载文件，
只占用少量 GPU 显存，并在数据盘保存 JSON/JSONL。打印每点两条路径的毫秒数
和 `correctness PASS`；最后输出 `Benchmark directory: ...`。
这是试跑，不是正式统计结果。核验试跑后再执行默认 5/30，分三个独立进程会话。

## 输出与诚实报告

Capture 输出：`capture.json`、三个真实输入 safetensors、完整短推理 trace。
Bench 输出：`summary.json`、逐次 `trials.jsonl`、逐点 `points.jsonl` 和 `point_summary.json`。
失败目录保留 `status=failed`、异常和已有记录，不混入通过样本。
记录精度、线程、亲和性、checkpoint 指纹、激活哈希、真实权重 tensor 哈希、资源快照。

本机只有记账单元测试和语法检查，未执行 Torch 内核；GPU 正确性与时间以服务器实际输出为准。
先不拟合/画图，拿到实测后按分专家结果检查稳定性、误差和交叉点，再决定常数/线性近似是否适用。
没有交叉点也如实报告，不能将 upload+GPU 总时间提前写成严格常数。

## 原始实现来源

- 固定 Qwen 专家实现：https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
- 局部 safetensors 读取：https://huggingface.co/docs/safetensors/main/en/index
- 固定 PyTorch 复制/计时接口：https://github.com/pytorch/pytorch/blob/v2.7.1/torch/cuda/__init__.py

