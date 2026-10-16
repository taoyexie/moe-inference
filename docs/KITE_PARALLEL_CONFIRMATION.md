# 冻结 6 GiB：留出提示词的独立确认

**状态更新：用户决定不追加这三轮确认，改为直接交付现有实测数据的简洁实验图。以下命令仅保留作历史参考，不是当前待执行任务。**

探索原始记录核验通过，见 `reports/kite_parallel_decode_20261016/search_findings.md`。
按已有选择规则冻结 6 GiB：82.173740 → 75.348227 秒为单次探索结果，尚不是独立确认平均值。
冻结清单和 SHA256 保存在 `reports/kite_parallel_decode_20261016/confirmation/frozen_manifest.json`。
原 zip、代码、模型、参数、请求长度、到达和输出长度不改，不需新上传文件。

## 这次只确认，不继续找场景

已有 `confirm` 入口会读取成功搜索的 `selection.json`，检查版本和模型一致，
只运行选中的 6144 MiB，而不是再次运行 6/7 GiB 搜索。
输入正文自动换为代码中预留的图书馆、烹饪、天文和训练主题，未参与搜索；
长度仍为 8192/11264/11264/11264，四条同时到达，各输出 16 tokens。

三个 seed 各启动一个新的 Python 进程、重新加载模型，分别测完整 baseline 和 KITE。
seed17/41 基线在前，seed23 KITE 在前，顺序按已声明协议，不是完全平衡的设计。
三个会话重复同一套留出提示，seed 主要用于固定程序状态和声明运行顺序，
不能把它们说成三种新工作负载或三个独立抽样流量集。
同一个进程内的四条请求也不是四次独立实验。

每次完整 workload 自己的标定、初始化、全部 prefill/decode、缓存分配、写回和搬运照常计入 E2E。
保留全部正负结果、启动区间、主执行和资源记录，不新增消融或精度探针。
新数值结果待 GPU 执行，不以探索数值或拟合估计填充确认表。

## 运行命令与逐项含义

执行位置：已经进入的 Docker 容器 `/workspace`。不需要再次上传、解压或安装依赖。
GPU 当前空闲再运行；脚本每个会话都会检查资源，忙卡、OOM 或异常会停止。

```bash
for seed in 17 23 41
do
  sleep 5
  /workspace/.venv/bin/python /workspace/kite-parallel-decode-20261016-v1/scripts/kite_parallel_decode.py confirm \
    --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 \
    --selection /workspace/results/kite_parallel_decode/parallel-decode-search-20261016T132937Z-8668e263/selection.json \
    --seed "$seed"
  if [ "$?" -ne 0 ]
  then
    break
  fi
done
```

- `for seed in 17 23 41`：顺序执行三个独立进程会话，不并行占卡。
- `sleep 5`：每次启动前等 5 秒，让前一进程 CUDA 清理完成；不绕过资源检查。
- `/workspace/.venv/bin/python`：使用现有实验虚拟环境，不修改系统 Python。
- `confirm`：进入冻结配置的留出提示确认，不重做场景搜索。
- `--checkpoint`：复用已有完整 BF16 权重，不重新下载。
- `--selection`：使用刚刚成功搜索生成的原始选择记录，自动固定 6 GiB 和版本。
- `--seed "$seed"`：分别传入 17、23、41，并按协议安排两组顺序。
- `if [ "$?" -ne 0 ]` 和 `break`：上一会话非正常结束时停止后续会话，保留错误记录，不强行重试。

每个会话仍要约十分钟加载，另有两次 workload，三次合计可能约 40 分钟；实际以机器状态为准。
不要同时再启动第二个实验。各会话结束会打印自己的
`/workspace/results/kite_parallel_decode/parallel-decode-confirm-...` 目录。
完成后下载这三个目录，分别离线核验，再汇总三个会话的配对结果；不得只保留最快的一次。

注意：v1 在加载模型时共用了一句包含“two pairs, 6/7 GiB”的旧提示文字。
`confirm` 的实际执行分支只取已冻结的 6 GiB；后续每个会话应只有两条 `budget=6144 MiB` 工作负载，
不是重新测 7 GiB。为保持实验代码和哈希冻结，不为这句提示重新发包。
