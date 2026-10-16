# 当前保留的项目文件

## 实验状态

已测的是四请求并行 decode 场景，6/7 GiB 预算下，原四路径固定成本 baseline 与完整 KITE 对照。原始运行目录为 `results/parallel-decode-search-20261016T132937Z-8668e263/`。

WiSP 式 Expert、KV 各一个固定 miss 代价的 baseline 目前只写入待测文书，尚未接入执行代码。本次仅清理目录，不修改 KITE、原 baseline、调度、计时或成本模型。不能将旧脚本的结果直接称为 WiSP 结果；后续重测前须先实现新对照。

## 四个主要入口

| 文件 | 用途 |
|---|---|
| `scripts/kite_parallel_decode.py` | 当前完整模型实验入口；现有对照仍是原四路径固定成本配置 |
| `scripts/audit_kite_parallel_decode.py` | 本地读取原始记录，核验实际计时、输出、缓存及资源日志，不使用 GPU |
| `scripts/plot_kite_parallel_results.py` | 由核验后的原始数据生成原版三张实验图，不使用 GPU；现有标签针对旧对照 |
| `scripts/package_kite_parallel_decode.py` | 打包保留的源码及必要测试，生成新名称 zip，不覆盖成功实测的原始包 |

其余保留文件是必要依赖，不需要逐个手工运行：

- `kite_controller.py`：历史复用距离、候选分配及当前信号融合。
- `kite_e2e_core.py`：共享池、请求和成本模型。
- `kite_e2e_runtime.py`：标定、KV 恢复/重放、请求执行。
- `kite_e2e_torch.py`：真实 Qwen 模型、Expert 两路径及 GPU 计算。
- `kite_e2e.py`：当前入口复用的执行、监测、指标和源码指纹函数。
- `kite_advantage.py`：当前入口复用的参数、提示词和性能运行配置。
- `kite_trace.py`、`kite_readiness.py`：环境、资源门禁和记录。
- `qwen_smoke.py`、`expert_paths.py`、`multi_trace.py`、`workload_baseline.py`：当前链路仍会导入的模型加载、计时、输入和指标工具。文件名虽然来自早期阶段，不能直接删除。
- `audit_kite_e2e.py`：结果核验共用函数。
- `screen_kite_parallel_decode.py`：当前容量逻辑测试所需的参考重放函数，不是新的 GPU 评测。

`tests/` 只保留以上链路的 15 个本地逻辑测试文件。这些测试不加载模型或启动远程 GPU 实验；测试通过不等于硬件性能或输出质量认证。

## 归档和保留原则

旧脚本、旧测试、17 个旧上传包和 14 个临时解压副本移入 `archive/cleanup-20261016/`。没有永久删除，移动前后核对内容指纹。快照 `source-before-cleanup.zip` 包含清理前源码、测试、文档和 README，清单在 `manifest.json`。

`results/`、`reports/`、专利文件和原始实验记录不动。原成功实测包 `dist/kite-parallel-decode-20261016-v1.zip` 保持原字节，供冻结证据核验。清理后的打包入口生成 `dist/kite-parallel-decode-clean-20261016-v1.zip`，不代表新的算法或新的 WiSP 对照。

`docs/` 的其他指南是历史过程记录，以本文和根目录 README 为当前说明。旧文书生成脚本也已归档；需要恢复时应还原到原路径，不能在归档目录直接执行。
