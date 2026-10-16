# KITE 真实模型实验

当前主线：Qwen3-30B-A3B 完整模型的 Expert/KV 联合显存分配与实际端到端测量。

## 现在需要知道的事

- 现有实测是原四路径固定成本 baseline 与完整 KITE 的 6/7 GiB 对照；原始结果和图片未改。
- WiSP 式两类固定代价 baseline 只更新了文书，尚未实现或重测。不要将现有结果标成 WiSP 实测。
- 本次仅整理本地目录，不修改推理代码、不运行 GPU 实验、不操作服务器数据。
- 当前文件及入口详见 [项目说明](docs/CURRENT_PROJECT.md)；服务器收尾与数据保留详见 [服务器数据](docs/SERVER_DATA.md)。

## 日常只需关注这些目录

| 目录 | 内容 |
|---|---|
| `scripts/` | 18 个保留文件：完整运行链路、核验、绘图及打包依赖 |
| `tests/` | 15 个保留的本地逻辑测试文件 |
| `results/` | 已下载的原始实验记录，未移动或删除 |
| `reports/` | 核验报告、实验图和专利文档，未移动或删除 |
| `dist/` | 原成功实测包及清理后的新源码包 |
| `archive/cleanup-20261016/` | 旧测试脚本、旧包、临时解压副本、完整清理前源码快照及恢复清单 |
| `docs/` | 当前说明和历史指南；其他历史运行安排不自动生效 |

四个主要入口是 `kite_parallel_decode.py`（完整实验）、`audit_kite_parallel_decode.py`（核验）、`plot_kite_parallel_results.py`（绘图）、`package_kite_parallel_decode.py`（打包）。其余保留脚本属于依赖，不能仅按旧文件名删除。

## 保留的成果

- [WiSP 对照待测版交底书](reports/KITE专利技术交底书_WiSP对照待测版.docx)：说明新对照方法，原图表暂留并标注实际来源。
- [原对照实测结果](reports/kite_parallel_decode_20261016/search_findings.md)。
- [第一版三张实验图](reports/kite_parallel_decode_20261016/figures/README.md)。
- 原成功实测源码包：`dist/kite-parallel-decode-20261016-v1.zip`，保持原哈希。
- 清理后源码包：`dist/kite-parallel-decode-clean-20261016-v1.zip`，算法未变，不是 WiSP 新实验包。

清理前的长版 README 和全部旧说明可从 [归档说明](archive/cleanup-20261016/README.md) 对应的源码快照恢复。
