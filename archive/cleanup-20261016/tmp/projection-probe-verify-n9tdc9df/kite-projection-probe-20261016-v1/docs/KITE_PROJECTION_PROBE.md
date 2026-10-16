# 首层轻量投影探针：不再加载完整 Qwen

完整短诊断已定位：混合 decode token 的第一层实际 Q/K/V 和 SDPA 原始输出对全前缀参考误差为零，
经过整个 attention 模块后的误差非零。固定版本模块在 SDPA 后只做 reshape 与 o_proj，
支持将该短例的第一次额外差异定位到输出投影环节；不是证明整个模型只有这一处差异。
原生增量自身还在第一层 v_proj 出现非常小的差异，随后路由可能变化。

本探针仅加载真实第一层 attention（Q/K/V/O 与两个 head norm）、第一层 input norm，
并从 safetensors 按 token ID 读取所需的 embedding 行。不加载任何专家、其他层、LM head 或完整 embedding。
使用上一轮已留档的同一组真实 token，重建第一层的全前缀、缓存增量与混合输入。
不生成文本，不启动标定或主实验，不作为速度/质量结果。

固定比较三种诊断设置，不自动搜索：

1. 当前默认 BF16 配置，确认能否复现第一层差异。
2. 仅关闭 `allow_bf16_reduced_precision_reduction`，检验一种最小数值设置调整。
3. Q/K/V/O 线性投影临时 FP32 计算、输出仍回到 BF16，作为算子级对照；关闭 TF32。

另将同一真实输入行重复为 1/9/13 行，隔离矩阵形状引起的投影差异。
所有临时配置和 forward 方法恢复；正式 v2 源码、阈值、精度和算法都不变。
任何设置即使改善首层，也必须再通过完整模型检查，不能直接声称修复或性能提升。
FP32 对照不等于原定全 BF16 实验；若以后正式采用，必须明确披露、两组同时修改并重新标定和记账。

依据：[PyTorch 2.7 数值说明](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html)
记录批形状差异及 BF16 reduction 开关；不能保证关掉它就逐位一致。

## 执行位置与每条命令的含义

Windows PowerShell：上传轻量探针代码包，不动旧代码、权重或结果。

```powershell
scp -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\lab_gpu_ed25519" -P 2022 "D:/moe inference/dist/kite-projection-probe-20261016-v1.zip" xty@10.214.243.19:/data2/xty/epkv/
```

已有服务器容器 `/workspace`：用现有环境解压到独立目录；`-e` 表示解压。

```bash
/workspace/.venv/bin/python -m zipfile -e /workspace/kite-projection-probe-20261016-v1.zip /workspace
```

同一容器、GPU 空闲时运行：`--checkpoint` 指现有权重，`--diagnostic` 读取上次真实诊断的 token 记录。
内置忙卡与源码/模型检查；只读首层，不再执行完整模型加载，实际耗时仍以服务器为准。

```bash
/workspace/.venv/bin/python /workspace/kite-projection-probe-20261016-v1/scripts/first_layer_probe.py --checkpoint /workspace/models/Qwen3-30B-A3B/modelscope-1208a85873f530e2 --diagnostic /workspace/results/kite_numerics/packed-diagnostic-20261016T112746Z-c8b95ded/numerical_diagnostic.json
```

结果在 `/workspace/results/kite_projection_probe/first-layer-...`，主要文件 `projection_probe.json`。
`probe_complete` 仅指探针执行完成，不是完整模型正确性或性能验收。
