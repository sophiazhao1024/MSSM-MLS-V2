# 运行与阅读输出

## 三条路径

1. `modern`：本次交付的 V5，默认 Gaussian process 辅助推荐。
2. `strong-reference`：与 V5 实际比较的较强 NN MLS 重建，输入/输出含标准化；不是作者原始代码。
3. `random`：均匀随机；使用 modern 的初始化随机流和预算设置，不训练模型。

`historical` 另保留原始数值输入、仅丰度取 log 的 NN 重建，不能把它混称为较强对照或将其停滞说成原论文必然有错。

运行前必须 `doctor` 成功，运行命令见 README。输出目录必须是新的，不能覆盖以前运行。32/24 的短预算用于物理接口检查，不能重现完整开发比较的训练过程。默认完整比较为初始1000、总预算20000。新运行不自动导入公开初始点或旧数据库。

## 数据流

```text
真实参数 x
  -> FH/HB/HS + micrOMEGAs
  -> status + 实测观测量 + HB + provenance
  -> journal 累积记录
  -> GP/分类器 fit
  -> 大量廉价候选 predict
  -> exploit / entropy / uniform random 选点并交错执行
  -> 再做真实物理计算并写 journal
```

网络/GP 预测不是最终物理判定。标准化只在模型训练/预测内部，恢复物理单位后才构造相容性/likelihood；真实验收始终按 GeV、真实 chi2 和实际丰度判定。

## 输出文件

- `manifest.json`：源码、设置、物理身份、种子和状态。`completed` 才表示预算完成。
- `records.jsonl`：每行一次逻辑评价，编号从1递增；保留失败和初始化。不要仅统计成功返回四观测量的点。
- `summary.json`：累计真实目标与 Eq7-only 计数。主效率 = targets / attempts。
- `training_history.json`：每轮 fit、候选来源与选择诊断；模型兼容度不是经校准的真实概率。
- `checkpoint.json`：完整轮边界；公共 CLI 不接受直接 `--resume`。
- `operational_recovery.json` / `operational_guard.jsonl`：统一错误守卫的运行与逐次账本。已知 MO 数值失败仍是 tool_error，计预算、不自动重试；连续五次未知技术失败写入第五条后停止。
- `physics/point_.../`：参数卡、物理程序输出及可追溯记录，可能占用大量磁盘。
- `physics_identity.json`、`physics_manifest.json`：真实程序与配置身份；不同身份不能无声复用旧真值。
- `execution.json`：仅启用缓存时出现的请求/命中/新调用账目。
- `portable_run.json`：辅助 benchmark 控制器记录，不改变核心算法。

缓存命中与继承初始化仍计逻辑预算。外部实际调用数、训练时间、候选预测数和总耗时属于不同成本；原实验存在历史缓存和断点，不能把其 walltime 当成冷缓存速度优势。

## 离线查看

```sh
python tools/verify_benchmark_data.py
mssm-mls-modern replay --method modern --seed 0 --output runs/offline-replay
```

前者仅复算公开60000行数据；后者从明确保存的旧1000点训练并推荐，**不会调用真实物理程序**。它们都不是新增实验。`compare` 要求匹配初始坐标、初始真值、物理设置与运行类别，不能直接把继承旧初始点的开发运行和新的 fresh run 混在一起。
