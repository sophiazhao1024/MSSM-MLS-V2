# V5 代码地图

入口是 `cli.py`，不是 `learning.py`。默认配置中的 `surrogate_kind=gp` 会选择 `gp_learning.py`；旧 NN ensemble 仍保留为另一路实现，不能用其说明冒充 V5 主回归器。

| 文件 | 作用 |
|---|---|
| `__main__.py`、`__init__.py` | 模块入口、包说明 |
| `cli.py` | doctor、point、scan、replay、compare；公共不安全续跑拒绝 |
| `settings.py` | modern 配置及有效性检查 |
| `domain.py` | 参数范围、固定参数、物理单位和 Eq7 硬切 |
| `cards.py` | FH 输入卡及 HS/HB 选项 |
| `backend.py` | 外部程序调用、解析、状态/来源与 warm 模板验证 |
| `execution.py` | 精确输入真值缓存；失败不缓存；成本单独记账 |
| `operational_recovery.py` | 统一已知/未知错误守卫和严格自身前缀校验 |
| `runner.py` | modern/random 主循环、初始化、journal、checkpoint 与回调 |
| `gp_learning.py` | V5 主模型：四个观测量 GP + 可用时的 HB ratio GP、分类器与变换 |
| `learning.py` | 共用 NN 构件、标签/holdout 辅助、可选 ensemble 路径 |
| `acquisition.py` | 候选池、近目标局部推荐、物理相容性、稳定 log entropy、距离排斥和交错执行 |
| `historical.py`、`historical_learning.py` | 原始输入/仅丰度 log 的独立 NN 重建，不是强对照 |
| `reference/baseline.py`、`reference/learning.py`、`reference/domain.py`、`reference/__init__.py` | 原较强标准化 NN 对照的保留实现与命名空间 |

`tools/benchmark.py` 是单独的可移植工程控制，不修改19个核心模块。`tools/verify_benchmark_data.py` 用标准库重算公开完整数据。`docs/core_preservation.json` 绑定19个模块和5个配置。`tests/_snapshots/` 只保存旧版本对照测试oracle，不是默认科学入口，也没有 V6 科学更新。

学习顺序建议：domain → cards/backend → runner → settings → gp_learning → acquisition → operational_recovery/execution → CLI → strong reference。先沿一次参数从输入、真实计算、记录、训练、推荐、再真实计算的闭环阅读；标准化/预测概率与真实物理验收分开理解。
