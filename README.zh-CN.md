# MSSM MLS

Python 包版本 **0.3.0** 是软件版本；**V5** 是所选科学实验版本的标签。

使用 MLS 方法扫描 MSSM 参数空间。[English](README.md)

本项目保留已经完成真实物理计算的 **V5 GP 辅助 MLS**，同时提供均匀随机扫描和较强神经网络 MLS 对照。

## 已有结果

三组均为开发种子 seed0、完整 20000 次逻辑评价；使用同一重建物理设置，以真实 Eq.(7) 硬切且 HiggsBounds 明确允许为主目标：

| 方法 | 目标点数 / 20000 | 累计效率 | 仅 Eq.(7) 点数 | 20 / 40 / 100 网格占用 | r=0.001 贪心间距计数 |
|---|---:|---:|---:|---|---:|
| V5 GP 辅助 MLS | 4642 | 23.21% | 6728（33.64%） | 30 / 94 / 448 | 3332 |
| 较强 NN MLS 对照 | 4248 | 21.24% | 10974（54.87%） | 30 / 94 / 398 | 3291 |
| 均匀随机扫描 | 0 | 0% | 1（0.005%） | 0 / 0 / 0 | 0 |

V5 的联合目标多 **394 个**，增加 **1.97 个百分点**，相对提高 **9.27495%**。仅 Eq.(7) 的数量反而降低，不能把该指标也描述为提高。间距计数只是完整归一化参数盒中按固定半径过滤后的几何数量，不代表统计独立、物理分辨率或找全目标区域。

原预设开发条件为至少提高 10%，即 4673 个目标；V5 少 31 个，没有达到。本次完整 V5 研究未满足该原预设条件，旧协议的未通过事实仍保留。本项目没有完成 V5 的独立多种子复验，也没有复现论文约 80% 的效率。seed0 用于开发选择，不能当成从未查看的测试集，不能据此声称胜过作者当年的原程序。

## 第一次使用

先将整个文件夹放在**路径不含空格**的位置，例如 `~/Projects/MSSM-MLS`，在终端进入该目录。需要 Python 3.12 或更新版本：

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python tools/verify_benchmark_data.py
```

这几步验证 Python 软件和公开数据，不会调用物理程序。实际导入依赖是 NumPy、PyTorch 和用于比较图的 Matplotlib；没有使用 SciPy。正态分布 log-CDF 由 `torch.special.log_ndtr` 计算。mpmath 用于高精度软件测试。

真实扫描还需要 C/C++、Fortran 编译器和 make，需下载并编译论文年代的外部程序：

```sh
python scripts/setup_physics.py
mssm-mls-modern doctor --backend configs/backend.local.json
```

请先读 [安装说明](docs/installation.md)。doctor 成功表示程序、版本与配置可用，不等于已经完成物理评价。本次打包没有重新下载或编译旧工具；不同电脑仍可能遇到旧源码兼容问题，不能静默换版本。源码地址与校验和见 `configs/sources.lock.json`。

先用小预算检查真实物理接口：

```sh
mssm-mls-modern scan --method random --backend configs/backend.local.json --budget 32 --initial 24 --seed 0 --output runs/smoke-random
mssm-mls-modern scan --method modern --backend configs/backend.local.json --budget 32 --initial 24 --seed 0 --output runs/smoke-modern
```

没有命中目标也可能是正常结果，小预算不能证明性能。新扫描自行生成初始参数，不会隐式使用公开的旧 1000 点、已知目标或本机旧数据库。

完整三组命令：

```sh
python tools/benchmark.py run --method modern --backend configs/backend.local.json --seed 0 --budget 20000 --initial 1000 --output runs/modern-s0
python tools/benchmark.py run --method random --backend configs/backend.local.json --seed 0 --budget 20000 --initial 1000 --output runs/random-s0
python tools/benchmark.py run --method strong-reference --backend configs/backend.local.json --seed 0 --budget 20000 --initial 1000 --output runs/strong-s0
```

建议依次运行并监控磁盘，至少保留 5 GiB 余量。每个评价的 micrOMEGAs 独立工作目录可能很大，本可移植入口不承诺提供研究工作区的硬链接去重。可明确添加 `--cache runs/truth.sqlite`，缓存命中仍占逻辑预算，失败不缓存；缓存后的耗时不是冷启动公平速度比较。

较强旧版在 2 亿廉价候选上限耗尽时可以停止。仅支持经核验的**完整轮边界**按 2 亿 → 10 亿 → 50 亿提高候选上限：[恢复步骤](docs/recovery.md)。普通公共命令的 `--resume` 和 `probe` 均关闭；未提供现代版的部分轮恢复。必须保留中断的 journal、checkpoint 和成本，不能删除已完成失败点后重算预算。

## 进一步阅读

- [安装与外部物理设置](docs/installation.md)
- [怎样运行、怎样阅读输出](docs/running.zh-CN.md)
- [原论文与 V5 方法区别](docs/method.md)
- [公开数据与口径](data/README.md)
- [逐文件代码地图](docs/code_map.zh-CN.md)
- [贡献与署名边界](CONTRIBUTING.md)

`docs/core_preservation.json` 证明 19 个科学源文件与 5 个配置逐字节保持 V5。 软件来源方面，本项目是新的重建与更新实现，不是找回的原作者源程序。公开旧 1000 点只用于明确标注的离线 `replay`：它产生训练和推荐，不产生新的真实 MSSM 评价。已有完整比较请使用公开数据验证脚本；公共 `compare` 会拒绝不同初始化类别，不能直接混合本次继承初始化的 V5 与新的 fresh 扫描。

论文：Ren 等，*Exploring supersymmetry with machine learning*，Nuclear Physics B 943 (2019) 114613，[DOI](https://doi.org/10.1016/j.nuclphysb.2019.114613)。xBIT 是后续实现参考，不是原作者源程序。项目采用 GPL-3.0-or-later；外部软件仍适用各自许可与引用要求，详见 `THIRD_PARTY_NOTICES.txt`。
