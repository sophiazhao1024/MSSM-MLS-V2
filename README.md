# MSSM MLS

Python package version **0.3.0** is the software version; **V5** identifies the selected scientific experiment.

MSSM parameter scans with the MLS method. [中文说明](README.zh-CN.md)

This release preserves **V5**, a Gaussian-process-assisted update of an MSSM Machine Learning Scan. It includes real physics interfaces, uniform-random and strong neural-network comparators, full seed0 comparison data, and software tests. 

## Observed result and release decision

One **development seed**, 20,000 logical evaluations per method, with the same reconstructed historical physics profile:

| Method | Eq.(7) + known HB allowed | Cumulative fraction | Eq.(7) only | Grids 20 / 40 / 100 | Greedy packing, r=0.001 |
|---|---:|---:|---:|---|---:|
| V5 GP-assisted MLS | 4,642 | 23.21% | 6,728 (33.64%) | 30 / 94 / 448 | 3,332 |
| Strong NN MLS reference | 4,248 | 21.24% | 10,974 (54.87%) | 30 / 94 / 398 | 3,291 |
| Uniform random | 0 | 0% | 1 (0.005%) | 0 / 0 / 0 | 0 |

V5 found **394 more joint targets**, a **1.97 percentage-point** or **9.27495% relative** increase over this strong reference. Its Eq.(7)-only count was **lower**. Packing is a deterministic spacing-filtered count in the normalized full parameter box; it does not establish statistical independence, physical resolution, or complete coverage.

The original development condition required at least 10% improvement (4,673 targets). V5 missed it by 31 targets. The completed V5 study does not satisfy that original criterion; its historical protocol result remains a failure, not a qualified pass. There is **no completed independent multi-seed validation** of V5 and no demonstrated reproduction of the paper's approximately 80% efficiency. This single development comparison does not prove superiority to the authors' original program. Development selection also prevents treating seed0 as an untouched test seed.

## Start here

Use Python 3.12+ in a project path without spaces. From this repository:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python tools/verify_benchmark_data.py
```

These checks require **no physics executables** and create no new MSSM results. Actual runtime imports are NumPy, PyTorch and Matplotlib (comparison plotting); SciPy is not used. `torch.special.log_ndtr` supplies the Gaussian log-CDF. Test dependencies also include pytest and mpmath. See [installation](docs/installation.md) for compilers and the pinned external programs.

For real physics, install a C/C++/Fortran toolchain and make, inspect upstream licences, then:

```sh
python scripts/setup_physics.py
mssm-mls-modern doctor --backend configs/backend.local.json
```

A successful doctor validates availability and identities; it does not evaluate a parameter point. Legacy downloads/builds may fail on a new host; this release did not repeat their download or compilation. `configs/sources.lock.json` fixes the source archives and documents the historical micrOMEGAs mirror. Do not silently substitute a version when a source is unavailable.

Run small **real physics interface checks**, in distinct output directories:

```sh
mssm-mls-modern scan --method random --backend configs/backend.local.json --budget 32 --initial 24 --seed 0 --output runs/smoke-random
mssm-mls-modern scan --method modern --backend configs/backend.local.json --budget 32 --initial 24 --seed 0 --output runs/smoke-modern
```

A smoke run can legitimately find no target. Its purpose is the interface and journal, not an efficiency claim. Every fresh run generates its own initial samples; it does not inject the shipped saved1000 fixture or a known target.

## Fresh full comparisons

Run the methods sequentially and keep their complete journals. Each command can be slow and consume substantial disk space; preserve at least 5 GiB of free reserve and monitor `physics/`, checkpoints and logs. The basic public interface creates a private micrOMEGAs work copy per evaluation; no storage deduplication is claimed for this portable release.

```sh
python tools/benchmark.py run --method modern --backend configs/backend.local.json --seed 0 --budget 20000 --initial 1000 --output runs/modern-s0
python tools/benchmark.py run --method random --backend configs/backend.local.json --seed 0 --budget 20000 --initial 1000 --output runs/random-s0
python tools/benchmark.py run --method strong-reference --backend configs/backend.local.json --seed 0 --budget 20000 --initial 1000 --output runs/strong-s0
```

`--cache runs/truth.sqlite` is an explicit optional addition. No private workspace SQLite file is needed. A cache hit retains its logical evaluation in the denominator; failed evaluations are not cached. Cache reuse changes timing, so the archived yields are not an uncached speed benchmark.

The strong reference starts at a 200-million cheap-proposal cap and can stop with `proposal_exhausted`. Only the separately verified complete-boundary cap control supports 200m → 1b → 5b continuation: [recovery instructions](docs/recovery.md). Public `scan --resume` and `probe` remain disabled. A partial journal cannot be discarded or silently restarted as if its completed calls never happened. Modern partial recovery is **not** supplied here.

The public `compare` command requires matching run kind, physics identity, budget, initial coordinates and initial truth. It is intended for new paired fresh runs, not directly mixing the archived inherited-initial V5 probe with fresh comparators. Verify the shipped archive with `tools/verify_benchmark_data.py`.

## Method and files

- [Published method, V5 changes and unresolved physics settings](docs/method.md)
- [Installation and real-physics setup](docs/installation.md)
- [Running, data flow and output interpretation](docs/running.zh-CN.md)
- [Public data, definitions and limitations](data/README.md)
- [Source map in Chinese](docs/code_map.zh-CN.md)
- [Contributions and attribution](CONTRIBUTING.md)

The 19 Python scientific modules and five scientific configuration files are byte-identical to the audited V5 development source; `docs/core_preservation.json` records their hashes. Packaging glue and portable tests are separately identified. Software provenance: this is a new reconstruction/update, not recovered original author source. `validation/initial1000_records.jsonl` is an explicitly saved-truth offline fixture, not independent new validation. `replay` only trains/recommends from it and performs zero new physics evaluations.

## Citation and licence

Scientific starting point: Ren et al., *Exploring supersymmetry with machine learning*, Nuclear Physics B 943 (2019) 114613, [DOI](https://doi.org/10.1016/j.nuclphysb.2019.114613). Cite external physics packages as required by their authors. xBIT is a later framework reference; it cannot supply the lost original MLS source.

GPL-3.0-or-later applies to this project's new code. The HiggsSignals-derived driver retains its upstream notice. External libraries are downloaded separately under their own terms; see [third-party notices](THIRD_PARTY_NOTICES.txt).
