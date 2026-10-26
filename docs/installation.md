# Installation and physics setup / 安装

Use the complete source checkout, Python 3.12+, and a path without whitespace. Linux and macOS are the intended build-script platforms. The archived calculations used macOS arm64; this packaging validation does not certify every compiler, platform or Python/dependency combination.

## Python

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
mssm-mls-modern --help
```

Declared runtime imports: NumPy, PyTorch and Matplotlib (plot command). PyTorch supplies `torch.special.log_ndtr`; SciPy is unnecessary. `pytest` and `mpmath` are explicit test dependencies. `requirements-tested.txt` records the environment used for software verification, not a portable cross-platform binary lock. See `validation_report.json` for the exact checks completed for this distribution. Wheels contain the Python package; run setup/tools from the full repository because third-party drivers, configurations and data are source-distribution assets.

## Compilers and external sources

You need `gfortran`, a C compiler, a C++ compiler and `make`. On macOS install Apple's command-line tools and a compatible GNU Fortran compiler using your preferred package manager. The scripts use `xcrun` for the SDK on macOS; compiler installation and Apple's licence acceptance are outside the Python package. On Linux provide the corresponding GNU compiler/development environment. Use a compiler toolchain directory explicitly if needed:

```sh
python scripts/setup_physics.py --toolchain-bin /your/compiler/bin
```

First inspect the licences in each external archive. The project GPL notice does not relicense them. Pinned sources are:

| Program | Version | Source provenance |
|---|---|---|
| FeynHiggs | 2.13.0 | Official archive, SHA256 locked |
| HiggsBounds | 4.3.1 | Official archive, SHA256 locked |
| HiggsSignals | 1.4.0 | Official archive, SHA256 locked; same-number updates exist |
| micrOMEGAs | 4.3.2 | Fixed historical HEPcodes mirror commit, SHA256 locked; original official archive checksum unavailable |

```sh
python scripts/setup_physics.py --download-only
python scripts/setup_physics.py
```

The first command downloads/checks/extracts but does not build. The second reuses verified archives and builds. Downloads and executables are deliberately excluded from Git. If an old archive URL is unavailable or its checksum differs, stop and retain the evidence; do not silently replace its contents. The setup applies recorded compiler-compatibility changes to downloaded sources (BOZ conversion and STOP spacing), without changing the supplied physical model expressions. Build logs are under `build/build_logs/` by default.

The builder writes `configs/backend.local.json` (or `backend.built.json` when local already exists). Local configurations are machine specific and ignored by Git. All paths in `configs/backend.example.json` are relative to that JSON file, not the shell working directory. Its five frozen preset files are unchanged; copy it to a new local configuration when editing host paths.

```sh
mssm-mls-modern doctor --backend configs/backend.local.json
```

Expect ready=true and the recorded versions. A missing executable/version/template or invalid warm-template manifest means not ready. Doctor checks dependencies without calling a new physics parameter point. Backend hashes change when binaries/configuration are rebuilt; cached labels or resume proofs from a different identity cannot be silently reused.

## Optional warm template

A verified warm CalcHEP template can reuse compiled process code while still solving each new parameter point. This requires a previously successful **real local evaluation**, the same binaries and settings, and the audited `scripts/warm_template.py` procedure. Read its `--help`; it does not import archived public CSVs as physical truth. It changes a local backend configuration, so start a new named run and keep that identity. The basic cold template is usable but can be slower and generate many files. No warm-template creation, source download or physics rebuild was performed in this packaging run.

## Troubleshooting

- `No module named ...`: activate the environment used for installation and run commands from the source root.
- Legacy make/path errors: move the checkout to a path without spaces and inspect the first build-log error.
- `doctor` fails: inspect its JSON diagnostics and exact tool paths; do not begin a scan with incomplete tools.
- `proposal_exhausted`: an NN rejection-sampling cap, not a physical failure; preserve checkpoint and follow the narrow cap recovery guide.
- Five consecutive unidentified technical errors: the shared typed guard stops after recording the fifth answer. Fix the operational cause and preserve all evidence; this release does not offer arbitrary partial recovery.
