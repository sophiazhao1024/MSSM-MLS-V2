# MSSM MLS V2

Machine Learning Scan for MSSM parameter exploration, with Gaussian-process surrogates and adaptive candidate selection. Selected points are evaluated with the physics programs before acceptance.

Main updates

Predictions with uncertainty estimates: Gaussian-process models support both exploitation of promising regions and exploration of uncertain regions.

Selection targeting joint acceptance: HiggsBounds predictions and physics-based features help identify candidates likely to satisfy the combined requirements.

Global and local search: Candidates are generated across the parameter space and around promising evaluated points, with diversity-aware batch selection.

Explicit exploration: Each full batch contains 63 exploitation points, 21 uncertainty-guided exploration points and 21 unfiltered random points.

Clear failure handling: Unphysical points and numerical failures are recorded separately.


Observed benchmark advantages

In one development-seed comparison under a fixed reconstructed physics setup, V5 was compared with a strong reconstructed neural-network MLS baseline.

Logical evaluations to obtain the first 1,000 jointly accepted points:
 Reconstructed MLS baseline 6,546
 V2 4485

V2 required 31.5% fewer logical evaluations to obtain the first 1,000 jointly accepted points.  


Complete comparison records are available in data/benchmark/.



## Method and files

- [Published method, V2 changes and unresolved physics settings](docs/method.md)
- [Installation and real-physics setup](docs/installation.md)
- [Running, data flow and output interpretation](docs/running.zh-CN.md)
- [Public data, definitions and limitations](data/README.md)
- [Source map in Chinese](docs/code_map.zh-CN.md)
- [Contributions and attribution](CONTRIBUTING.md)



