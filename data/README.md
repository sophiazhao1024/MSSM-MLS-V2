# Public complete seed0 data

`benchmark/` contains60000 sanitized logical evaluations:20000 each for V5, strong standardized NN MLS and uniform random. Each method has equivalent JSONL and CSV files. `summary.json` is independently recomputed from those files; `provenance.json` binds the original opaque journal SHA256 and sanitized output SHA256. No final101/102/103 scientific data are included or read for this release.

```sh
python tools/verify_benchmark_data.py
```

Every initialization, exploration point, physical rejection, technical error and cache logical request remains. V5's1000 inherited real labels are explicitly marked, not claimed as new external calls. The reviewed105 failed-initial HB metadata transform is documented in provenance; it changes no coordinates, status, available observables or initial targets. The public derivative is not a raw journal and must not be imported as a cache/recovery proof.

Columns include exact float-representable parameters/complete observed labels, eval_id, seed, method, origin, iteration, status, HB and target flags, cache-hit indicator when known, validation/candidate source where available, and selected partial raw-Higgs numbers. Missing labels are null in JSONL and empty in CSV. `cache_hit=null` means unavailable, not a claimed cache miss. No paths, commands, walltimes, free-text errors or surrogate predictions are distributed. This preserves the requested scientific comparison while avoiding machine-specific provenance leakage.

Main numerator = successful true Eq7 strict hard cuts AND HB integer1; denominator = all20000 logical attempts. Eq7-only is shown separately and was lower for V5. Grid occupation uses the fixed full box mapped to [0,1]^3. Packing sorts target coordinates lexicographically and retains a point only if Euclidean distance from every previous retained point exceeds0.001. It is greedy, not maximum packing, and makes no statistical-independence/physical-resolution claim.

The old10%/4673-target development condition was not met. The completed study retains that original protocol outcome. There is one development seed and no completed independent multi-seed V5 replication. Physical source/configuration reconstruction uncertainties remain; this is not an80% paper reproduction or an uncached timing benchmark.
