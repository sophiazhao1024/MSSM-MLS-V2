# Complete-boundary strong-reference cap recovery

The default strong NN reference uses a finite200-million cheap-proposal cap. A narrow valid target/low likelihood can exhaust it before finding the requested guided batch. `proposal_exhausted` does not imply that200million expensive physical evaluations occurred. The last complete checkpoint and all failed-selection diagnostics must be retained.

Public `scan --resume` is deliberately disabled for every method: a checkpoint alone may omit completed calls in a partial journal. This portable tool supports **only** a strong-reference run that stopped specifically on cap exhaustion, with its entire journal exactly equal to the complete checkpoint. It rejects partial suffixes, arbitrary errors, modern/historical runs and skipped cap steps. There is no general modern partial-run recovery in this release.

After a run such as `runs/strong-s0` reports `proposal_exhausted`, prepare the next cap without evaluating physics:

```sh
python tools/benchmark.py prepare-cap --source-run runs/strong-s0 --output runs/strong-s0-cap1b --next-cap 1000000000 --bundle runs/strong-s0-cap1b-proof
```

It requires the same frozen scientific sources, contiguous complete boundary, exact seed/settings/physics identity, checksum and explicit exhausted-cap diagnostics. It creates a separate checkpoint/settings/proof bundle and changes only `identity.settings.max_proposals` plus its envelope checksum. Original source files and the predecessor run are untouched.

Execute that proof using the identical backend configuration/binaries:

```sh
python tools/benchmark.py resume-cap --proof runs/strong-s0-cap1b-proof/proof.json --backend configs/backend.local.json
```

An optional `--cache runs/truth.sqlite` must be explicit. The inherited prefix counts once with no new physical calls. The unchanged packaged typed guard checks the inherited journal and carries its error streak; all new returns use after-journal callbacks. Known MO numerical errors remain failures in the budget and are not retried. No failed historical label is converted to a successful point.

If the1b run again exhausts at an exact boundary, the next and final allowed step is5b:

```sh
python tools/benchmark.py prepare-cap --source-run runs/strong-s0-cap1b --output runs/strong-s0-cap5b --next-cap 5000000000 --bundle runs/strong-s0-cap5b-proof
python tools/benchmark.py resume-cap --proof runs/strong-s0-cap5b-proof/proof.json --backend configs/backend.local.json
```

A5b exhaustion stays an incomplete comparison; no further automatic step or random replacement exists. The finite tail chunk can change proposal/acceptance RNG consumption, so future trajectory equality across different caps is not claimed. Retain failed-selection training/proposal costs separately. Count the authoritative continuation's full inherited+new journal **once**, not predecessor targets plus continuation targets.

These portable controls were validated with meaningful invented backends and equivalence/corruption tests. The research workspace's analogous recovery was used in real calculations; this packaging release did **not** execute a new real full-cap scan with this portable wrapper. `validation_report.json` distinguishes software validation from physical evidence. If the tool refuses a journal/proof, preserve it and diagnose; do not edit records, delete failures or relax the checks to continue.
