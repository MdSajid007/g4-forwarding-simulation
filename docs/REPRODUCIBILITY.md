# Reproducibility record

Verification date: 2026-10-11 (Asia/Seoul).

The focused release candidate was verified on Windows, AMD64, CPython
3.14.4 using two newly created virtual environments with system site
packages and user site packages excluded. Environment A was installed from
the five direct pinned dependencies. Its complete dependency set was saved
as `requirements-lock.txt`. Environment B was independently installed from
that lock. Public package wheels were obtained through pip from PyPI, with
its download cache available. This was a fresh-environment installation,
not a claim of offline installation or an independent-machine replication.

| Check | Result |
|---|---|
| Dependency consistency, both environments | `pip check` passed |
| Easy-fixture smoke, both environments | 48 delivered; PDR 1.0; 144 transmission attempts |
| Frozen-result reanalysis, both environments | All 90 configuration bindings, nine summaries, and report text verified |
| Focused original suite, environment A | 267 passed, 1 skipped in 94.52 seconds |
| Full G4d replay, environment A | 90 paired rows; both CSVs byte-identical to references; report text identical |
| Full G4d replay, environment B | Same 90 paired rows and reference matches |
| Output-path guard | Writing a replay to the frozen `outputs/` directory was rejected with status 2 |

The skipped test is `test_fading_deterministic_when_enabled`: the easy
configuration disables fading and uses pathloss-only behavior. This skip
does not certify the unexercised fading configuration. Only the selected
G4 and simulator tests were run; the original project's larger legacy,
ML, ns-3, and evidence-tooling suites are outside this release.

The frozen historical experiment has nine scenarios and ten seeds per
scenario. Each seed compares two complete modes, giving 90 paired records
and 180 arm executions per full replay. The gate parameters and scenario
definitions are preserved in source.

The original source tree remains outside this checkout and is not modified.
Public publication was authorized by the authenticated account holder on
2026-10-11. No open-source license is assigned, and copyright ownership and
attribution have not been independently verified; see
[licensing status](../LICENSE_STATUS.md).

## Commands

From the checkout root, the public commands are:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B scripts\smoke.py
.\.venv\Scripts\python.exe -B scripts\verify_results.py
.\.venv\Scripts\python.exe -B -m pytest tests -p no:cacheprovider -q
.\.venv\Scripts\python.exe -B scripts\reproduce_g4d.py
.\.venv\Scripts\python.exe -B scripts\verify_integrity.py
```

The preparation runs used `reproduction/repeat_a` and
`reproduction/repeat_b` rather than the default `reproduction/latest`.
The wrapper permits only destinations inside the ignored `reproduction/`
tree. It calls the preserved experiment without changing its parameters.
Tests may create temporary or ignored run/log artifacts; these are excluded
from the proposed public file set.

## What is reproducible

Each complete replay performed 180 simulation-arm executions: nine
scenarios by ten seeds by two modes. The paired-result CSV and scenario
summary CSV match their original bytes, including outcome and diagnostic
hash fields. The report matches as decoded text; report newline bytes are
not used to claim cross-platform identity.

The outcome remains `NULL/NEGATIVE` with no selector authorization. Neither
successful installation nor matching file hashes establishes physical
realism, Bluetooth Mesh compliance, hardware transfer, or ML effectiveness.

## Limits

- Windows/AMD64/Python 3.14.4 on one host was checked. Other platforms,
  Python versions, and hardware have not been tested.
- The environments are separate but the host and checkout are shared.
- Dependencies are version-pinned; wheel hashes and an offline wheelhouse
  are not supplied. Package installation needs access to packages or a cache.
- Intended retransmission timing is not wired to delayed policy timers.
- SciPy must remain installed for the recorded interval calculations.
- The simulator's radio, MAC, emergency-priority, and energy-proxy limits
  described in the README remain unchanged.
- There is no Git history for the original snapshot. Provenance identifies
  file bytes without establishing authorship or distribution rights.

Machine-readable results and dependency versions are in `VALIDATION.json`.
