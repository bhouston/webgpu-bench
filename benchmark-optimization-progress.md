# Benchmark optimization progress

Baseline: `c9a2ddb` (core and CLI 0.9.2), MacBook Air M3, macOS 27.0.

## Method

Changes are evaluated sequentially and retained improvements are committed separately.
The unrelated `microbenchmark-suggestions.md` is not part of this work.

- Run all browser correctness and responsiveness tests in Chromium and WebKit, sequentially.
- Record suite-test and whole-browser wall time; do not shorten tests just to improve the number.
- Use `scripts/profile-suite.mjs` for matched reference/candidate runs of 11 representative
  workloads at production compute sizes. Alternate order (ABBA), use identical shuffle seeds
  within each pair, and record scores, discarded samples, timing methods, phase costs, and timer gaps.
- Initial score screen: investigate any median per-kernel shift over 5%; do not accept a
  broad timing-policy change solely because correctness tests pass. Repeatability, clock
  calibration and timer-method changes can invalidate a simple percentage comparison.
- Local measurements do not establish equivalence on phones, other GPUs, or other browsers.
- Temporary raw reports live in `/tmp/webgpu-bench-optimization/`; no diagnostics folder is created.

Example (build before profiling; copy the reference build before editing):

```sh
pnpm --filter webgpu-bench-core build
node scripts/profile-suite.mjs --browser webkit \
  --reference /tmp/webgpu-bench-optimization/baseline-dist \
  --candidate '{"targetMs":20}' \
  --output /tmp/webgpu-bench-optimization/short-samples-webkit.json
```

## Sequence and decisions

| Step | Idea from brainstorm                      | Status             | Evidence / decision                                                              |
| ---- | ----------------------------------------- | ------------------ | -------------------------------------------------------------------------------- |
| 0    | Runtime breakdown and baseline            | Running            | Establish correctness, runtime and score comparison procedure.                   |
| 1    | Shorter precision-aware measurements (#1) | Pending            | Compare 20 ms with 100 ms before designing adaptation.                           |
| 2    | Work-proportional idle gaps (#2)          | Pending            | Measure score and throttling effects before changing rest.                       |
| 3    | Share empty-submit overhead probes (#4)   | Pending            | Preserve per-sample timestamp cross-checks.                                      |
| 4    | Async pipeline preparation (#6)           | Pending            | Preserve error reporting and isolate compilation from measurement.               |
| 5    | Reuse calibration/warmup work (#3)        | Pending            | Avoid selecting headline samples based on favorable calibration timings.         |
| 6    | Encode during idle (#8)                   | Pending            | Evaluate remaining CPU encoding cost and resource hazards.                       |
| 7    | Cache calibration hints (#5)              | Pending            | First-run versus repeat-run benefit must be explicit.                            |
| 8    | Share immutable buffers (#7)              | Pending            | Check remaining setup cost and cache effects.                                    |
| 9    | Statistical stopping (#9)                 | Pending            | Existing best-of-N stopping already adapts; mean CI is not a CI for the minimum. |
| 10   | Concurrent kernels (#10)                  | Rejected by design | Would measure contention rather than isolated kernel ceilings.                   |

## Results

### Baseline: `c9a2ddb`

All 206 browser tests passed (103 Chromium, 103 WebKit). Fixed test settings are
unchanged: catalog smoke test at 20 ms with three kept samples, responsiveness
at 50 ms with one round. These are reduced-compute-size tests, not a claim about
production-size measurement equivalence.

| Browser  | Catalog smoke | Responsiveness | Browser test span |
| -------- | ------------: | -------------: | ----------------: |
| Chromium |       53.84 s |        15.36 s |           70.67 s |
| WebKit   |       49.28 s |        15.40 s |           65.85 s |

The profiling harness measures setup, calibration, sampling and total duration
without adding per-probe logging or changing the GPU timer. Production-size score
experiments are separate from the correctness tests above.
