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

| Step | Idea from brainstorm                      | Status                 | Evidence / decision                                                              |
| ---- | ----------------------------------------- | ---------------------- | -------------------------------------------------------------------------------- |
| 0    | Runtime breakdown and baseline            | Complete               | 206 browser tests pass; profiling harness added.                                 |
| 1    | Shorter precision-aware measurements (#1) | Rejected for now       | 20 ms increased cooldowns, runtime and score instability in both browsers.       |
| 2    | Work-proportional idle gaps (#2)          | Deferred               | 50 ms improved runtime but failed the WebKit score screen.                       |
| 3    | Share empty-submit overhead probes (#4)   | Rejected               | Runtime benefit was marginal; one WebKit score shifted beyond the screen.        |
| 4    | Async pipeline preparation (#6)           | Retained for stability | Fully compiled pipelines before calibration; no demonstrated warm-run speedup.   |
| 5    | Reuse calibration/warmup work (#3)        | Pending                | Avoid selecting headline samples based on favorable calibration timings.         |
| 6    | Encode during idle (#8)                   | Pending                | Evaluate remaining CPU encoding cost and resource hazards.                       |
| 7    | Cache calibration hints (#5)              | Pending                | First-run versus repeat-run benefit must be explicit.                            |
| 8    | Share immutable buffers (#7)              | Pending                | Check remaining setup cost and cache effects.                                    |
| 9    | Statistical stopping (#9)                 | Pending                | Existing best-of-N stopping already adapts; mean CI is not a CI for the minimum. |
| 10   | Concurrent kernels (#10)                  | Rejected by design     | Would measure contention rather than isolated kernel ceilings.                   |

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

### Step 1 — shorter measurements: rejected

ABBA comparison, 11 production-size workloads, two runs per setting/browser,
100 ms reference versus 20 ms candidate; other settings identical.

| Browser  | Reference median | Candidate median | Cooldowns per reference / candidate run |
| -------- | ---------------: | ---------------: | --------------------------------------- |
| WebKit   |          8.030 s |         15.557 s | 0 / 3                                   |
| Chromium |          9.065 s |         15.980 s | 0 / 3                                   |

The shorter runs spent less time doing GPU work but their noisier measurements
triggered cooldowns. In WebKit, 10 of 11 median scores shifted by more than 5%,
and some candidate runs differed in their eventual timing method. A cooldown is
a scheduler classification, not direct evidence of actual thermal throttling.
No default or test workload was changed. Adaptive shortening needs a reliable
precision/timer-method eligibility gate first; a blanket 20 ms default is rejected.
Raw reports: `short-webkit.json`, `short-chromium.json` in the temporary results directory.

### Step 2 — shorter idle gaps: defer changing the default

A 50 ms gap versus 100 ms reduced the 11-kernel ABBA median from 8.616 to 6.671 s
in WebKit (-22.6%) and 8.360 to 7.293 s in Chromium (-12.8%). No cooldowns fired.
Chromium median score shifts stayed within 2.85%. WebKit had a large FP16 outlier
and layout/reduction differences near 5%.

A three-pair WebKit follow-up on the outliers removed the FP16 anomaly, but
layout-soa still shifted -7.02%. Thus the speedup is real, but score equivalence
has not passed the agreed screen. Keep the existing configurable `idleMs` and
100 ms default. A work-proportional policy remains deferred, rather than assuming
these constant-gap results justify changing rest on every device.
Reports: `idle-webkit.json`, `idle-chromium.json`, `idle-confirm-webkit.json`.

### Step 3 — shared overhead probes: trial reverted

Implemented a device-local estimate refreshed every 16 calibrations, with a unit
test verifying the reduced empty-submit count and refresh. All 35 unit tests,
type checking and lint passed. The per-sample timestamp cross-check was unchanged.

ABBA medians: WebKit 8.697 to 8.668 s (-0.3%); Chromium 8.624 to 8.435 s (-2.2%).
The observed end-to-end gain is small relative to run variation. WebKit's
layout-soa score shifted -9.77%, while the other ten median shifts were within 1%.
This does not prove the cache caused that outlier, but it fails the acceptance
screen and does not justify extra timer-state complexity. Reverted the trial
implementation and its cache-specific test. Reports: `overhead-*.json`.

### Step 4 — asynchronous pipeline creation: retained, not counted as a speedup

Pipeline creation now awaits `createComputePipelineAsync`; shader-module and async
pipeline validation failures retain the benchmark label. Error scopes are popped
before awaiting, avoiding accidental nesting across asynchronous preparations.
This ensures calibration begins with a compiled pipeline. No concurrent GPU work
or parallel benchmark preparation was introduced.

All 206 browser tests and 36 unit tests passed; type checking and lint passed
(with the same two existing CLI lint warnings). Browser spans: Chromium 74.92 s,
WebKit 67.89 s. Catalog smoke: 58.12 / 51.28 s; responsiveness: 15.41 / 15.46 s.
These single test runs are slower than baseline; there is **no demonstrated
end-to-end speedup** from this change.

Warm ABBA production-size profiles were effectively unchanged: WebKit 8.469 to
8.438 s; Chromium 8.753 to 8.732 s. WebKit score shifts were within 4.24%; Chromium
was within 1% except layout-soa (-5.95%). An unchanged-code three-pair control
showed layout-soa run-to-run variation up to 8.17%, and a 4.42% apparent runtime
difference despite identical code. The layout result therefore remains noisy;
this is not proof of tight cross-device score equivalence. Retained for explicit
compilation completion and error handling, not credited as a runtime win.

Bounded parallel preparation is deferred: measured warm setup is roughly 1% of
total runtime, and custom benchmark preparation can depend on serial execution.
Reports: `async-*.json`, `async-tests.json`, `unchanged-chromium.json`.
