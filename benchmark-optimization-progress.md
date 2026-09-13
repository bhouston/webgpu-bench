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

| Step | Idea from brainstorm                      | Status                 | Evidence / decision                                                                |
| ---- | ----------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------- |
| 0    | Runtime breakdown and baseline            | Complete               | 206 browser tests pass; profiling harness added.                                   |
| 1    | Shorter precision-aware measurements (#1) | Rejected for now       | 20 ms increased cooldowns, runtime and score instability in both browsers.         |
| 2    | Work-proportional idle gaps (#2)          | Deferred               | 50 ms improved runtime but failed the WebKit score screen.                         |
| 3    | Share empty-submit overhead probes (#4)   | Rejected               | Runtime benefit was marginal; one WebKit score shifted beyond the screen.          |
| 4    | Async pipeline preparation (#6)           | Retained for stability | Fully compiled pipelines before calibration; no demonstrated warm-run speedup.     |
| 5    | Reuse calibration/warmup work (#3)        | Deferred               | Removing warmup saves 11–17%, but WebKit score equivalence remains inconclusive.   |
| 6    | Encode during idle (#8)                   | Deferred               | WebKit encoding consumes under 1% of representative suite time.                    |
| 7    | Cache calibration hints (#5)              | Deferred               | No first-run benefit; stale hints need bounded device-local validation.            |
| 8    | Share immutable buffers (#7)              | Trial rejected         | Lazy fixture saves setup but not consistent end-to-end time; GPU sharing deferred. |
| 9    | Statistical stopping (#9)                 | Pending                | Existing best-of-N stopping already adapts; mean CI is not a CI for the minimum.   |
| 10   | Concurrent kernels (#10)                  | Rejected by design     | Would measure contention rather than isolated kernel ceilings.                     |

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

### Step 5 — reuse warmup work: promising, deferred

Using `warmups: 0` after the existing calibration reduced WebKit median runtime
from 8.909 to 7.387 s (-17.1%) and Chromium from 8.570 to 7.586 s (-11.5%).
No cooldowns occurred; Chromium score shifts stayed within 3.16%. WebKit layout-soa
shifted -6.20%, with substantial reference variation, and workgroup-64 shifted
-4.47%. This does not establish a regression, but also does not establish equivalence.
The default remains one discarded warmup. A future conditional policy could credit
only a final calibration probe at the actual final batch size as warmup, with an
explicit warmup override honored. Calibration timings must never become selected
headline samples. Reports: `warmup-webkit.json`, `warmup-chromium.json`.

### Step 6 — encode during idle: defer after profiling

Added nested `encodeMs` instrumentation to the opt-in profiler. In two WebKit
runs of unchanged code, kernel encoding consumed 80 / 79 ms out of 8.788 / 9.048 s
(0.91% / 0.87%). This measures the kernel encode callback, not all driver work;
coarse clocks and instrumentation overhead limit precision. Even hiding this
entire measured cost would save under 1%. Moving commands earlier would also
require preserving shared timer/query lifetimes, single-use command buffers and
custom callback ordering. No production scheduling change is justified by this
profile. Reports: `encode-webkit.json`, `encode-chromium.json`.

### Step 7 — cache calibration hints: defer on design grounds

Each `runSuite` acquires a device and destroys it at completion. A device-local
cache therefore cannot accelerate a later suite without changing that lifecycle;
a persistent cache needs identity and invalidation for browser, GPU, shader,
workload size and power/thermal state. It offers no first-run saving. Calibration
currently starts small and caps growth at 4×, the safeguard added after the freeze.
Jumping to a stale batch size would weaken that safeguard. A safe repeat-run
hint design may be useful, but was not implemented or measured in this pass.
This is a design deferral, not evidence that caching can never help.

### Step 8 — avoid unused fixture generation; defer GPU buffer sharing

Rather than sharing mutable GPU resources between kernels, the trial generated
its deterministic CPU matrix/vector fixture on first access. Compute-only filtered
runs avoided generating the unused 4096 × 4096 Float32 matrix (64 MiB). Bandwidth
kernels received the same bytes and the same shared CPU object as before. The full
catalog still needs this fixture, so this is specifically a filtered-run improvement.
GPU buffer sharing is deferred: setup is roughly 1% of the warm representative
suite, and shared allocations could change cache behavior or custom benchmark ownership.

Unit tests check that unused fixtures are not generated and that padding, contents
and identity across preparations remain correct. All 38 unit tests, type checking
and lint passed (the two existing CLI warnings remain). All 206 browser tests also passed before reverting the trial.

Three alternating pairs of a filtered workgroup-64 run: WebKit median 0.328 to
0.306 s (-6.7%), Chromium 0.312 to 0.465 s (+49.1%). Median score shifts were
-3.24% / -0.14%, respectively. Candidate batch sizes sometimes nearly doubled
(about 8,000 versus 4,096 dispatches), overwhelming the saved setup time. One
WebKit candidate also took 0.593 s. The deterministic allocation saving does not
establish a reliable start-to-finish gain, so the fixture trial and its two new unit tests were reverted.
Reports: `lazy-webkit.json`, `lazy-chromium.json`.
