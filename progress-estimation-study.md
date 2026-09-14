# Progress estimation study

Baseline: `2231170`. Existing package-version edits are unrelated and excluded from commits.

## Goal and measurements

Show a numeric estimate only when there is evidence it is within roughly 20%.
Evaluate percentage against **elapsed wall time / final wall time**, rather than
row count; evaluate ETA against actual wall time remaining. Relative error is
`abs(prediction / truth - 1)`. Report the time-weighted share of visible estimates
within 20%, and coverage (visible time / entire run time, including setup).
Completion itself does not count as coverage. Percentage and ETA have separate
eligibility gates: a good percentage near completion does not imply a good ETA.
Replay identical captured runs for each candidate, using only information available
at that instant. Synthetic stress runs exercise the real sampling scheduler;
local GPU runs validate behavior on actual workloads. These are empirical checks,
not a guarantee against future GPU stalls or background-tab suspension.

## Six approaches, in trial order

1. **Correct work accounting.** Exclude setup/status rows and consume remaining work
   after each sample. Removes a deterministic denominator error; units still have unequal cost.
2. **Separate calibration from sampling.** Measure sample wall time directly and
   account for idle gaps. Include completed setup/calibration in elapsed progress,
   but never use them to predict the cost of future samples.
3. **Weight each benchmark separately.** Keep per-benchmark duration histories so
   retiring cheap kernels cannot make an expensive tail look almost finished.
4. **Model the actual stopping rule.** Predict remaining attempts from kept samples,
   stability, discarded samples and configured caps instead of a fixed five-round guess.
5. **Gate using a conservative work envelope.** Consider earliest convergence and
   the retry/round caps; show percentage/ETA only when their possible range is narrow.
   Strong uncertainty protection may sacrifice too much coverage.
6. **Use observed predictability to narrow the envelope.** Compare successive
   forecasts with subsequent work, require timing evidence, and revoke estimates
   on cooldowns, discards, pauses or duration changes. Compare accuracy and coverage
   with the conservative alternative before retaining any relaxation.

## Findings before experiments

Setup rows counted as progress; remaining units were not consumed within rounds;
round-one calibration inflated the sampling rate; earliest stable completion is
three samples with defaults, not five. Cooldowns never invalidated an estimate.
`fraction` was always numeric, giving callers no way to distinguish a trustworthy
estimate from a guess. The repository contains a CLI consumer; browser consumers
of the exported core API also need a documented nullable display API.

## Results and decisions

Experiments and retained changes will be recorded here as they finish.

### Baseline and measurement harness

144 synthetic runs (12 families × 12 seeds), with time-weighted polling between
events: percentage accuracy within 20% **33.40%**, coverage **100%**; ETA accuracy
**11.01%**, coverage **84.47%**. Heavy stress cases deliberately exceed ordinary
workload variability; they are not an estimate of production failure frequency.
The frozen baseline implementation is retained in the replay script.

Added optional effective settings/ordered IDs on round events and sample events
with wall duration excluding calibration/idle. No benchmark scheduling or stopping
behavior changed. Added a fake-clock test confirming those exclusions. All 37
core Node tests pass. Raw captures and reports are in `/tmp/webgpu-bench-progress`.

### Trials 1–3: retain phase-aware, per-benchmark timing

All trials replay the same trace prefixes. Values below are **within-20% accuracy /
coverage**, both percentages. ETA uses unrounded seconds in the new API; the frozen
baseline rounds to whole seconds, as it actually did.

| Trial                                                | Synthetic percentage | Chromium percentage | Synthetic ETA |  Chromium ETA |
| ---------------------------------------------------- | -------------------: | ------------------: | ------------: | ------------: |
| Baseline                                             |          33.40 / 100 |         11.21 / 100 | 11.01 / 84.47 |     0 / 87.46 |
| 1: consume units, exclude setup                      |          25.28 / 100 |          0.98 / 100 | 12.31 / 84.46 |     0 / 87.46 |
| 2: separate sample wall time + idle                  |        46.78 / 62.42 |        7.73 / 59.81 | 17.29 / 62.42 |  0.40 / 59.81 |
| 3: weight each active benchmark, remove retired work |        65.53 / 62.42 |       79.09 / 59.81 | 24.80 / 62.42 | 16.86 / 59.81 |

Trials 1 and 2 alone did not meet the goal; retained their accounting fixes as
part of trial 3's demonstrably better model. WebKit trial 3: percentage **76.85 /
61.88**, versus baseline **12.61 / 100**; ETA **16.03 / 61.88**, versus **0 / 87.69**.
Six runs per browser: two each of representative 11-kernel suites, a single
kernel, and a fixed-six-round suite, at production workload sizes. Calibration
must finish for every scheduled kernel before showing estimates. Percentage now
measures wall-time completion, not how many rows have arrived. Remaining idle gaps
account for round boundaries and shrinking active sets. 39 core Node tests pass.

This is an intermediate foundation: the fixed-five-sample stopping guess still
has large errors. The CLI confidence-gate integration follows the next trials.

### Trial 4: retain stopping-rule prediction

Use effective sampling settings, the exact earliest convergence check, and a
finite probability model for additional improvements (a quiet prior plus each
kernel's observed improvement frequency). The model sums survival probability
until the configured kept-sample cap. It predicts work; it does not alter which
samples the scheduler collects. Retired/error kernels contribute no future work.

| Dataset   | Percentage accuracy / coverage | ETA accuracy / coverage |
| --------- | -----------------------------: | ----------------------: |
| Synthetic |                  70.27 / 62.42 |           37.26 / 62.42 |
| Chromium  |                    100 / 59.81 |           95.14 / 59.81 |
| WebKit    |                  97.09 / 61.88 |           82.67 / 61.88 |

Substantial improvement on both GPU datasets, particularly fixed-round runs.
Synthetic mean percentage error increased from 20.64% to 23.12% despite better
within-20% accuracy: cooldowns and late surprises still make some estimates very
wrong. This motivates the next trial's revocable confidence gate rather than
claiming the point prediction alone solves reliability.

### Trial 5: reject the full retry-cap envelope as the display policy

Bound sample work between earliest convergence and the remaining kept/discarded
attempt budget, with a ±20% duration allowance. Percentage accuracy reached 100%,
but coverage was only **0.77% synthetic, 1.68% Chromium, 2.02% WebKit**. ETA
coverage was zero. This does not include arbitrary future pauses (which have no
finite bound). It is still too pessimistic to be useful; the gate framework is
retained, but this work envelope is replaced in trial 6.

### Trial 6: retain empirical, revocable confidence gates

Require at least two timing observations for every active benchmark. Bound work
between earliest convergence and the predicted work plus a full stability window,
capped by the effective kept-sample limit. Use recent observed duration extrema
with a 10% allowance. Require the entire resulting percentage range to fit within
20% relative error around the point prediction. ETA uses its own relative-error
check against the remaining-time bounds, rather than the total-time bounds.

This is an **empirical envelope**, not a formal 95% confidence interval. Compare
new sample durations with their previous forecasts; a >20% change, invalid timing,
discard, cooldown or hidden-page event withdraws estimates until two later round
numbers. A timer-based freshness check also withdraws a stalled estimate without
waiting for another sample. The policy can recover, and does not force monotonic
percentages that would conceal a revised prediction. Completed setup/calibration
remains credited in elapsed time. No sampling parameters or workloads change.

| Dataset             | Percentage accuracy / coverage | ETA accuracy / coverage |
| ------------------- | -----------------------------: | ----------------------: |
| Synthetic, 144 runs |                  99.88 / 16.50 |              100 / 5.64 |
| Chromium, 6 runs    |                    100 / 32.04 |             100 / 28.50 |
| WebKit, 6 runs      |                  99.56 / 34.64 |             100 / 32.59 |

Percentage mean relative error falls to **3.29%, 0.30%, 0.64%**, respectively.
These are time-weighted metrics for the API getters, not independent statistical
trials: many successive estimates share the same run. The confidence gate trades
coverage for accuracy deliberately. The full-runtime denominator includes setup
and calibration; ordinary short/default suites have less coverage than fixed-six-
round suites. Per-family results and independent follow-up appear below.

The CLI now uses nullable `displayFraction`, prints `Estimating runtime…` while
uncertain, refreshes during event-free waits, and clears the timer on success or
failure. It shows an ETA only when separately eligible, to one decimal place.
It reports 100% only after the generator finishes. The core README documents API
migration and periodic polling for browser consumers outside this repository.

Validation so far: **48 core Node tests, 8 CLI tests, TypeScript build pass**.
Tests cover setup, calibration, effective settings, empty/legacy streams, separate
ETA eligibility, expiry, pause/abort, discards, recovery and invalid/changed timing.
Machine: Apple M3, macOS 27.0 (26A428). Fresh GPU validation follows without tuning
these thresholds against its outcomes.

### Trial 6 follow-up: retain uncertainty per benchmark

The workload breakdown exposed poor coverage in ordinary short suites, and a
recovery boundary could reopen after only one subsequent clean sample. Recovery
now requires two complete later rounds before narrowing uncertainty again.
An individual noisy kernel gets the full remaining retry-cap envelope (and an
attempt/kept-sample rate correction), rather than invalidating the entire suite.
Suite-wide cooldowns and visibility pauses still withdraw the whole estimate.
Quiet, actually converged peers narrow the stability allowance by at most half.
The earliest-work bound now also accounts for termination by the discarded-attempt
cap; reaching that cap can finish earlier than collecting enough kept samples.

On the original six-run datasets, percentage accuracy / coverage improves to
**100 / 39.07 Chromium** and **100 / 37.16 WebKit**. Synthetic accuracy / coverage
is **99.89 / 18.08**, versus the previous gate's **99.88 / 16.50**.
The CLI omits sub-second ETAs and uses one decimal place for longer ETAs.

Extended replay now includes **20 GPU traces**, including two 71-kernel full-catalog
runs and noisy WebKit cooldown cases. API percentage accuracy is **100%** with
**20.26% runtime coverage**; ETA accuracy is **100%** with **13.15% coverage**.
A separate replay of actual TTY formatting and 100 ms refresh cadence gives
**99.94% percentage accuracy / 20.26% coverage**, versus the old display's
**26.12% / 99.68%**. Displayed ETA improves from **4.61% / 91.10%** to
**100% / 11.62%**. The small remaining percentage misses occur in a short noisy
single-kernel run; API accuracy is not a guarantee about rounded, held UI text.

Coverage must not be hidden by the aggregate: representative default suites show
percentages for **11–20%** of runtime; quiet fixed-six-round suites for **54–55%**.
A 60.90 s full Chromium suite has **16.09%** coverage. A 72.56 s full WebKit suite
with three cooldowns has **3.45%** coverage, and a noisy fixed-round WebKit run
shows no estimate. These runs support hiding uncertain estimates, not a claim
that progress can currently be shown throughout normal execution.

The first 12 traces were used for development; eight fresh traces were subsequently
reused while refining the gate, so the final combined result is not a locked
holdout evaluation. An additional Chromium pair with a new shuffle seed, captured
after the final refinement, reached **100% display accuracy / 34.23% coverage**.

All **206 browser tests, 49 core Node tests, and 8 CLI tests pass**. Type checking
and lint pass, with only the two pre-existing CLI function-scoping warnings.
The compressed GPU trace fixture is committed for reproducibility; synthetic
traces can be regenerated from the seeded scheduler harness.

### User-suggested next trial: linear benchmark execution

Evaluate one benchmark to completion before moving to the next. Distinguish an
exact completed-benchmark count from an estimate of time completion: the former
can always be shown honestly, but unequal benchmark costs can still make its
percentage misleading as a time estimate. Compare linear order with round-robin
on runtime, sample results, throttling, accuracy and coverage before deciding
whether changing the default scheduler is justified.

### Exact completed-test fallback: retained

`SuiteProgress.completedBenchmarks` now counts unique terminal results (`ok`,
`skipped`, or `error`), and `benchmarkCount` exposes the total. The CLI shows
`Completed X of Y benchmarks · Estimating runtime…` while wall-time progress is
uncertain, including in non-TTY logs as each result finishes. This provides an
exact, always-available work count without implying that each test costs the same
time. Duplicate terminal rows do not double count. 50 core Node tests pass.

### Linear-order trial: experimental setup

The isolated research variant runs one prepared benchmark through calibration and
sampling to completion, then advances in catalog order. Preparation remains shared
and upfront, with one GPU device for the suite. Sample duration, warmups,
convergence rule, configured caps and the 100 ms inter-sample idle remain the same.
Two consecutive discards of the current kernel trigger its cooldown; a per-kernel
cooldown budget replaces round-robin's cross-kernel thermal consensus. That policy
change is explicit: executing only one kernel cannot supply cross-kernel consensus.
The trial runs only in temporary compiled modules, not in the production scheduler.

Two alternating pairs (ABBA), 11 representative production-size benchmarks,
`maxRounds: 10`, matched seeds within pairs. Chromium results:

- Median runtime: round-robin **8.709 s**, linear **9.151 s** (+5.1%).
- First final result: round-robin **75.7–77.7%** of runtime, linear **6.5–9.1%**.
- Nonzero completed-count coverage: round-robin **22.3–24.3%**, linear **90.9–93.5%**.
- If that count fraction is interpreted as elapsed-time fraction: round-robin
  within-20% accuracy **13.0–15.2%**, linear **87.3–90.6%**.
- No cooldowns or discarded samples. All median score shifts within **3.4%**.

These count-as-time percentages are a secondary screen; `X of Y completed` itself
is exact in either scheduler. Further browser/workload results follow below.

### Linear-order trial: WebKit and the full Chromium catalog

WebKit's two representative ABBA pairs:

- Median runtime **8.485 s round-robin / 9.186 s linear** (+8.3%).
- First result **65.2–77.6% / 9.5%** of runtime.
- Nonzero count coverage **22.4–34.8% / 90.5%**.
- Count-as-time within-20% accuracy **17.2–20.5% / 69.0–84.8%**.
- No discards/cooldowns in either mode. Median `layout-soa` score changed **-9.2%**
  and `f16-fma-mat4` **-13.7%**. Both modes used CPU-wallclock timing, but FP16's
  calibrated batch changed from 42 to 256 inner iterations. Other shifts were
  within 3.4%. This does not isolate which calibration/order effect caused the
  difference, but it does not establish measurement equivalence.

A full 71-kernel Chromium pair is substantially more favorable for count progress:

| Measure                           | Round-robin |   Linear |
| --------------------------------- | ----------: | -------: |
| Total time                        |    58.840 s | 59.002 s |
| First final result (% of runtime) |      75.90% |    1.59% |
| Nonzero count coverage            |      24.13% |   98.39% |
| Count-as-time accuracy within 20% |      23.94% |   98.62% |
| Mean relative count-as-time error |      44.48% |    2.86% |

No discards/cooldowns occurred. Two layout scores shifted **+11.0% and +11.7%**;
this single pair is insufficient to attribute those shifts to the scheduler.
The early completed-count cadence is a real benefit of linear execution.
The representative trials and this first full-catalog trial used a plain 100 ms
sleep in the research variant; the follow-up variant reuses the production
`realSleep` including its browser-idle callback. The production scheduler has
not changed. A deterministic check of the variant also passed serial order,
calibration, idle spacing, consecutive-discard cooldowns, budget exhaustion and
continuation to a healthy next benchmark.

### Linear-order trial: full WebKit catalog and decision

The follow-up full 71-kernel WebKit pair uses the production idle helper in both
variants:

| Measure                            | Round-robin |   Linear |
| ---------------------------------- | ----------: | -------: |
| Total time                         |    68.369 s | 60.671 s |
| First final result (% of runtime)  |      59.98% |    1.72% |
| Nonzero count coverage             |      40.00% |   98.23% |
| Count-as-time accuracy within 20%  |      66.73% |   93.88% |
| Mean relative count-as-time error  |      23.22% |    6.02% |
| Cooldowns / discarded measurements |      3 / 23 |    0 / 0 |

Linear execution improves cadence and avoids cooldowns in this pair. Seven scores
shift more than 5%, including `f16-div` (-49.1%) and `f32-sincos` (-24.3%). Those
two reference rows were flagged `throttled` and used GPU timestamps; their linear
counterparts converged using CPU-wallclock timing with different calibration
batch sizes. Thus the score differences are **not proof that linear execution is
worse**: the reference itself has suspect measurements, and timing-method changes
confound an equivalence comparison. Other layout/atomic scores also change.

**Decision:** keep exact completed-test progress in the production CLI and keep
the linear scheduler as a reproducible research variant for now. It is a strong
candidate for a simpler scheduling design: full-catalog nonzero progress coverage
rises to about 98%, versus 24–40% with round-robin. However, these Mac-only trials
do not establish equivalent benchmark measurements, and linear thermal detection
requires a different policy. No production scheduling change is made merely to
make a progress bar easier to estimate. The count itself remains exact under
both schedulers, independently of whether its fraction matches wall time.

If the product chooses completed tests as its definition of progress, no ETA or
runtime estimator is required for that display. The retained exact-count fallback
already supports that definition. The time estimator remains separately gated
for callers that still want an elapsed-time percentage/ETA.

## Reproduce the research

Run from the repository root. Profiling is opt-in, sequential, uses the local GPU,
and does not submit results to an external service. Raw development captures are
in `/tmp/webgpu-bench-progress`; compressed GPU and linear-trial fixtures are
retained in `scripts/fixtures/`.

```sh
pnpm --filter webgpu-bench-core build

# Re-run the seeded 144-run stress set against the frozen baseline and current API/UI.
node scripts/progress-experiment.mjs --capture synthetic \
  --output /tmp/progress-synthetic.json

# Replay all 20 retained GPU traces (includes full catalogs and cooldown cases).
node scripts/progress-experiment.mjs \
  --input scripts/fixtures/progress-gpu-traces.json.gz \
  --output /tmp/progress-gpu.json

# Collect new traces. --groups also accepts single, fixed and all.
node scripts/progress-experiment.mjs --capture chromium --seed 401 \
  --repeats 1 --groups representative,fixed \
  --traces /tmp/progress-new-traces.json --output /tmp/progress-new.json

# Prepare an isolated linear scheduler using the current compiled kernels.
node scripts/linear-progress-experiment.mjs --prepare /tmp/progress-linear-dist
node scripts/profile-suite.mjs --browser webkit \
  --candidate-root /tmp/progress-linear-dist \
  --candidate '{"maxRounds":10}' --reference-options '{"maxRounds":10}' \
  --repeats 2 --output /tmp/progress-linear-profile.json

# Summarize the preserved full-catalog comparisons without running the GPU.
node scripts/linear-progress-experiment.mjs \
  --report scripts/fixtures/linear-all-chromium.json.gz \
  --report scripts/fixtures/linear-all-webkit.json.gz \
  --output /tmp/progress-linear-summary.json
```

The replay reports both raw API estimates and TTY-style displayed estimates
(whole percentages, one-decimal ETAs of at least one second, event updates plus a
100 ms refresh). Non-TTY logs are historical snapshots rather than a continuously
visible estimate. Coverage integrates time between events; the completion instant
itself is excluded. The synthetic and captured GPU workloads are complementary,
not a claim about how often real users encounter each kind of noise.

## Retained commits

| Commit    | Change                                                                       |
| --------- | ---------------------------------------------------------------------------- |
| `869938f` | Telemetry, frozen baseline and repeatable accuracy/coverage experiments      |
| `ef0ebc9` | Per-benchmark wall-time model, calibration exclusion and work accounting     |
| `cbd9f02` | Actual convergence/configuration-based work prediction                       |
| `c0e5681` | Revocable, separate percentage/ETA confidence gates and CLI integration      |
| `ccdd715` | Per-benchmark uncertainty, recovery/cap corrections and preserved GPU traces |
| `d24f586` | Exact completed-test fallback and linear-scheduler comparison harness        |

Final validation: 206 browser correctness/responsiveness tests, 50 core Node tests,
and 8 CLI tests pass; TypeScript and lint pass with two existing CLI warnings.
The user's pre-existing package-version edits are excluded from all commits.
