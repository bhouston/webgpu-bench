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
