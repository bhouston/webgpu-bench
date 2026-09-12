import type { BenchmarkDefinition } from './benchmarks/common.ts';

/** Summary statistics computed over the kept (non-throttled) timed measurements. */
export interface Stats {
  mean: number;
  /** The best (fastest) kept measurement — the headline number, see `BenchmarkResult.metricValue`. */
  min: number;
  max: number;
  median: number;
  stddev: number;
  /** Half-width of the 95% confidence interval on the mean (Student-t); Infinity with a single sample. */
  ci95: number;
}

/**
 * Why sampling stopped for a benchmark:
 * - `converged`: the best run stopped improving (see `SuiteOptions.stableRounds`).
 * - `max-rounds`: the per-benchmark cap was hit — the best was still improving, or the benchmark
 *   kept producing throttled runs on its own while the rest of the suite was fine.
 * - `throttled`: the device kept coming back throttled even after the suite's cooldown pauses
 *   were exhausted, so the best run may understate the device (treat the row with suspicion).
 */
export type SamplingStopReason = 'converged' | 'max-rounds' | 'throttled';

export type TimingMethod = 'gpu-timestamp' | 'cpu-wallclock';

/** `bandwidth`: streams memory with ~no compute. `compute`: pure ALU with ~no memory traffic. */
export type BenchmarkCategory = 'bandwidth' | 'compute';

export type BenchmarkStatus = 'running' | 'ok' | 'skipped' | 'error';

/**
 * Identifies what a benchmark's throughput number counts. `key` is a
 * JSON-safe id shared by every benchmark that measures the same kind of
 * thing (so it's stable to key off, not unique per-benchmark); `unit` is
 * the short symbol shown as `<unit>/s` (FLOP, OP, B); `name` is a short
 * friendly label for display (e.g. in a tooltip).
 */
export interface MetricDef {
  key: string;
  unit: string;
  name: string;
}

/**
 * One row of the results table. `runSuite` yields a fresh object for a given
 * `id` every time the row changes — when the benchmark starts (`running`),
 * after every sample it takes (still `running`, with the partial stats so
 * far), and when it finishes (ok/skipped/error) — so the UI can render
 * progress incrementally.
 *
 * Pure measurement, no identity/display metadata: label, description, WGSL
 * source, and the metric definition all live on the matching `id`'s
 * `BenchmarkDefinition` instead, so a UI joins a result onto its definition
 * rather than getting them handed twice.
 */
export interface BenchmarkResult {
  id: string;
  category: BenchmarkCategory;
  status: BenchmarkStatus;
  message?: string;
  /**
   * Logical problem size actually benchmarked. For `bandwidth` kernels:
   * the streamed buffer's rows/cols (after padding to a multiple of 4).
   * For `compute` kernels: thread count / in-shader loop trip count (as
   * calibrated for this GPU, see `SuiteOptions.targetDispatchMs`).
   */
  rows: number;
  cols: number;
  /** Number of GPU dispatches batched into each timed measurement. */
  innerIterations: number;
  /** Per-op time (ms) for each kept timed measurement, in the order taken. Length varies: sampling is adaptive. */
  timesMs: number[];
  /**
   * Per-op time (ms) for measurements that were discarded as throttled —
   * more than `throttleThreshold` slower than the best run seen so far. Kept for diagnostics; never feeds `stats`.
   */
  throttledMs: number[];
  /** Summary over `timesMs`. `stats.min` is the best run. */
  stats?: Stats;
  /** Set for `ok` rows: see `SamplingStopReason`. */
  stopReason?: SamplingStopReason;
  /** Peak `<metric.unit>/s`, computed from the best run (`stats.min`). */
  metricValue?: number;
  timingMethod: TimingMethod;
}

/** Suite-level progress events (round boundaries and thermal cooldown pauses). */
export type SuiteProgressEvent =
  | {
      type: 'round';
      round: number;
      active: number;
      /**
       * Estimated measurements still needed across every active benchmark:
       * each still-active benchmark contributes `max(minRounds + stableRounds
       * - <its kept measurements so far>, 1)`. Recomputed fresh every round
       * from each benchmark's own progress (not a fixed total), so it shrinks
       * as benchmarks converge and never has to be corrected upward the way a
       * naive "total rounds" guess does — a UI can turn it into an ETA via
       * `elapsedMs / (unitsCompleted so far)`.
       */
      estimatedRemainingUnits: number;
    }
  | { type: 'cooldown'; attempt: number; maxAttempts: number; ms: number; throttledIds: string[] }
  | { type: 'throttle-abort'; throttledIds: string[] };

export interface SuiteOptions {
  /**
   * Which benchmarks to run — this package's own `BENCHMARKS` by default.
   * Pass a filtered subset, your own `BenchmarkDefinition`s, or a mix of
   * both; `runSuite` doesn't otherwise know or care what a definition
   * measures.
   */
  benchmarks?: readonly BenchmarkDefinition[];
  /** Streamed buffer rows (for the bandwidth benchmarks). Padded up to a multiple of 4. Default 4096. */
  rows?: number;
  /** Streamed buffer cols (for the bandwidth benchmarks). Padded up to a multiple of 4. Default 4096. */
  cols?: number;
  /**
   * Target duration of a single timed measurement, in ms. Sets how many
   * dispatches get batched per measurement. Kept short (default 100) so the
   * suite's total GPU-busy time — and therefore heat — stays low.
   */
  targetMs?: number;
  /**
   * Target duration of a single GPU dispatch, in ms. A dispatch can't be
   * preempted, so a long one freezes the display for its whole duration;
   * the raw-FLOPS kernels have their loop trip count calibrated (from a tiny
   * first probe upward, never more than 4x per step) so one dispatch lands
   * near this on whatever GPU is running. Default 10.
   */
  targetDispatchMs?: number;
  /** Number of discarded warmup measurements per benchmark before timing starts (the calibration pass is a further untimed run). Default 1. */
  warmups?: number;
  /** Fewest kept measurements per benchmark before it can be declared converged. Default 3. */
  minRounds?: number;
  /** Most kept measurements per benchmark, reached only if the best run keeps improving. Default 10. */
  maxRounds?: number;
  /**
   * A benchmark is converged once its best run has not improved by more than
   * `improvementTolerance` over this many consecutive kept measurements.
   * Default 2.
   */
  stableRounds?: number;
  /** Relative improvement of the best run that counts as "still improving" for `stableRounds`. Default 0.01 (1%). */
  improvementTolerance?: number;
  /**
   * A measurement more than this fraction slower than the benchmark's best
   * run is classified as thermally throttled and discarded. Default 0.20
   * (20%): real throttling is 20–50%, while ordinary wall-clock jitter on
   * Safari runs to ~15%.
   */
  throttleThreshold?: number;
  /** Idle gap between consecutive measurements, in ms, so the GPU duty-cycles instead of running flat out. Default 100. */
  idleMs?: number;
  /** How long the whole suite pauses when it detects throttling, in ms. Default 3000. */
  cooldownMs?: number;
  /** How many cooldown pauses to attempt before giving up and reporting the still-unconverged benchmarks as `throttled`. Default 3. */
  maxCooldowns?: number;
  /**
   * Suite-wide throttle trigger: a cooldown starts when at least this
   * fraction (and at least two) of the benchmarks measured in a round come
   * back throttled. Default 0.5.
   */
  throttledFraction?: number;
  /** Number of GPU threads (invocations) launched by the raw-FLOPS compute benchmarks. Default 1,048,576. */
  computeThreads?: number;
  /**
   * Pins the trip count of the in-shader loop in the raw-FLOPS compute
   * benchmarks, disabling per-dispatch calibration. Left unset, each kernel's
   * default (256–1024) is the *ceiling* and the sampler picks the count that
   * makes one dispatch take ~`targetDispatchMs`.
   */
  computeIterations?: number;
  /** Called once with the resolved GPU device/adapter info before benchmarks start. */
  onDeviceInfo?: (info: DeviceInfo) => void;
  /** Called at round boundaries and when the suite pauses for a thermal cooldown. */
  onProgress?: (event: SuiteProgressEvent) => void;
}

export interface DeviceInfo {
  vendor?: string;
  architecture?: string;
  description?: string;
  features: string[];
  limits: Record<string, number>;
  supportsF16: boolean;
  supportsI8Dot: boolean;
  supportsTimestampQuery: boolean;
}
