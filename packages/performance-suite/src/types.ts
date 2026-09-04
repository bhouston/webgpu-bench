/** Summary statistics computed over a set of timed measurements. */
export interface Stats {
  mean: number;
  min: number;
  max: number;
  median: number;
  stddev: number;
  /** Half-width of the 95% confidence interval on the mean (Student-t); Infinity with a single sample. */
  ci95: number;
}

/**
 * Why adaptive sampling stopped: `converged` once the 95% CI on the mean was
 * within the precision target, `max-runs` when the cap was hit first (the
 * timings were too noisy to converge — treat the row with some suspicion).
 */
export type SamplingStopReason = 'converged' | 'max-runs';

export type TimingMethod = 'gpu-timestamp' | 'cpu-wallclock';

/** `bandwidth`: streams memory with ~no compute. `compute`: pure ALU with ~no memory traffic. */
export type BenchmarkCategory = 'bandwidth' | 'compute';

export type BenchmarkStatus = 'running' | 'ok' | 'skipped' | 'error';

/**
 * One row of the results table. `runSuite` yields a fresh object for a given
 * `id` each time its status changes (running -> ok/skipped/error) so the UI
 * can render progress incrementally.
 */
export interface BenchmarkResult {
  id: string;
  label: string;
  description: string;
  category: BenchmarkCategory;
  status: BenchmarkStatus;
  message?: string;
  /**
   * Logical problem size actually benchmarked. For `bandwidth` kernels:
   * the streamed buffer's rows/cols (after padding to a multiple of 4).
   * For `compute` kernels: thread count / in-shader FMA loop trip count.
   */
  rows: number;
  cols: number;
  /** Number of GPU dispatches batched into each timed measurement. */
  innerIterations: number;
  /** Per-op time (ms) for each of the timed measurement runs (after warmup). Length varies: sampling is adaptive. */
  timesMs: number[];
  stats?: Stats;
  /** Set for `ok` rows: whether sampling stopped because the timings converged or because it hit `maxRuns`. */
  stopReason?: SamplingStopReason;
  /** GFLOP/s throughput, computed from stats.median when available. */
  gflops?: number;
  /** Achieved memory bandwidth in GB/s, computed from stats.median when available. */
  gbps?: number;
  timingMethod: TimingMethod;
}

export interface SuiteOptions {
  /** Streamed buffer rows (for the bandwidth benchmarks). Padded up to a multiple of 4. Default 4096. */
  rows?: number;
  /** Streamed buffer cols (for the bandwidth benchmarks). Padded up to a multiple of 4. Default 4096. */
  cols?: number;
  /** Target wall-clock duration per timed measurement, in ms. Default 300. */
  targetMs?: number;
  /** Number of discarded warmup measurements before timing starts (the calibration pass is a further untimed run). Default 1. */
  warmups?: number;
  /** Fewest timed measurements per benchmark before the convergence check can stop sampling. Default 3. */
  minRuns?: number;
  /** Most timed measurements per benchmark, reached only if the timings never converge. Default 10. */
  maxRuns?: number;
  /**
   * Convergence target: sampling stops once the 95% confidence-interval
   * half-width on the mean is at most this fraction of the mean. Default 0.03
   * (±3%). Lower is stricter and takes more runs.
   */
  precision?: number;
  /** Number of GPU threads (invocations) launched by the raw-FLOPS compute benchmarks. Default 1,048,576. */
  computeThreads?: number;
  /** Trip count of the in-shader loop each thread runs in the raw-FLOPS compute benchmarks. Overrides every kernel's own default (256–1024, chosen so each dispatch takes several ms). */
  computeIterations?: number;
  /** Called once with the resolved GPU device/adapter info before benchmarks start. */
  onDeviceInfo?: (info: DeviceInfo) => void;
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
