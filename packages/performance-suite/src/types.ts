/** Summary statistics computed over a set of timed measurements. */
export interface Stats {
  mean: number;
  min: number;
  max: number;
  median: number;
  stddev: number;
}

export type TimingMethod = 'gpu-timestamp' | 'cpu-wallclock';

export type BenchmarkCategory = 'dtype' | 'layout' | 'workgroup' | 'multilayer' | 'bandwidth' | 'compute';

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
  /** Logical problem size actually benchmarked (after padding to a multiple of 4). */
  rows: number;
  cols: number;
  /** Number of GPU dispatches batched into each of the 10 timed measurements. */
  innerIterations: number;
  /** Per-op time (ms) for each of the timed measurement runs (after warmup). */
  timesMs: number[];
  stats?: Stats;
  /** GFLOP/s throughput, computed from stats.mean when available. */
  gflops?: number;
  /** Achieved memory bandwidth in GB/s, computed from stats.mean when available. */
  gbps?: number;
  timingMethod: TimingMethod;
}

export interface SuiteOptions {
  /** Matrix rows (output features). Padded up to a multiple of 4. Default 4096. */
  rows?: number;
  /** Matrix cols (input features). Padded up to a multiple of 4. Default 4096. */
  cols?: number;
  /** Target wall-clock duration per timed measurement, in ms. Default 300. */
  targetMs?: number;
  /** Number of warmup measurements before the 10 timed ones. Default 3. */
  warmups?: number;
  /** Number of timed measurements per benchmark. Default 10. */
  runs?: number;
  /** Layer sizes (rows per layer) for the multi-layer MLP benchmark. */
  mlpLayers?: number[];
  /** Number of GPU threads (invocations) launched by the raw-FLOPS compute benchmarks. Default 1,048,576. */
  computeThreads?: number;
  /** Trip count of the in-shader FMA loop each thread runs in the raw-FLOPS compute benchmarks. Default 128. */
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
