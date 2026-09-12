// The runner — knows nothing about what any benchmark measures.
export { runSuite } from './suite.ts';
export type {
  BenchmarkResult,
  SuiteOptions,
  SuiteProgressEvent,
  DeviceInfo,
  Stats,
  BenchmarkCategory,
  BenchmarkStatus,
  TimingMethod,
  SamplingStopReason,
  MetricDef,
} from './types.ts';
export { computeStats, isBestStable, isThrottled, tQuantile975 } from './stats.ts';
export type { BestStableOptions } from './stats.ts';
export { DEFAULT_SAMPLING, runSampling, recordSample, roundIsThrottled, resolveSamplingConfig } from './sampling.ts';
export type { SamplingConfig, Sampleable, SampleState, SampleStateSnapshot, RunSamplingOptions } from './sampling.ts';
export { DEFAULT_MEASUREMENT, KernelSampler } from './gpu/benchmarkRunner.ts';
export type { MeasurementConfig, KernelHarness, WorkKnob } from './gpu/benchmarkRunner.ts';
export { acquireGpuContext } from './gpu/context.ts';
export type { GpuContext } from './gpu/context.ts';

// The contract a benchmark implements, plus the building blocks
// (`createPipeline`, `prepareKernelBenchmark`, the shared metric defs) for
// writing one — whether it's a new one, or a variant of a built-in.
export type {
  BenchmarkContext,
  BenchmarkDefinition,
  BenchmarkMeta,
  HarnessConfig,
  PreparedBenchmark,
} from './benchmarks/common.ts';
export {
  createPipeline,
  prepareKernelBenchmark,
  metricPerSecond,
  rowFromMeta,
  skippedResult,
  errorResult,
  FLOPS_METRIC,
  OPS_METRIC,
  BYTES_METRIC,
} from './benchmarks/common.ts';
export { generateMatVecData, padToMultipleOf4, mulberry32 } from './data/generate.ts';
export type { GeneratedData } from './data/generate.ts';

// This package's own benchmark definitions (memory bandwidth + raw-FLOPS
// ALU throughput) — the default for `runSuite`, but just one possible
// `BenchmarkDefinition[]` among others; see `SuiteOptions.benchmarks`.
export { BENCHMARKS } from './catalog.ts';

// Completion fraction + ETA from the result stream, shared by every front end.
export { SuiteProgress } from './progress.ts';
