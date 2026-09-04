export { runSuite } from './suite.ts';
export type {
  BenchmarkResult,
  SuiteOptions,
  DeviceInfo,
  Stats,
  BenchmarkCategory,
  BenchmarkStatus,
  TimingMethod,
  SamplingStopReason,
} from './types.ts';
export { computeStats, hasConverged, tQuantile975 } from './stats.ts';
export type { ConvergenceOptions } from './stats.ts';
export { DEFAULT_SAMPLING } from './gpu/benchmarkRunner.ts';
export type { SamplingConfig } from './gpu/benchmarkRunner.ts';
export { generateMatVecData, padToMultipleOf4, mulberry32 } from './data/generate.ts';
export { acquireGpuContext } from './gpu/context.ts';
export type { GpuContext } from './gpu/context.ts';
