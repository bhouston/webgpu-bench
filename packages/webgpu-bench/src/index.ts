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
} from './types.ts';
export { computeStats, isBestStable, isThrottled, tQuantile975 } from './stats.ts';
export type { BestStableOptions } from './stats.ts';
export { DEFAULT_SAMPLING, runSampling, recordSample, roundIsThrottled, resolveSamplingConfig } from './sampling.ts';
export type { SamplingConfig, Sampleable, SampleState, SampleStateSnapshot, RunSamplingOptions } from './sampling.ts';
export { DEFAULT_MEASUREMENT, KernelSampler } from './gpu/benchmarkRunner.ts';
export type { MeasurementConfig, KernelHarness, WorkKnob } from './gpu/benchmarkRunner.ts';
export { generateMatVecData, padToMultipleOf4, mulberry32 } from './data/generate.ts';
export { acquireGpuContext } from './gpu/context.ts';
export type { GpuContext } from './gpu/context.ts';
