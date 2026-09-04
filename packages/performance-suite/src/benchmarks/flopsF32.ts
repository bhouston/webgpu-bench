import type { GpuContext } from '../gpu/context.ts';
import { runFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { flopsF32ScalarWgsl } from '../shaders/flopsF32Scalar.ts';
import { flopsF32Vec4Wgsl } from '../shaders/flopsF32Vec4.ts';
import { flopsF32Mat4Wgsl } from '../shaders/flopsF32Mat4.ts';
import type { BenchmarkResult } from '../types.ts';

/** Raw fp32 FLOPS: scalar FMA chain, no vectors or matrices. */
export function benchmarkFlopsF32Scalar(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<BenchmarkResult> {
  return runFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-scalar',
      label: 'fp32 raw FLOPS',
      description: 'One thread per lane, a long chain of scalar f32 fused multiply-adds held in a register. No vectors, no matrices, ~no memory traffic.',
      wgsl: flopsF32ScalarWgsl,
      flopsPerIteration: 2,
    },
    harness,
  );
}

/** fp32 Vec SIMD FLOPS: the same FMA chain, but on a vec4<f32> register. */
export function benchmarkFlopsF32Vec4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<BenchmarkResult> {
  return runFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-vec4',
      label: 'fp32 Vec SIMD FLOPS',
      description: 'Same FMA chain as the raw fp32 test, but held in a vec4<f32> register — every multiply/add is a 4-wide SIMD op.',
      wgsl: flopsF32Vec4Wgsl,
      flopsPerIteration: 8,
    },
    harness,
  );
}

/** fp32 Mat SIMD FLOPS: x = m*x + c chained with a mat4x4<f32>. */
export function benchmarkFlopsF32Mat4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<BenchmarkResult> {
  return runFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-mat4',
      label: 'fp32 Mat SIMD FLOPS',
      description: 'x = m * x + c chained in a register with a mat4x4<f32> (a bounded contraction so it stays numerically stable), exercising full mat4 x vec4 multiplies.',
      wgsl: flopsF32Mat4Wgsl,
      flopsPerIteration: 32,
    },
    harness,
  );
}
