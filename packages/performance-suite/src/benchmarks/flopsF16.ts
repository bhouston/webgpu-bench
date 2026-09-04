import type { GpuContext } from '../gpu/context.ts';
import { runFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { flopsF16ScalarWgsl } from '../shaders/flopsF16Scalar.ts';
import { flopsF16Vec4Wgsl } from '../shaders/flopsF16Vec4.ts';
import { flopsF16Mat4Wgsl } from '../shaders/flopsF16Mat4.ts';
import type { BenchmarkResult } from '../types.ts';

/** Raw fp16 FLOPS: scalar FMA chain entirely in f16, no vectors or matrices. */
export function benchmarkFlopsF16Scalar(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<BenchmarkResult> {
  return runFlopsBenchmark(
    ctx,
    {
      id: 'flops-f16-scalar',
      label: 'fp16 raw FLOPS',
      description: 'Same shape as the fp32 raw-FLOPS test, but every operand and the accumulator are f16, so the FMA chain runs entirely in half precision.',
      wgsl: flopsF16ScalarWgsl,
      flopsPerIteration: 2,
      requiresF16: true,
    },
    harness,
  );
}

/** fp16 Vec SIMD FLOPS: the same f16 FMA chain, but on a vec4<f16> register. */
export function benchmarkFlopsF16Vec4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<BenchmarkResult> {
  return runFlopsBenchmark(
    ctx,
    {
      id: 'flops-f16-vec4',
      label: 'fp16 Vec SIMD FLOPS',
      description: 'Same f16 FMA chain as the raw fp16 test, but held in a vec4<f16> register — every multiply/add is a 4-wide half-precision SIMD op.',
      wgsl: flopsF16Vec4Wgsl,
      flopsPerIteration: 8,
      requiresF16: true,
    },
    harness,
  );
}

/** fp16 Mat SIMD FLOPS: x = m*x + c chained with a mat4x4<f16>. */
export function benchmarkFlopsF16Mat4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<BenchmarkResult> {
  return runFlopsBenchmark(
    ctx,
    {
      id: 'flops-f16-mat4',
      label: 'fp16 Mat SIMD FLOPS',
      description: 'Same bounded x = m * x + c recurrence as the fp32 mat4 test, but m, c, and x are all f16, so the mat4x4 x vec4 multiply runs entirely in half precision.',
      wgsl: flopsF16Mat4Wgsl,
      flopsPerIteration: 32,
      requiresF16: true,
    },
    harness,
  );
}
