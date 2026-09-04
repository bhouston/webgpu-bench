import type { GpuContext } from '../gpu/context.ts';
import { runFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { flopsI8ScalarWgsl } from '../shaders/flopsI8Scalar.ts';
import { flopsI8Vec4Wgsl } from '../shaders/flopsI8Vec4.ts';
import { flopsI8Mat4Wgsl } from '../shaders/flopsI8Mat4.ts';
import { flopsI8Dp4aWgsl } from '../shaders/flopsI8Dp4a.ts';
import type { BenchmarkResult } from '../types.ts';

/** Raw int8-range FLOPS: scalar i32 FMA chain, no vectors or matrices. */
export function benchmarkFlopsI8Scalar(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<BenchmarkResult> {
  return runFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-scalar',
      label: 'int8-range raw FLOPS',
      description: 'One thread per lane, a long chain of scalar i32 multiply-adds held in a register (WGSL has no first-class i8 type). No vectors, no matrices, ~no memory traffic.',
      wgsl: flopsI8ScalarWgsl,
      flopsPerIteration: 2,
    },
    harness,
  );
}

/** int8-range Vec SIMD FLOPS: the same integer FMA chain, but on a vec4<i32> register. */
export function benchmarkFlopsI8Vec4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<BenchmarkResult> {
  return runFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-vec4',
      label: 'int8-range Vec SIMD FLOPS',
      description: 'Same integer FMA chain as the raw int8-range test, but held in a vec4<i32> register — every multiply/add is a 4-wide integer SIMD op.',
      wgsl: flopsI8Vec4Wgsl,
      flopsPerIteration: 8,
    },
    harness,
  );
}

/** int8-range Mat SIMD FLOPS: a 4x4 integer matvec emulated via four dot(vec4<i32>) calls (WGSL has no mat4x4<i32>). */
export function benchmarkFlopsI8Mat4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<BenchmarkResult> {
  return runFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-mat4',
      label: 'int8-range Mat SIMD FLOPS',
      description: 'x = m * x + c chained in a register, where m is a 4x4 integer matrix emulated as four vec4<i32> rows combined with dot() (WGSL has no mat4x4<i32>) — the same 4 dot-products a real int4x4 multiply compiles to.',
      wgsl: flopsI8Mat4Wgsl,
      flopsPerIteration: 32,
    },
    harness,
  );
}

/** int8 packed-dot-product FLOPS: dot4I8Packed from the packed_4x8_integer_dot_product extension, run in a tight accumulation loop. */
export function benchmarkFlopsI8Dp4a(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<BenchmarkResult> {
  return runFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-dp4a',
      label: 'int8 dot4I8Packed FLOPS',
      description: "One thread per lane, accumulating dot4I8Packed(a, b) — the packed_4x8_integer_dot_product extension's 4-wide int8 dot-product instruction — in a tight loop to measure its peak throughput in isolation.",
      wgsl: flopsI8Dp4aWgsl,
      flopsPerIteration: 8,
      requiresI8Dot: true,
    },
    harness,
  );
}
