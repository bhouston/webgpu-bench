import type { GpuContext } from '../gpu/context.ts';
import { prepareFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { OPS_METRIC } from './common.ts';
import { flopsI8ScalarWgsl } from '../shaders/flopsI8Scalar.ts';
import { flopsI8Vec4Wgsl } from '../shaders/flopsI8Vec4.ts';
import { flopsI8Mat4Wgsl } from '../shaders/flopsI8Mat4.ts';
import { flopsI8Dp4aWgsl } from '../shaders/flopsI8Dp4a.ts';
import { flopsI8MatvecWgsl } from '../shaders/flopsI8Matvec.ts';
import { flopsI8MatvecDp4aWgsl } from '../shaders/flopsI8MatvecDp4a.ts';
import type { PreparedBenchmark } from './common.ts';

/** Raw int8-range FLOPS: eight independent scalar i32 multiply-add chains per thread, unrolled 4x. */
export function prepareFlopsI8Scalar(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-scalar',
      label: 'int8-range scalar FMA FLOPS',
      description:
        'Same eight independent, 4x-unrolled multiply-add chains as the fp32 scalar test, but on i32 (WGSL has no first-class i8 type). Measures integer multiply-add throughput; overflow wraps.',
      wgsl: flopsI8ScalarWgsl,
      metric: OPS_METRIC,
      flopsPerIteration: 64,
      defaultIterations: 256,
    },
    harness,
  );
}

/** int8-range vec4 FLOPS: one integer FMA chain on a vec4<i32> register (4 independent lanes). */
export function prepareFlopsI8Vec4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-vec4',
      label: 'int8-range vec4 FLOPS',
      description:
        'Same single multiply-add chain as the fp32 vec4 test, but held in a vec4<i32> register: 4 independent integer lanes per step.',
      wgsl: flopsI8Vec4Wgsl,
      metric: OPS_METRIC,
      flopsPerIteration: 8,
      defaultIterations: 1024,
    },
    harness,
  );
}

/** int8-range mat4 FLOPS: a 4x4 integer matvec emulated via four dot(vec4<i32>) calls (WGSL has no mat4x4<i32>). */
export function prepareFlopsI8Mat4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-mat4',
      label: 'int8-range mat4 FLOPS',
      description:
        'x = m * x + c chained in a register, where m is a 4x4 integer matrix emulated as four vec4<i32> rows combined with dot() (WGSL has no mat4x4<i32>): the same 4 dot products a real int4x4 multiply compiles to.',
      wgsl: flopsI8Mat4Wgsl,
      metric: OPS_METRIC,
      flopsPerIteration: 32,
      defaultIterations: 1024,
    },
    harness,
  );
}

/** int8 register-resident matvec tile with weights unpacked to vec4<i32>, two integer dot() calls per output row. */
export function prepareFlopsI8Matvec(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-matvec',
      label: 'int8 matvec FLOPS (i32 dot)',
      description:
        'Same register-resident 4x8 matvec tile as the fp32 matvec test, with int8-range weights and inputs held unpacked as vec4<i32> and accumulated with integer dot(). The no-extension int8 path: what dot4I8Packed is competing against.',
      wgsl: flopsI8MatvecWgsl,
      metric: OPS_METRIC,
      flopsPerIteration: 64,
      defaultIterations: 512,
    },
    harness,
  );
}

/** int8 register-resident matvec tile on packed u32 words, two dot4I8Packed calls per output row. */
export function prepareFlopsI8MatvecDp4a(
  ctx: GpuContext,
  harness: FlopsHarnessConfig = {},
): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-matvec-dp4a',
      label: 'int8 matvec FLOPS (dot4I8Packed)',
      description:
        "Same 4x8 matvec tile, but weights and inputs stay packed four int8 lanes per u32 and each 4-wide dot product is one dot4I8Packed call from the packed_4x8_integer_dot_product extension. On GPUs without a native dp4a instruction this runs the extension's polyfill.",
      wgsl: flopsI8MatvecDp4aWgsl,
      metric: OPS_METRIC,
      flopsPerIteration: 64,
      defaultIterations: 512,
      requiresI8Dot: true,
    },
    harness,
  );
}

/** int8 packed-dot-product FLOPS: dot4I8Packed from the packed_4x8_integer_dot_product extension, run in a tight accumulation loop. */
export function prepareFlopsI8Dp4a(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-dp4a',
      label: 'int8 dot4I8Packed FLOPS',
      description:
        "One thread per lane, accumulating dot4I8Packed(a, b) — the packed_4x8_integer_dot_product extension's 4-wide int8 dot-product instruction — in a tight loop to measure its peak throughput in isolation.",
      wgsl: flopsI8Dp4aWgsl,
      metric: OPS_METRIC,
      flopsPerIteration: 8,
      defaultIterations: 1024,
      requiresI8Dot: true,
    },
    harness,
  );
}
