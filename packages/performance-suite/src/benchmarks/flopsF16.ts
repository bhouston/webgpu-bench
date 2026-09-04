import type { GpuContext } from '../gpu/context.ts';
import { prepareFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { flopsF16ScalarWgsl } from '../shaders/flopsF16Scalar.ts';
import { flopsF16Vec4Wgsl } from '../shaders/flopsF16Vec4.ts';
import { flopsF16Mat4Wgsl } from '../shaders/flopsF16Mat4.ts';
import { flopsF16MatvecWgsl } from '../shaders/flopsF16Matvec.ts';
import type { PreparedBenchmark } from './common.ts';

/** Raw fp16 FLOPS: eight independent scalar f16 FMA chains per thread, unrolled 4x. */
export function prepareFlopsF16Scalar(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f16-scalar',
      label: 'fp16 scalar FMA FLOPS',
      description:
        'Same eight independent, 4x-unrolled FMA chains as the fp32 scalar test, but every operand and accumulator is f16, so the chains run entirely in half precision.',
      wgsl: flopsF16ScalarWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
      requiresF16: true,
    },
    harness,
  );
}

/** fp16 vec4 FLOPS: one f16 FMA chain on a vec4<f16> register (4 independent lanes). */
export function prepareFlopsF16Vec4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f16-vec4',
      label: 'fp16 vec4 FLOPS',
      description:
        'Same single FMA chain as the fp32 vec4 test, but held in a vec4<f16> register: 4 independent half-precision lanes per step. Only GPUs with packed-half ALUs run this faster than fp32.',
      wgsl: flopsF16Vec4Wgsl,
      flopsPerIteration: 8,
      defaultIterations: 1024,
      requiresF16: true,
    },
    harness,
  );
}

/** fp16 mat4 FLOPS: x = m*x + c chained with a mat4x4<f16>. */
export function prepareFlopsF16Mat4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f16-mat4',
      label: 'fp16 mat4 FLOPS',
      description:
        'Same bounded x = m * x + c recurrence as the fp32 mat4 test, but m, c, and x are all f16, so the mat4x4 x vec4 multiply runs entirely in half precision.',
      wgsl: flopsF16Mat4Wgsl,
      flopsPerIteration: 32,
      defaultIterations: 1024,
      requiresF16: true,
    },
    harness,
  );
}

/** fp16 register-resident matvec tile: 4x8 f16 weights in registers, two dot() calls per output row. */
export function prepareFlopsF16Matvec(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f16-matvec',
      label: 'fp16 matvec FLOPS',
      description:
        'Same register-resident 4x8 matvec tile as the fp32 matvec test, but weights, inputs, and dot() accumulation are all f16. Pure ALU: the fp16 win here comes only from the ALU, not from halved memory traffic.',
      wgsl: flopsF16MatvecWgsl,
      flopsPerIteration: 64,
      defaultIterations: 512,
      requiresF16: true,
    },
    harness,
  );
}
