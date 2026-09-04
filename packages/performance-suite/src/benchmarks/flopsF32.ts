import type { GpuContext } from '../gpu/context.ts';
import { prepareFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { flopsF32ScalarWgsl } from '../shaders/flopsF32Scalar.ts';
import { flopsF32Vec4Wgsl } from '../shaders/flopsF32Vec4.ts';
import { flopsF32Mat4Wgsl } from '../shaders/flopsF32Mat4.ts';
import { flopsF32MatvecWgsl } from '../shaders/flopsF32Matvec.ts';
import type { PreparedBenchmark } from './common.ts';

/** Raw fp32 FLOPS: eight independent scalar FMA chains per thread, unrolled 4x. */
export function prepareFlopsF32Scalar(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-scalar',
      label: 'fp32 scalar FMA FLOPS',
      description:
        'Eight independent scalar f32 fused multiply-add chains per thread, unrolled 4x, so the ALU always has work in flight and the number reflects throughput rather than FMA latency. ~no memory traffic.',
      wgsl: flopsF32ScalarWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
    },
    harness,
  );
}

/** fp32 vec4 FLOPS: one FMA chain on a vec4<f32> register (4 independent lanes). */
export function prepareFlopsF32Vec4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-vec4',
      label: 'fp32 vec4 FLOPS',
      description:
        'A single FMA chain held in a vec4<f32> register. On scalar-SIMT GPUs (Apple, NVIDIA, AMD) this compiles to 4 independent scalar FMAs per step, so it measures how well 4-wide instruction-level parallelism hides latency, not a wider ALU.',
      wgsl: flopsF32Vec4Wgsl,
      flopsPerIteration: 8,
      defaultIterations: 1024,
    },
    harness,
  );
}

/** fp32 mat4 FLOPS: x = m*x + c chained with a mat4x4<f32>. */
export function prepareFlopsF32Mat4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-mat4',
      label: 'fp32 mat4 FLOPS',
      description:
        'x = m * x + c chained in a register with a mat4x4<f32> (a bounded contraction so it stays numerically stable): 16 FMAs per step with plenty of independent work, exercising the full mat4 x vec4 multiply.',
      wgsl: flopsF32Mat4Wgsl,
      flopsPerIteration: 32,
      defaultIterations: 1024,
    },
    harness,
  );
}

/** fp32 register-resident matvec tile: 4x8 weights in registers, two dot() calls per output row. */
export function prepareFlopsF32Matvec(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-matvec',
      label: 'fp32 matvec FLOPS',
      description:
        'A 4-row x 8-column f32 weight tile held in registers, applied to an 8-wide input every iteration via dot() (outputs feed back as the next inputs). The dot-product-accumulate shape of a GEMV inner loop with zero buffer traffic: pure ALU.',
      wgsl: flopsF32MatvecWgsl,
      flopsPerIteration: 64,
      defaultIterations: 512,
    },
    harness,
  );
}
