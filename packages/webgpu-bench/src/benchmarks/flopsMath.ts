import type { GpuContext } from '../gpu/context.ts';
import { prepareFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { flopsF32DivWgsl } from '../shaders/flopsF32Div.ts';
import { flopsI32DivWgsl } from '../shaders/flopsI32Div.ts';
import { flopsF32SqrtWgsl } from '../shaders/flopsF32Sqrt.ts';
import { flopsF32RsqrtWgsl } from '../shaders/flopsF32Rsqrt.ts';
import { flopsF32PowWgsl } from '../shaders/flopsF32Pow.ts';
import { flopsF32SincosWgsl } from '../shaders/flopsF32Sincos.ts';
import { flopsF32LogWgsl } from '../shaders/flopsF32Log.ts';
import type { PreparedBenchmark } from './common.ts';

/**
 * "Real" ALU ops beyond multiply-add: division and the transcendentals
 * (sqrt, pow, sin/cos, log) that 3D math leans on constantly (normalize,
 * lighting falloff, exponential roughness/tonemap curves, trig for angles).
 * These aren't FLOPs in the IEEE sense — a GPU's sqrt/sin/pow are either a
 * dedicated special-function-unit instruction or a multi-instruction
 * polynomial approximation, so "ops/s" here means "results/s", not a fixed
 * instruction count — but they're counted with the same MAC-as-2 convention
 * as the multiply-add kernels so the throughput numbers stay comparable.
 */

/** fp32 divide-add ops: same shape as the fp32 scalar FMA test, multiply swapped for divide. */
export function prepareFlopsF32Div(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-div',
      wgsl: flopsF32DivWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
    },
    harness,
  );
}

/** i32 divide-add ops: same shape as the int8-range scalar test, multiply swapped for divide. */
export function prepareFlopsI32Div(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i32-div',
      wgsl: flopsI32DivWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
    },
    harness,
  );
}

/** fp32 sqrt-add ops: sqrt in place of the FMA multiply. */
export function prepareFlopsF32Sqrt(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-sqrt',
      wgsl: flopsF32SqrtWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
    },
    harness,
  );
}

/** fp32 rsqrt-add ops: inverseSqrt in place of the FMA multiply. */
export function prepareFlopsF32Rsqrt(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-rsqrt',
      wgsl: flopsF32RsqrtWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
    },
    harness,
  );
}

/** fp32 pow-add ops: pow in place of the FMA multiply. */
export function prepareFlopsF32Pow(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-pow',
      wgsl: flopsF32PowWgsl,
      flopsPerIteration: 64,
      defaultIterations: 128,
    },
    harness,
  );
}

/** fp32 sin(cos(x)) ops: the trig pair 3D math uses for angles/rotations. */
export function prepareFlopsF32Sincos(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-sincos',
      wgsl: flopsF32SincosWgsl,
      flopsPerIteration: 64,
      defaultIterations: 128,
    },
    harness,
  );
}

/** fp32 ln-add ops: natural log in place of the FMA multiply. */
export function prepareFlopsF32Log(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-log',
      wgsl: flopsF32LogWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
    },
    harness,
  );
}
