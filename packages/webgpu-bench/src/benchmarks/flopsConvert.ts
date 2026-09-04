import type { GpuContext } from '../gpu/context.ts';
import { prepareFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { flopsU32PackUnpackWgsl } from '../shaders/flopsU32PackUnpack.ts';
import { flopsI32F32ConvertWgsl } from '../shaders/flopsI32F32Convert.ts';
import { flopsF32F16ConvertWgsl } from '../shaders/flopsF32F16Convert.ts';
import { flopsI32F16ConvertWgsl } from '../shaders/flopsI32F16Convert.ts';
import type { PreparedBenchmark } from './common.ts';

/**
 * Bit-twiddling and type-conversion ops: what quantized/packed formats
 * actually cost to get in and out of, as opposed to the flops-i8-* kernels
 * (which measure compute once data is already unpacked into i32/u32
 * registers). "Ops/s" here again isn't IEEE FLOPs, same caveat as flopsMath.ts.
 */

/** Manual (shift+mask) byte pack/unpack: no pack4x8 (or unpack4x8) builtin, just what those compile to. */
export function prepareFlopsU32PackUnpack(
  ctx: GpuContext,
  harness: FlopsHarnessConfig = {},
): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-u32-packunpack',
      wgsl: flopsU32PackUnpackWgsl,
      flopsPerIteration: 200,
      defaultIterations: 256,
    },
    harness,
  );
}

/** i32<->f32 round-trip: convert, FMA, convert back — the delta against the fp32 scalar test isolates conversion cost. */
export function prepareFlopsI32F32Convert(
  ctx: GpuContext,
  harness: FlopsHarnessConfig = {},
): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i32-f32-convert',
      wgsl: flopsI32F32ConvertWgsl,
      flopsPerIteration: 128,
      defaultIterations: 256,
    },
    harness,
  );
}

/** fp32<->fp16 round-trip via pack2x16float/unpack2x16float — no shader-f16 feature needed. */
export function prepareFlopsF32F16Convert(
  ctx: GpuContext,
  harness: FlopsHarnessConfig = {},
): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f32-f16-convert',
      wgsl: flopsF32F16ConvertWgsl,
      flopsPerIteration: 192,
      defaultIterations: 256,
    },
    harness,
  );
}

/** i32<->f16 round-trip, chained through f32 (there's no native int<->f16 builtin). */
export function prepareFlopsI32F16Convert(
  ctx: GpuContext,
  harness: FlopsHarnessConfig = {},
): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i32-f16-convert',
      wgsl: flopsI32F16ConvertWgsl,
      flopsPerIteration: 192,
      defaultIterations: 256,
    },
    harness,
  );
}
