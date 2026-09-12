import type { GpuContext } from '../gpu/context.ts';
import { prepareFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { flopsU32PackUnpackWgsl } from '../shaders/flopsU32PackUnpack.ts';
import { flopsI32F32ConvertWgsl } from '../shaders/flopsI32F32Convert.ts';
import { flopsF32F16ConvertWgsl } from '../shaders/flopsF32F16Convert.ts';
import { flopsU32ShiftWgsl } from '../shaders/flopsU32Shift.ts';
import type { PreparedBenchmark } from './common.ts';

/**
 * Bit-twiddling and type-conversion ops: what quantized/packed formats
 * actually cost to get in and out of, as opposed to the i8-* kernels
 * (which measure compute once data is already unpacked into i32/u32
 * registers). "Ops" here count only the conversions themselves, one per
 * direction per lane (an unpack or pack of a whole u32 counts as one), not
 * the FMA / shifts / masks around them: the compiler folds and fuses those,
 * so a source-level op count would overstate what executes.
 */

/** Manual (shift+mask) byte pack/unpack: no pack4x8 (or unpack4x8) builtin, just what those compile to. 2 ops per lane step (unpack + pack). */
export function prepareFlopsU32PackUnpack(
  ctx: GpuContext,
  harness: FlopsHarnessConfig = {},
): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'u32-packunpack',
      wgsl: flopsU32PackUnpackWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
    },
    harness,
  );
}

/** i32<->f32 round-trip: convert, FMA, convert back — 2 ops per lane step (one convert each way). */
export function prepareFlopsI32F32Convert(
  ctx: GpuContext,
  harness: FlopsHarnessConfig = {},
): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'i32-f32-convert',
      wgsl: flopsI32F32ConvertWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
    },
    harness,
  );
}

/** fp32<->fp16 round-trip via pack2x16float/unpack2x16float — no shader-f16 feature needed. Counted as 2 ops per lane round-trip (one convert each way). */
export function prepareFlopsF32F16Convert(
  ctx: GpuContext,
  harness: FlopsHarnessConfig = {},
): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'f32-f16-convert',
      wgsl: flopsF32F16ConvertWgsl,
      flopsPerIteration: 128,
      defaultIterations: 256,
    },
    harness,
  );
}

/** Variable-amount u32 shifts, left and right, cycling through all 32 amounts. 1 op per lane step (the shift). */
export function prepareFlopsU32Shift(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'u32-shift',
      wgsl: flopsU32ShiftWgsl,
      flopsPerIteration: 32,
      defaultIterations: 256,
    },
    harness,
  );
}
