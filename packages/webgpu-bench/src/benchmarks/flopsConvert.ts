import type { GpuContext } from '../gpu/context.ts';
import { prepareFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { flopsU32PackUnpackWgsl } from '../shaders/flopsU32PackUnpack.ts';
import { flopsI32F32ConvertWgsl } from '../shaders/flopsI32F32Convert.ts';
import { flopsF32F16ConvertWgsl } from '../shaders/flopsF32F16Convert.ts';
import { flopsI32F16ConvertWgsl } from '../shaders/flopsI32F16Convert.ts';
import { OPS_METRIC } from './common.ts';
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
      label: 'u32 byte pack/unpack ops',
      description:
        'Eight independent u32 lanes, each step unpacking 4 bytes via shift+mask, incrementing them, and repacking the same way — no pack4x8 (or unpack4x8) builtin, just the bit-twiddling those compile to. 25 ops/lane/step, no unrolling.',
      wgsl: flopsU32PackUnpackWgsl,
      metric: OPS_METRIC,
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
      label: 'i32<->f32 convert FLOPS',
      description:
        'Eight independent chains per thread, unrolled 4x: xi -> f32(xi)*a+b -> back to i32 each step. Same FMA as the fp32 scalar test plus a convert on each side, so the gap against that test isolates int<->float conversion cost.',
      wgsl: flopsI32F32ConvertWgsl,
      metric: OPS_METRIC,
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
      label: 'f32<->f16 convert FLOPS',
      description:
        'Eight independent vec2<f32> chains per thread, unrolled 4x: pack2x16float then unpack2x16float (round-trips through fp16 bits) plus a vec2 FMA to keep the chain moving. Unlike flops-f16-*, this needs no shader-f16 device feature — it measures the conversion, not f16 compute.',
      wgsl: flopsF32F16ConvertWgsl,
      metric: OPS_METRIC,
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
      label: 'i32<->f16 convert FLOPS',
      description:
        'Eight independent i32 chains per thread, unrolled 4x: xi -> f32 -> pack2x16float -> unpack2x16float -> f32*a+b -> i32. There is no native int<->f16 conversion, so this is what the real path (through f32) costs.',
      wgsl: flopsI32F16ConvertWgsl,
      metric: OPS_METRIC,
      flopsPerIteration: 192,
      defaultIterations: 256,
    },
    harness,
  );
}
