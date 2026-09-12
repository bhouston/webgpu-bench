import {
  BYTES_METRIC,
  FLOPS_METRIC,
  OPS_METRIC,
  type BenchmarkContext,
  type BenchmarkDefinition,
  type PreparedBenchmark,
} from './benchmarks/common.ts';
import type { MetricDef } from './types.ts';
import type { GpuContext } from './gpu/context.ts';
import type { FlopsHarnessConfig } from './benchmarks/flopsCommon.ts';
import { streamReadWgsl } from './shaders/streamRead.ts';
import { streamWriteWgsl } from './shaders/streamWrite.ts';
import {
  prepareReadBandwidth,
  prepareWriteBandwidth,
  prepareGatherRead,
  prepareScatterWrite,
} from './benchmarks/streamBandwidth.ts';
import { gatherReadWgsl } from './shaders/gatherRead.ts';
import { scatterWriteWgsl } from './shaders/scatterWrite.ts';
import { flopsF32ScalarWgsl } from './shaders/flopsF32Scalar.ts';
import { flopsF32Vec4Wgsl } from './shaders/flopsF32Vec4.ts';
import { flopsF32Mat4Wgsl } from './shaders/flopsF32Mat4.ts';
import { flopsF32MatvecWgsl } from './shaders/flopsF32Matvec.ts';
import {
  prepareFlopsF32Scalar,
  prepareFlopsF32Vec4,
  prepareFlopsF32Mat4,
  prepareFlopsF32Matvec,
} from './benchmarks/flopsF32.ts';
import { flopsF16ScalarWgsl } from './shaders/flopsF16Scalar.ts';
import { flopsF16Vec4Wgsl } from './shaders/flopsF16Vec4.ts';
import { flopsF16Mat4Wgsl } from './shaders/flopsF16Mat4.ts';
import { flopsF16MatvecWgsl } from './shaders/flopsF16Matvec.ts';
import {
  prepareFlopsF16Scalar,
  prepareFlopsF16Vec4,
  prepareFlopsF16Mat4,
  prepareFlopsF16Matvec,
} from './benchmarks/flopsF16.ts';
import { flopsI8ScalarWgsl } from './shaders/flopsI8Scalar.ts';
import { flopsI8Vec4Wgsl } from './shaders/flopsI8Vec4.ts';
import { flopsI8Mat4Wgsl } from './shaders/flopsI8Mat4.ts';
import { flopsI8MatvecWgsl } from './shaders/flopsI8Matvec.ts';
import { flopsI8MatvecDp4aWgsl } from './shaders/flopsI8MatvecDp4a.ts';
import { flopsI8Dp4aWgsl } from './shaders/flopsI8Dp4a.ts';
import {
  prepareFlopsI8Scalar,
  prepareFlopsI8Vec4,
  prepareFlopsI8Mat4,
  prepareFlopsI8Matvec,
  prepareFlopsI8MatvecDp4a,
  prepareFlopsI8Dp4a,
} from './benchmarks/flopsI8.ts';
import { flopsF32DivWgsl } from './shaders/flopsF32Div.ts';
import { flopsI32DivWgsl } from './shaders/flopsI32Div.ts';
import { flopsF32SqrtWgsl } from './shaders/flopsF32Sqrt.ts';
import { flopsF32RsqrtWgsl } from './shaders/flopsF32Rsqrt.ts';
import { flopsF32PowWgsl } from './shaders/flopsF32Pow.ts';
import { flopsF32SincosWgsl } from './shaders/flopsF32Sincos.ts';
import { flopsF32LogWgsl } from './shaders/flopsF32Log.ts';
import { flopsF16DivWgsl } from './shaders/flopsF16Div.ts';
import { flopsF16SqrtWgsl } from './shaders/flopsF16Sqrt.ts';
import { flopsF16RsqrtWgsl } from './shaders/flopsF16Rsqrt.ts';
import { flopsF16PowWgsl } from './shaders/flopsF16Pow.ts';
import { flopsF16SincosWgsl } from './shaders/flopsF16Sincos.ts';
import { flopsF16LogWgsl } from './shaders/flopsF16Log.ts';
import {
  prepareFlopsF32Div,
  prepareFlopsI32Div,
  prepareFlopsF32Sqrt,
  prepareFlopsF32Rsqrt,
  prepareFlopsF32Pow,
  prepareFlopsF32Sincos,
  prepareFlopsF32Log,
  prepareFlopsF16Div,
  prepareFlopsF16Sqrt,
  prepareFlopsF16Rsqrt,
  prepareFlopsF16Pow,
  prepareFlopsF16Sincos,
  prepareFlopsF16Log,
} from './benchmarks/flopsMath.ts';
import { flopsU32PackUnpackWgsl } from './shaders/flopsU32PackUnpack.ts';
import { flopsI32F32ConvertWgsl } from './shaders/flopsI32F32Convert.ts';
import { flopsF32F16ConvertWgsl } from './shaders/flopsF32F16Convert.ts';
import { flopsU32ShiftWgsl } from './shaders/flopsU32Shift.ts';
import {
  flopsF32FmaWgsl,
  flopsF32SelectWgsl,
  flopsF32ClampWgsl,
  flopsF32MinMaxWgsl,
  flopsU32CountOneBitsWgsl,
  flopsU32FirstLeadingBitWgsl,
  flopsU32ReverseBitsWgsl,
} from './shaders/flopsBuiltins.ts';
import {
  branchNoneWgsl,
  branchUniformWgsl,
  branchCoherentWgsl,
  branchDivergentWgsl,
} from './shaders/flopsF32Branch.ts';
import { flopsU8Dp4aWgsl } from './shaders/flopsU8Dp4a.ts';
import { prepareFlopsBenchmark } from './benchmarks/flopsCommon.ts';
import {
  prepareFlopsU32PackUnpack,
  prepareFlopsI32F32Convert,
  prepareFlopsF32F16Convert,
  prepareFlopsU32Shift,
} from './benchmarks/flopsConvert.ts';

export type { BenchmarkContext, BenchmarkDefinition } from './benchmarks/common.ts';

/** Wraps a `prepareRead/WriteBandwidth`-shaped function as a `BenchmarkDefinition.prepare`. */
function bandwidthPrepare(
  fn: (
    ctx: GpuContext,
    data: BenchmarkContext['data'],
    harness: BenchmarkContext['harness'],
  ) => Promise<PreparedBenchmark>,
) {
  return ({ ctx, data, harness }: BenchmarkContext) => fn(ctx, data, harness);
}

/** Wraps a `prepareFlops*`-shaped function as a `BenchmarkDefinition.prepare`, feeding in the compute work knobs. */
function flopsPrepare(fn: (ctx: GpuContext, harness: FlopsHarnessConfig) => Promise<PreparedBenchmark>) {
  return ({ ctx, harness, computeThreads, computeIterations }: BenchmarkContext) =>
    fn(ctx, { ...harness, threads: computeThreads, iterations: computeIterations });
}

/** A raw-FLOPS-harness kernel (see `prepareFlopsBenchmark`) as a full definition. `opsPerIteration` is per thread per loop trip. */
function flopsKernel(
  id: string,
  label: string,
  description: string,
  wgsl: string,
  metric: MetricDef,
  opsPerIteration: number,
  requiresI8Dot = false,
): BenchmarkDefinition {
  return {
    id,
    label,
    description,
    source: wgsl,
    category: 'compute',
    metric,
    prepare: ({ ctx, harness, computeThreads, computeIterations }) =>
      prepareFlopsBenchmark(
        ctx,
        { id, wgsl, flopsPerIteration: opsPerIteration, defaultIterations: 256, requiresI8Dot },
        { ...harness, threads: computeThreads, iterations: computeIterations },
      ),
  };
}

const KB = 1024;
const MB = 1024 * KB;

/** One read-gather + one write-scatter definition for a random-access window of `windowBytes` (clamped to the buffer). */
function gatherScatter(suffix: string, windowBytes: number, where: string): BenchmarkDefinition[] {
  return [
    {
      id: `read-gather-${suffix}`,
      label: `read gather ${suffix}`,
      description: `Same grid-stride loop and byte count as the linear read, but each thread loads a pseudo-random vec4<f32> from ${where}. ${
        windowBytes >= 64 * MB ? 'DRAM random-access throughput.' : 'Random access served by cache.'
      } Loads are independent, so this is throughput, not latency.`,
      source: gatherReadWgsl,
      category: 'bandwidth',
      metric: BYTES_METRIC,
      prepare: ({ ctx, data, harness }) => prepareGatherRead(ctx, data, harness, `read-gather-${suffix}`, windowBytes),
    },
    {
      id: `write-scatter-${suffix}`,
      label: `write scatter ${suffix}`,
      description: `Same grid-stride loop and byte count as the linear write, but each thread stores a vec4<f32> to a pseudo-random slot in ${where}. Colliding stores are a benign data race; the smaller the window the more they contend.`,
      source: scatterWriteWgsl,
      category: 'bandwidth',
      metric: BYTES_METRIC,
      prepare: ({ ctx, data, harness }) =>
        prepareScatterWrite(ctx, data, harness, `write-scatter-${suffix}`, windowBytes),
    },
  ];
}

/**
 * This package's own benchmarks: memory bandwidth and raw-FLOPS ALU
 * throughput. Each entry is fully self-contained (display metadata + how to
 * build it) per the `BenchmarkDefinition` contract in `benchmarks/common.ts`
 * — `runSuite` doesn't know anything about these specifically, so a caller
 * can run a filtered subset (`BENCHMARKS.filter(...)`), add their own
 * definitions alongside them, or ignore this list entirely and pass their
 * own via `SuiteOptions.benchmarks`.
 */
export const BENCHMARKS: readonly BenchmarkDefinition[] = [
  {
    id: 'read-linear',
    label: 'read linear',
    description:
      'Coalesced grid-stride loop: adjacent threads load adjacent vec4<f32>s from a large buffer, folded with addition only, one scalar written per thread. Read-bandwidth-bound.',
    source: streamReadWgsl,
    category: 'bandwidth',
    metric: BYTES_METRIC,
    prepare: bandwidthPrepare(prepareReadBandwidth),
  },
  {
    id: 'write-linear',
    label: 'write linear',
    description:
      'Coalesced grid-stride loop: adjacent threads store adjacent computed vec4<f32>s into a large buffer with no buffer reads. Write-bandwidth-bound.',
    source: streamWriteWgsl,
    category: 'bandwidth',
    metric: BYTES_METRIC,
    prepare: bandwidthPrepare(prepareWriteBandwidth),
  },
  ...gatherScatter('16kb', 16 * KB, 'a 16 KB window (L1-resident)'),
  ...gatherScatter('4mb', 4 * MB, 'a 4 MB window (L2-resident on most desktop GPUs)'),
  ...gatherScatter('64mb', 64 * MB, 'a 64 MB window (the whole default buffer)'),
  {
    id: 'f32-fma-scalar',
    label: 'fp32 scalar FMA',
    description:
      'Eight independent scalar f32 fused multiply-add chains per thread, unrolled 4x, so the ALU always has work in flight and the number reflects throughput rather than FMA latency. ~no memory traffic.',
    source: flopsF32ScalarWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF32Scalar),
  },
  {
    id: 'f32-fma-vec4',
    label: 'fp32 vec4 FMA',
    description:
      'Eight independent FMA chains held in vec4<f32> registers, unrolled 4x: the fp32 scalar test with every chain 4 lanes wide. On scalar-SIMT GPUs (Apple, NVIDIA, AMD) each step is 4 scalar FMAs, so this should match the scalar number; a gap means vector ops cost extra.',
    source: flopsF32Vec4Wgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF32Vec4),
  },
  {
    id: 'f32-fma-mat4',
    label: 'fp32 mat4 FMA',
    description:
      'x = m * x + c chained in a register with a mat4x4<f32> (a bounded contraction so it stays numerically stable): 16 FMAs per step with plenty of independent work, exercising the full mat4 x vec4 multiply.',
    source: flopsF32Mat4Wgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF32Mat4),
  },
  {
    id: 'f32-fma-matvec',
    label: 'fp32 matvec FMA',
    description:
      'A 4-row x 8-column f32 weight tile held in registers, applied to an 8-wide input every iteration via dot() (outputs feed back as the next inputs). The dot-product-accumulate shape of a GEMV inner loop with zero buffer traffic: pure ALU.',
    source: flopsF32MatvecWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF32Matvec),
  },
  {
    id: 'f16-fma-scalar',
    label: 'fp16 scalar FMA',
    description:
      'Same eight independent, 4x-unrolled FMA chains as the fp32 scalar test, but every operand and accumulator is f16, so the chains run entirely in half precision.',
    source: flopsF16ScalarWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF16Scalar),
  },
  {
    id: 'f16-fma-vec4',
    label: 'fp16 vec4 FMA',
    description:
      'Same eight 4x-unrolled vec4 FMA chains as the fp32 vec4 test, but in vec4<f16>. Only GPUs with packed-half ALUs run this faster than fp32.',
    source: flopsF16Vec4Wgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF16Vec4),
  },
  {
    id: 'f16-fma-mat4',
    label: 'fp16 mat4 FMA',
    description:
      'Same bounded x = m * x + c recurrence as the fp32 mat4 test, but m, c, and x are all f16, so the mat4x4 x vec4 multiply runs entirely in half precision.',
    source: flopsF16Mat4Wgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF16Mat4),
  },
  {
    id: 'f16-fma-matvec',
    label: 'fp16 matvec FMA',
    description:
      'Same register-resident 4x8 matvec tile as the fp32 matvec test, but weights, inputs, and dot() accumulation are all f16. Pure ALU: the fp16 win here comes only from the ALU, not from halved memory traffic.',
    source: flopsF16MatvecWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF16Matvec),
  },
  {
    id: 'i32-mad-scalar',
    label: 'i32 scalar multiply-add',
    description:
      'Same eight independent, 4x-unrolled multiply-add chains as the fp32 scalar test, but on i32 with int8-range operands. Measures 32-bit integer multiply-add throughput; overflow wraps.',
    source: flopsI8ScalarWgsl,
    category: 'compute',
    metric: OPS_METRIC,
    prepare: flopsPrepare(prepareFlopsI8Scalar),
  },
  {
    id: 'i32-mad-vec4',
    label: 'i32 vec4 multiply-add',
    description: 'Same eight 4x-unrolled vec4 multiply-add chains as the fp32 vec4 test, but in vec4<i32>.',
    source: flopsI8Vec4Wgsl,
    category: 'compute',
    metric: OPS_METRIC,
    prepare: flopsPrepare(prepareFlopsI8Vec4),
  },
  {
    id: 'i32-mad-mat4',
    label: 'i32 mat4 multiply-add',
    description:
      'x = m * x + c chained in a register, where m is a 4x4 integer matrix emulated as four vec4<i32> rows combined with dot() (WGSL has no mat4x4<i32>): the same 4 dot products a real int4x4 multiply compiles to.',
    source: flopsI8Mat4Wgsl,
    category: 'compute',
    metric: OPS_METRIC,
    prepare: flopsPrepare(prepareFlopsI8Mat4),
  },
  {
    id: 'i32-mad-matvec',
    label: 'i32 matvec multiply-add',
    description:
      'Same register-resident 4x8 matvec tile as the fp32 matvec test, with int8-range weights and inputs held unpacked as vec4<i32> and accumulated with integer dot(). The no-extension int8 path: what dot4I8Packed is competing against.',
    source: flopsI8MatvecWgsl,
    category: 'compute',
    metric: OPS_METRIC,
    prepare: flopsPrepare(prepareFlopsI8Matvec),
  },
  {
    id: 'i8-dp4a-matvec',
    label: 'int8 dp4a matvec',
    description:
      "Same 4x8 matvec tile, but weights and inputs stay packed four int8 lanes per u32 and each 4-wide dot product is one dot4I8Packed call from the packed_4x8_integer_dot_product extension. On GPUs without a native dp4a instruction this runs the extension's polyfill.",
    source: flopsI8MatvecDp4aWgsl,
    category: 'compute',
    metric: OPS_METRIC,
    prepare: flopsPrepare(prepareFlopsI8MatvecDp4a),
  },
  {
    id: 'i8-dp4a',
    label: 'int8 dp4a',
    description:
      "Eight independent accumulators per thread, each summing dot4I8Packed(a, b) — the packed_4x8_integer_dot_product extension's 4-wide int8 dot-product instruction — in a tight loop to measure its peak throughput in isolation.",
    source: flopsI8Dp4aWgsl,
    category: 'compute',
    metric: OPS_METRIC,
    prepare: flopsPrepare(prepareFlopsI8Dp4a),
  },
  {
    id: 'f32-div',
    label: 'fp32 div',
    description:
      'Eight independent scalar f32 chains of x = a / x + b per thread, unrolled 4x — the fp32 scalar FMA test with divide in place of multiply, so the gap between the two isolates the cost of division. The loop-carried value is the divisor, so the compiler cannot hoist a reciprocal and turn it back into an FMA.',
    source: flopsF32DivWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF32Div),
  },
  {
    id: 'i32-div',
    label: 'i32 div',
    description:
      'Eight independent scalar i32 chains of x = a / x + b per thread, unrolled 4x — the i32 multiply-add scalar test with divide in place of multiply, divisor loop-carried. Integer division is typically the slowest basic ALU op on a GPU.',
    source: flopsI32DivWgsl,
    category: 'compute',
    metric: OPS_METRIC,
    prepare: flopsPrepare(prepareFlopsI32Div),
  },
  {
    id: 'f32-sqrt',
    label: 'fp32 sqrt',
    description:
      'Eight independent scalar f32 sqrt-add chains per thread, unrolled 4x. sqrt is a common special-function-unit instruction; this measures its throughput in isolation.',
    source: flopsF32SqrtWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF32Sqrt),
  },
  {
    id: 'f32-rsqrt',
    label: 'fp32 rsqrt',
    description:
      'Eight independent scalar f32 inverseSqrt-add chains per thread, unrolled 4x. The op behind every normalize(); most GPUs have a dedicated fast-rsqrt path, so compare against f32-sqrt to see the gap.',
    source: flopsF32RsqrtWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF32Rsqrt),
  },
  {
    id: 'f32-pow',
    label: 'fp32 pow',
    description:
      'Eight independent scalar f32 pow-add chains per thread, unrolled 4x. pow(x, e) for a non-integer e is usually exp2(e * log2(x)) under the hood — several instructions — so expect this well below sqrt/div throughput.',
    source: flopsF32PowWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF32Pow),
  },
  {
    id: 'f32-sincos',
    label: 'fp32 sin/cos',
    description:
      'Eight independent scalar f32 cos(sin(x)) chains per thread, unrolled 4x. Naturally bounded to [-1, 1], so no stabilization term is needed. Measures combined sin+cos throughput.',
    source: flopsF32SincosWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF32Sincos),
  },
  {
    id: 'f32-log',
    label: 'fp32 ln',
    description:
      'Eight independent scalar f32 ln-add chains per thread, unrolled 4x. log(x) is usually log2(x) * ln(2) under the hood, so expect throughput close to a raw log2 special-function call.',
    source: flopsF32LogWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF32Log),
  },
  {
    id: 'f16-div',
    label: 'fp16 div',
    description:
      'Eight independent scalar f16 chains of x = a / x + b per thread, unrolled 4x — the fp16 scalar FMA test with divide in place of multiply, so the gap against f16-fma-scalar isolates the cost of division in half precision. The loop-carried value is the divisor, so the compiler cannot hoist a reciprocal and turn it back into an FMA.',
    source: flopsF16DivWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF16Div),
  },
  {
    id: 'f16-sqrt',
    label: 'fp16 sqrt',
    description:
      'Eight independent scalar f16 sqrt-add chains per thread, unrolled 4x. Same shape as f32-sqrt but entirely in half precision.',
    source: flopsF16SqrtWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF16Sqrt),
  },
  {
    id: 'f16-rsqrt',
    label: 'fp16 rsqrt',
    description:
      'Eight independent scalar f16 inverseSqrt-add chains per thread, unrolled 4x. Same shape as f32-rsqrt but entirely in half precision; compare against f16-sqrt to see the gap.',
    source: flopsF16RsqrtWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF16Rsqrt),
  },
  {
    id: 'f16-pow',
    label: 'fp16 pow',
    description:
      'Eight independent scalar f16 pow-add chains per thread, unrolled 4x. Same shape as f32-pow but entirely in half precision.',
    source: flopsF16PowWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF16Pow),
  },
  {
    id: 'f16-sincos',
    label: 'fp16 sin/cos',
    description:
      'Eight independent scalar f16 cos(sin(x)) chains per thread, unrolled 4x. Same shape as f32-sincos but entirely in half precision.',
    source: flopsF16SincosWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF16Sincos),
  },
  {
    id: 'f16-log',
    label: 'fp16 ln',
    description:
      'Eight independent scalar f16 ln-add chains per thread, unrolled 4x. Same shape as f32-log but entirely in half precision.',
    source: flopsF16LogWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF16Log),
  },
  {
    id: 'u32-packunpack',
    label: 'u32 byte pack/unpack',
    description:
      'Eight independent u32 lanes, unrolled 4x, each step unpacking 4 bytes via shift+mask, incrementing them, and repacking — no pack4x8 (or unpack4x8) builtin, just the bit-twiddling those compile to. Counted as 2 ops per lane step (one unpack + one pack of a whole u32), since the compiler folds the individual shifts and masks.',
    source: flopsU32PackUnpackWgsl,
    category: 'compute',
    metric: OPS_METRIC,
    prepare: flopsPrepare(prepareFlopsU32PackUnpack),
  },
  {
    id: 'u32-shift',
    label: 'u32 variable shift',
    description:
      'Eight independent u32 lanes, unrolled 4x, even lanes shifting left and odd lanes right by an amount taken from the loop counter, so every lane cycles through all 32 shift amounts and the number is the average across them. Counted as 1 op per lane step (the shift only; the XOR that refills shifted-out bits is not counted).',
    source: flopsU32ShiftWgsl,
    category: 'compute',
    metric: OPS_METRIC,
    prepare: flopsPrepare(prepareFlopsU32Shift),
  },
  {
    id: 'i32-f32-convert',
    label: 'i32<->f32 convert',
    description:
      'Eight independent chains per thread, unrolled 4x: xi -> f32(xi)*a+b -> back to i32 each step. Counted as 2 ops per lane step (one convert each way; the FMA is not counted).',
    source: flopsI32F32ConvertWgsl,
    category: 'compute',
    metric: OPS_METRIC,
    prepare: flopsPrepare(prepareFlopsI32F32Convert),
  },
  {
    id: 'f32-f16-convert',
    label: 'f32<->f16 convert',
    description:
      'Eight independent vec2<f32> chains per thread, unrolled 4x: pack2x16float then unpack2x16float (round-trips through fp16 bits) plus a vec2 FMA to keep the chain moving. Counted as 2 ops per lane per step (one convert each way; the FMA is not counted). Unlike f16-*, this needs no shader-f16 device feature — it measures the conversion, not f16 compute.',
    source: flopsF32F16ConvertWgsl,
    category: 'compute',
    metric: OPS_METRIC,
    prepare: flopsPrepare(prepareFlopsF32F16Convert),
  },
  flopsKernel(
    'f32-fma-builtin',
    'fp32 fma() builtin',
    'The fp32 scalar test with the explicit fma(x, a, b) builtin in place of x * a + b. WGSL lets the compiler contract the latter into an FMA; if this number matches f32-fma-scalar, it does.',
    flopsF32FmaWgsl,
    FLOPS_METRIC,
    64,
  ),
  flopsKernel(
    'f32-select',
    'fp32 select',
    'Eight independent f32 lanes, unrolled 4x, each stepping x = select(x + b, x - a, x > 1.0): a bounded walk whose direction is a runtime compare. Counted as 1 op per lane step (the select; compare and add/sub not counted).',
    flopsF32SelectWgsl,
    OPS_METRIC,
    32,
  ),
  flopsKernel(
    'f32-clamp',
    'fp32 clamp',
    'Eight independent f32 lanes, unrolled 4x, each stepping x = clamp(x * a + b, 0, 1). Counted as 1 op per lane step (the clamp; the FMA not counted). Half the ops/s of f32-minmax means clamp compiled to a separate min and max.',
    flopsF32ClampWgsl,
    OPS_METRIC,
    32,
  ),
  flopsKernel(
    'f32-minmax',
    'fp32 min/max',
    'Eight independent f32 lanes, unrolled 4x, each stepping x = max(min(x * a + b, 1), 0). Counted as 2 ops per lane step (min + max; the FMA not counted).',
    flopsF32MinMaxWgsl,
    OPS_METRIC,
    64,
  ),
  flopsKernel(
    'u32-countonebits',
    'u32 countOneBits',
    'Eight independent u32 lanes, unrolled 4x, each stepping x = countOneBits(x) ^ k. Counted as 1 op per lane step (the popcount; the XOR that keeps the lane moving not counted).',
    flopsU32CountOneBitsWgsl,
    OPS_METRIC,
    32,
  ),
  flopsKernel(
    'u32-firstleadingbit',
    'u32 firstLeadingBit',
    'Eight independent u32 lanes, unrolled 4x, each stepping x = firstLeadingBit(x) ^ k. Counted as 1 op per lane step (the leading-bit scan; the XOR not counted).',
    flopsU32FirstLeadingBitWgsl,
    OPS_METRIC,
    32,
  ),
  flopsKernel(
    'u32-reversebits',
    'u32 reverseBits',
    'Eight independent u32 lanes, unrolled 4x, each stepping x = reverseBits(x) ^ k. Counted as 1 op per lane step (the bit reverse; the XOR not counted).',
    flopsU32ReverseBitsWgsl,
    OPS_METRIC,
    32,
  ),
  flopsKernel(
    'u8-dp4a',
    'uint8 dp4a',
    'Unsigned twin of i8-dp4a: the same eight loop-carried accumulator chains summing dot4U8Packed(a, b). Shows whether the hardware has a native unsigned dp4a as well as a signed one.',
    flopsU8Dp4aWgsl,
    OPS_METRIC,
    64,
    true,
  ),
  flopsKernel(
    'branch-none',
    'branch none',
    'Baseline for the branch tests: the same eight-chain fp32 FMA body as the other three with no if at all (identical to f32-fma-scalar).',
    branchNoneWgsl,
    FLOPS_METRIC,
    64,
  ),
  flopsKernel(
    'branch-uniform',
    'branch uniform',
    'The branch-none body with every unrolled step wrapped in an if/else whose two sides do equal work with different constants. The condition alternates each loop trip but is identical for every lane, so only one side ever executes. The gap against branch-none is the cost of the compare and jump alone.',
    branchUniformWgsl,
    FLOPS_METRIC,
    64,
  ),
  flopsKernel(
    'branch-coherent',
    'branch coherent',
    'Same as branch-uniform, but the condition also flips per 64-thread workgroup: lanes within a wave agree while neighbouring workgroups disagree. Tests whether the GPU detects dynamic uniformity at runtime; a gap against branch-uniform means it does not.',
    branchCoherentWgsl,
    FLOPS_METRIC,
    64,
  ),
  flopsKernel(
    'branch-divergent',
    'branch divergent',
    'Same as branch-uniform, but the condition flips per lane: adjacent threads take opposite sides, so every wave executes both sides under a mask. The worst case for SIMT; expect roughly half of branch-uniform.',
    branchDivergentWgsl,
    FLOPS_METRIC,
    64,
  ),
];
