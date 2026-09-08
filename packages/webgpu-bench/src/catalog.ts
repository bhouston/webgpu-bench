import { BYTES_METRIC, FLOPS_METRIC, OPS_METRIC } from './benchmarks/common.ts';
import type { BenchmarkCategory, MetricDef } from './types.ts';
import { streamReadWgsl } from './shaders/streamRead.ts';
import { streamWriteWgsl } from './shaders/streamWrite.ts';
import { flopsF32ScalarWgsl } from './shaders/flopsF32Scalar.ts';
import { flopsF32Vec4Wgsl } from './shaders/flopsF32Vec4.ts';
import { flopsF32Mat4Wgsl } from './shaders/flopsF32Mat4.ts';
import { flopsF32MatvecWgsl } from './shaders/flopsF32Matvec.ts';
import { flopsF16ScalarWgsl } from './shaders/flopsF16Scalar.ts';
import { flopsF16Vec4Wgsl } from './shaders/flopsF16Vec4.ts';
import { flopsF16Mat4Wgsl } from './shaders/flopsF16Mat4.ts';
import { flopsF16MatvecWgsl } from './shaders/flopsF16Matvec.ts';
import { flopsI8ScalarWgsl } from './shaders/flopsI8Scalar.ts';
import { flopsI8Vec4Wgsl } from './shaders/flopsI8Vec4.ts';
import { flopsI8Mat4Wgsl } from './shaders/flopsI8Mat4.ts';
import { flopsI8MatvecWgsl } from './shaders/flopsI8Matvec.ts';
import { flopsI8MatvecDp4aWgsl } from './shaders/flopsI8MatvecDp4a.ts';
import { flopsI8Dp4aWgsl } from './shaders/flopsI8Dp4a.ts';
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
import { flopsU32PackUnpackWgsl } from './shaders/flopsU32PackUnpack.ts';
import { flopsI32F32ConvertWgsl } from './shaders/flopsI32F32Convert.ts';
import { flopsF32F16ConvertWgsl } from './shaders/flopsF32F16Convert.ts';

/**
 * Identity of one benchmark, known without touching the GPU: label,
 * description and WGSL source for display, plus category/metric for
 * formatting a result once one arrives. The sole source of truth for all
 * of this — `runSuite`'s `BenchmarkResult` rows carry only what's known
 * from actually running (status, timings, `metricValue`), keyed by `id`
 * back onto this catalog rather than repeating it.
 */
export interface BenchmarkInfo {
  id: string;
  label: string;
  description: string;
  /** WGSL source of the kernel, for display alongside a result. */
  source: string;
  category: BenchmarkCategory;
  metric: MetricDef;
}

/**
 * Every benchmark `runSuite` can yield a row for, in the same order it
 * schedules them — so a UI can render the full results table (id, label,
 * category, metric unit, description, source) before a device is even
 * acquired, then fill in throughput as `runSuite`'s rows arrive and merge
 * onto these ids.
 *
 * Kept as a hand-written list, not derived from `suite.ts`'s benchmark
 * table, because `suite.ts` only wires up *how* to run each kernel (its
 * `prepare*` call) — it doesn't carry a duplicate copy of this metadata to
 * derive from. `suite.browser.test.ts` asserts this list's ids match
 * `runSuite`'s one-to-one, so the two can't drift silently.
 */
export const BENCHMARK_CATALOG: readonly BenchmarkInfo[] = [
  {
    id: 'read-bandwidth',
    label: 'Read bandwidth',
    description:
      'Coalesced grid-stride loop: adjacent threads load adjacent vec4<f32>s from a large buffer, folded with addition only, one scalar written per thread. Read-bandwidth-bound.',
    source: streamReadWgsl,
    category: 'bandwidth',
    metric: BYTES_METRIC,
  },
  {
    id: 'write-bandwidth',
    label: 'Write bandwidth',
    description:
      'Coalesced grid-stride loop: adjacent threads store adjacent computed vec4<f32>s into a large buffer with no buffer reads. Write-bandwidth-bound.',
    source: streamWriteWgsl,
    category: 'bandwidth',
    metric: BYTES_METRIC,
  },
  {
    id: 'flops-f32-scalar',
    label: 'fp32 scalar FMA FLOPS',
    description:
      'Eight independent scalar f32 fused multiply-add chains per thread, unrolled 4x, so the ALU always has work in flight and the number reflects throughput rather than FMA latency. ~no memory traffic.',
    source: flopsF32ScalarWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f32-vec4',
    label: 'fp32 vec4 FLOPS',
    description:
      'Eight independent FMA chains held in vec4<f32> registers, unrolled 4x: the fp32 scalar test with every chain 4 lanes wide. On scalar-SIMT GPUs (Apple, NVIDIA, AMD) each step is 4 scalar FMAs, so this should match the scalar number; a gap means vector ops cost extra.',
    source: flopsF32Vec4Wgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f32-mat4',
    label: 'fp32 mat4 FLOPS',
    description:
      'x = m * x + c chained in a register with a mat4x4<f32> (a bounded contraction so it stays numerically stable): 16 FMAs per step with plenty of independent work, exercising the full mat4 x vec4 multiply.',
    source: flopsF32Mat4Wgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f32-matvec',
    label: 'fp32 matvec FLOPS',
    description:
      'A 4-row x 8-column f32 weight tile held in registers, applied to an 8-wide input every iteration via dot() (outputs feed back as the next inputs). The dot-product-accumulate shape of a GEMV inner loop with zero buffer traffic: pure ALU.',
    source: flopsF32MatvecWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f16-scalar',
    label: 'fp16 scalar FMA FLOPS',
    description:
      'Same eight independent, 4x-unrolled FMA chains as the fp32 scalar test, but every operand and accumulator is f16, so the chains run entirely in half precision.',
    source: flopsF16ScalarWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f16-vec4',
    label: 'fp16 vec4 FLOPS',
    description:
      'Same eight 4x-unrolled vec4 FMA chains as the fp32 vec4 test, but in vec4<f16>. Only GPUs with packed-half ALUs run this faster than fp32.',
    source: flopsF16Vec4Wgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f16-mat4',
    label: 'fp16 mat4 FLOPS',
    description:
      'Same bounded x = m * x + c recurrence as the fp32 mat4 test, but m, c, and x are all f16, so the mat4x4 x vec4 multiply runs entirely in half precision.',
    source: flopsF16Mat4Wgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f16-matvec',
    label: 'fp16 matvec FLOPS',
    description:
      'Same register-resident 4x8 matvec tile as the fp32 matvec test, but weights, inputs, and dot() accumulation are all f16. Pure ALU: the fp16 win here comes only from the ALU, not from halved memory traffic.',
    source: flopsF16MatvecWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-i8-scalar',
    label: 'int8-range scalar FMA FLOPS',
    description:
      'Same eight independent, 4x-unrolled multiply-add chains as the fp32 scalar test, but on i32 (WGSL has no first-class i8 type). Measures integer multiply-add throughput; overflow wraps.',
    source: flopsI8ScalarWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-i8-vec4',
    label: 'int8-range vec4 FLOPS',
    description: 'Same eight 4x-unrolled vec4 multiply-add chains as the fp32 vec4 test, but in vec4<i32>.',
    source: flopsI8Vec4Wgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-i8-mat4',
    label: 'int8-range mat4 FLOPS',
    description:
      'x = m * x + c chained in a register, where m is a 4x4 integer matrix emulated as four vec4<i32> rows combined with dot() (WGSL has no mat4x4<i32>): the same 4 dot products a real int4x4 multiply compiles to.',
    source: flopsI8Mat4Wgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-i8-matvec',
    label: 'int8 matvec FLOPS (i32 dot)',
    description:
      'Same register-resident 4x8 matvec tile as the fp32 matvec test, with int8-range weights and inputs held unpacked as vec4<i32> and accumulated with integer dot(). The no-extension int8 path: what dot4I8Packed is competing against.',
    source: flopsI8MatvecWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-i8-matvec-dp4a',
    label: 'int8 matvec FLOPS (dot4I8Packed)',
    description:
      "Same 4x8 matvec tile, but weights and inputs stay packed four int8 lanes per u32 and each 4-wide dot product is one dot4I8Packed call from the packed_4x8_integer_dot_product extension. On GPUs without a native dp4a instruction this runs the extension's polyfill.",
    source: flopsI8MatvecDp4aWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-i8-dp4a',
    label: 'int8 dot4I8Packed FLOPS',
    description:
      "Eight independent accumulators per thread, each summing dot4I8Packed(a, b) — the packed_4x8_integer_dot_product extension's 4-wide int8 dot-product instruction — in a tight loop to measure its peak throughput in isolation.",
    source: flopsI8Dp4aWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-f32-div',
    label: 'fp32 div FLOPS',
    description:
      'Eight independent scalar f32 chains of x = a / x + b per thread, unrolled 4x — the fp32 scalar FMA test with divide in place of multiply, so the gap between the two isolates the cost of division. The loop-carried value is the divisor, so the compiler cannot hoist a reciprocal and turn it back into an FMA.',
    source: flopsF32DivWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-i32-div',
    label: 'i32 div FLOPS',
    description:
      'Eight independent scalar i32 chains of x = a / x + b per thread, unrolled 4x — the int8-range scalar test with divide in place of multiply, divisor loop-carried. Integer division is typically the slowest basic ALU op on a GPU.',
    source: flopsI32DivWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-f32-sqrt',
    label: 'fp32 sqrt FLOPS',
    description:
      'Eight independent scalar f32 sqrt-add chains per thread, unrolled 4x. sqrt is a common special-function-unit instruction; this measures its throughput in isolation.',
    source: flopsF32SqrtWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f32-rsqrt',
    label: 'fp32 rsqrt FLOPS',
    description:
      'Eight independent scalar f32 inverseSqrt-add chains per thread, unrolled 4x. The op behind every normalize(); most GPUs have a dedicated fast-rsqrt path, so compare against flops-f32-sqrt to see the gap.',
    source: flopsF32RsqrtWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f32-pow',
    label: 'fp32 pow FLOPS',
    description:
      'Eight independent scalar f32 pow-add chains per thread, unrolled 4x. pow(x, e) for a non-integer e is usually exp2(e * log2(x)) under the hood — several instructions — so expect this well below sqrt/div throughput.',
    source: flopsF32PowWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f32-sincos',
    label: 'fp32 sin/cos FLOPS',
    description:
      'Eight independent scalar f32 cos(sin(x)) chains per thread, unrolled 4x. Naturally bounded to [-1, 1], so no stabilization term is needed. Measures combined sin+cos throughput.',
    source: flopsF32SincosWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f32-log',
    label: 'fp32 ln FLOPS',
    description:
      'Eight independent scalar f32 ln-add chains per thread, unrolled 4x. log(x) is usually log2(x) * ln(2) under the hood, so expect throughput close to a raw log2 special-function call.',
    source: flopsF32LogWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f16-div',
    label: 'fp16 div FLOPS',
    description:
      'Eight independent scalar f16 chains of x = a / x + b per thread, unrolled 4x — the fp16 scalar FMA test with divide in place of multiply, so the gap against flops-f16-scalar isolates the cost of division in half precision. The loop-carried value is the divisor, so the compiler cannot hoist a reciprocal and turn it back into an FMA.',
    source: flopsF16DivWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f16-sqrt',
    label: 'fp16 sqrt FLOPS',
    description:
      'Eight independent scalar f16 sqrt-add chains per thread, unrolled 4x. Same shape as flops-f32-sqrt but entirely in half precision.',
    source: flopsF16SqrtWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f16-rsqrt',
    label: 'fp16 rsqrt FLOPS',
    description:
      'Eight independent scalar f16 inverseSqrt-add chains per thread, unrolled 4x. Same shape as flops-f32-rsqrt but entirely in half precision; compare against flops-f16-sqrt to see the gap.',
    source: flopsF16RsqrtWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f16-pow',
    label: 'fp16 pow FLOPS',
    description:
      'Eight independent scalar f16 pow-add chains per thread, unrolled 4x. Same shape as flops-f32-pow but entirely in half precision.',
    source: flopsF16PowWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f16-sincos',
    label: 'fp16 sin/cos FLOPS',
    description:
      'Eight independent scalar f16 cos(sin(x)) chains per thread, unrolled 4x. Same shape as flops-f32-sincos but entirely in half precision.',
    source: flopsF16SincosWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-f16-log',
    label: 'fp16 ln FLOPS',
    description:
      'Eight independent scalar f16 ln-add chains per thread, unrolled 4x. Same shape as flops-f32-log but entirely in half precision.',
    source: flopsF16LogWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-u32-packunpack',
    label: 'u32 byte pack/unpack',
    description:
      'Eight independent u32 lanes, unrolled 4x, each step unpacking 4 bytes via shift+mask, incrementing them, and repacking — no pack4x8 (or unpack4x8) builtin, just the bit-twiddling those compile to. Counted as 2 ops per lane step (one unpack + one pack of a whole u32), since the compiler folds the individual shifts and masks.',
    source: flopsU32PackUnpackWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-i32-f32-convert',
    label: 'i32<->f32 convert',
    description:
      'Eight independent chains per thread, unrolled 4x: xi -> f32(xi)*a+b -> back to i32 each step. Counted as 2 ops per lane step (one convert each way; the FMA is not counted).',
    source: flopsI32F32ConvertWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-f32-f16-convert',
    label: 'f32<->f16 convert',
    description:
      'Eight independent vec2<f32> chains per thread, unrolled 4x: pack2x16float then unpack2x16float (round-trips through fp16 bits) plus a vec2 FMA to keep the chain moving. Counted as 2 ops per lane per step (one convert each way; the FMA is not counted). Unlike flops-f16-*, this needs no shader-f16 device feature — it measures the conversion, not f16 compute.',
    source: flopsF32F16ConvertWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
];
