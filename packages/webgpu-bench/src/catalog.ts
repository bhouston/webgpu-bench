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
import { flopsU32PackUnpackWgsl } from './shaders/flopsU32PackUnpack.ts';
import { flopsI32F32ConvertWgsl } from './shaders/flopsI32F32Convert.ts';
import { flopsF32F16ConvertWgsl } from './shaders/flopsF32F16Convert.ts';
import { flopsI32F16ConvertWgsl } from './shaders/flopsI32F16Convert.ts';

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
      'One thread per row; streams a large buffer in via vec4<f32> loads and addition only, writes one scalar. Read-bandwidth-bound.',
    source: streamReadWgsl,
    category: 'bandwidth',
    metric: BYTES_METRIC,
  },
  {
    id: 'write-bandwidth',
    label: 'Write bandwidth',
    description:
      'One thread per row; stores computed vec4<f32> values into a large buffer with no buffer reads. Write-bandwidth-bound.',
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
      'A single FMA chain held in a vec4<f32> register. On scalar-SIMT GPUs (Apple, NVIDIA, AMD) this compiles to 4 independent scalar FMAs per step, so it measures how well 4-wide instruction-level parallelism hides latency, not a wider ALU.',
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
      'Same single FMA chain as the fp32 vec4 test, but held in a vec4<f16> register: 4 independent half-precision lanes per step. Only GPUs with packed-half ALUs run this faster than fp32.',
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
    description:
      'Same single multiply-add chain as the fp32 vec4 test, but held in a vec4<i32> register: 4 independent integer lanes per step.',
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
      "One thread per lane, accumulating dot4I8Packed(a, b) — the packed_4x8_integer_dot_product extension's 4-wide int8 dot-product instruction — in a tight loop to measure its peak throughput in isolation.",
    source: flopsI8Dp4aWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-f32-div',
    label: 'fp32 div FLOPS',
    description:
      'Eight independent scalar f32 divide-add chains per thread, unrolled 4x — the fp32 scalar FMA test with divide in place of multiply, so the gap between the two isolates the cost of division.',
    source: flopsF32DivWgsl,
    category: 'compute',
    metric: FLOPS_METRIC,
  },
  {
    id: 'flops-i32-div',
    label: 'i32 div FLOPS',
    description:
      'Eight independent scalar i32 divide-add chains per thread, unrolled 4x — the int8-range scalar test with divide in place of multiply. Integer division is typically the slowest basic ALU op on a GPU.',
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
    id: 'flops-u32-packunpack',
    label: 'u32 byte pack/unpack ops',
    description:
      'Eight independent u32 lanes, each step unpacking 4 bytes via shift+mask, incrementing them, and repacking the same way — no pack4x8 (or unpack4x8) builtin, just the bit-twiddling those compile to. 25 ops/lane/step, no unrolling.',
    source: flopsU32PackUnpackWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-i32-f32-convert',
    label: 'i32<->f32 convert FLOPS',
    description:
      'Eight independent chains per thread, unrolled 4x: xi -> f32(xi)*a+b -> back to i32 each step. Same FMA as the fp32 scalar test plus a convert on each side, so the gap against that test isolates int<->float conversion cost.',
    source: flopsI32F32ConvertWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-f32-f16-convert',
    label: 'f32<->f16 convert FLOPS',
    description:
      'Eight independent vec2<f32> chains per thread, unrolled 4x: pack2x16float then unpack2x16float (round-trips through fp16 bits) plus a vec2 FMA to keep the chain moving. Unlike flops-f16-*, this needs no shader-f16 device feature — it measures the conversion, not f16 compute.',
    source: flopsF32F16ConvertWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
  {
    id: 'flops-i32-f16-convert',
    label: 'i32<->f16 convert FLOPS',
    description:
      'Eight independent i32 chains per thread, unrolled 4x: xi -> f32 -> pack2x16float -> unpack2x16float -> f32*a+b -> i32. There is no native int<->f16 conversion, so this is what the real path (through f32) costs.',
    source: flopsI32F16ConvertWgsl,
    category: 'compute',
    metric: OPS_METRIC,
  },
];
