/**
 * Builtin-throughput probes that all share one shape — the eight
 * independent lanes, unrolled 4x, of {@link flopsF32ScalarWgsl} — and
 * differ only in the lane type, its starting value, and the one expression
 * applied per step. Generated from that template so the seven kernels
 * can't drift apart; the WGSL displayed for each is the generated text.
 */
function laneKernel(opts: {
  type: 'f32' | 'u32';
  consts: string;
  init: (lane: number) => string;
  step: (x: string) => string;
}): string {
  const lanes = [0, 1, 2, 3, 4, 5, 6, 7];
  const decls = lanes.map((l) => `  var x${l}: ${opts.type} = ${opts.init(l)};`).join('\n');
  const block = lanes.map((l) => `    x${l} = ${opts.step(`x${l}`)};`).join('\n');
  const fold =
    opts.type === 'f32' ? lanes.map((l) => `x${l}`).join(' + ') : `f32(${lanes.map((l) => `x${l}`).join(' ^ ')})`;
  return /* wgsl */ `
struct Params {
  threads: u32,
  iterations: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.threads) {
    return;
  }
${opts.consts}
${decls}
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
${block}
${block}
${block}
${block}
  }
  out[idx] = ${fold};
}
`;
}

/** Same a/b/x0 as the fp32 scalar test: a < 1 keeps the chain bounded. */
const f32Consts = `  let a: f32 = 1.0 - f32(idx & 15u) * 0.001;
  let b: f32 = 0.5 + f32(idx & 7u) * 0.001;`;
const f32Init = (l: number) => `f32(idx & 255u) * 0.0001 + 0.0${l + 1}`;
const u32Consts = `  let k: u32 = (idx * 2654435761u) | 1u;`;
const u32Init = (l: number) => `k + ${l}u`;

/** `fma(x, a, b)` via the builtin, instead of the `x * a + b` the scalar test leaves for the compiler to contract. 2 FLOPs per lane step. */
export const flopsF32FmaWgsl = laneKernel({
  type: 'f32',
  consts: f32Consts,
  init: f32Init,
  step: (x) => `fma(${x}, a, b)`,
});

/** Bounded walk: add while below 1, subtract once above. 1 op per lane step (the select; the compare and add/sub are not counted). */
export const flopsF32SelectWgsl = laneKernel({
  type: 'f32',
  consts: f32Consts,
  init: f32Init,
  step: (x) => `select(${x} + b, ${x} - a, ${x} > 1.0)`,
});

/** `clamp(x * a + b, 0, 1)`: 1 op per lane step (the clamp; the FMA is not counted). Compare with the min/max kernel to see whether clamp is one instruction or two. */
export const flopsF32ClampWgsl = laneKernel({
  type: 'f32',
  consts: f32Consts,
  init: f32Init,
  step: (x) => `clamp(${x} * a + b, 0.0, 1.0)`,
});

/** `max(min(x * a + b, 1), 0)`: 2 ops per lane step (min + max; the FMA is not counted). */
export const flopsF32MinMaxWgsl = laneKernel({
  type: 'f32',
  consts: f32Consts,
  init: f32Init,
  step: (x) => `max(min(${x} * a + b, 1.0), 0.0)`,
});

/** `countOneBits(x) ^ k`: 1 op per lane step (the popcount; the XOR that keeps the lane moving is not counted). */
export const flopsU32CountOneBitsWgsl = laneKernel({
  type: 'u32',
  consts: u32Consts,
  init: u32Init,
  step: (x) => `countOneBits(${x}) ^ k`,
});

/** `firstLeadingBit(x) ^ k`: 1 op per lane step (the count-leading-zeros; the XOR is not counted). */
export const flopsU32FirstLeadingBitWgsl = laneKernel({
  type: 'u32',
  consts: u32Consts,
  init: u32Init,
  step: (x) => `firstLeadingBit(${x}) ^ k`,
});

/** `reverseBits(x) ^ k`: 1 op per lane step (the bit reverse; the XOR is not counted). */
export const flopsU32ReverseBitsWgsl = laneKernel({
  type: 'u32',
  consts: u32Consts,
  init: u32Init,
  step: (x) => `reverseBits(${x}) ^ k`,
});
