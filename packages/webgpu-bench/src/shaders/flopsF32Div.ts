/**
 * fp32 divide-add FLOPS probe. Same shape as {@link flopsF32ScalarWgsl}
 * (eight independent chains, unrolled 4x) but with the multiply replaced by
 * a divide: `x = a / x + b`. The *divisor* must be the loop-carried value:
 * with `x / a` the compiler hoists `1/a` out of the loop (WGSL allows the
 * approximation) and the loop silently becomes an FMA — it measured 2.9
 * TFLOP/s on an M3 vs 0.84 for a real divide. `x >= b > 0` from the first
 * step, so the chain is bounded and never divides by zero.
 * 64 ops (32 divides + 32 adds) per loop iteration, same MAC-as-2 convention
 * as the multiply version, so throughput is directly comparable.
 */
export const flopsF32DivWgsl = /* wgsl */ `
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
  let a: f32 = 1.2 + f32(idx & 15u) * 0.05;
  let b: f32 = 0.5 + f32(idx & 7u) * 0.001;
  var x0: f32 = f32(idx & 255u) * 0.0001 + 0.01;
  var x1: f32 = f32(idx & 255u) * 0.0001 + 0.02;
  var x2: f32 = f32(idx & 255u) * 0.0001 + 0.03;
  var x3: f32 = f32(idx & 255u) * 0.0001 + 0.04;
  var x4: f32 = f32(idx & 255u) * 0.0001 + 0.05;
  var x5: f32 = f32(idx & 255u) * 0.0001 + 0.06;
  var x6: f32 = f32(idx & 255u) * 0.0001 + 0.07;
  var x7: f32 = f32(idx & 255u) * 0.0001 + 0.08;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x0 = a / x0 + b;
    x1 = a / x1 + b;
    x2 = a / x2 + b;
    x3 = a / x3 + b;
    x4 = a / x4 + b;
    x5 = a / x5 + b;
    x6 = a / x6 + b;
    x7 = a / x7 + b;
    x0 = a / x0 + b;
    x1 = a / x1 + b;
    x2 = a / x2 + b;
    x3 = a / x3 + b;
    x4 = a / x4 + b;
    x5 = a / x5 + b;
    x6 = a / x6 + b;
    x7 = a / x7 + b;
    x0 = a / x0 + b;
    x1 = a / x1 + b;
    x2 = a / x2 + b;
    x3 = a / x3 + b;
    x4 = a / x4 + b;
    x5 = a / x5 + b;
    x6 = a / x6 + b;
    x7 = a / x7 + b;
    x0 = a / x0 + b;
    x1 = a / x1 + b;
    x2 = a / x2 + b;
    x3 = a / x3 + b;
    x4 = a / x4 + b;
    x5 = a / x5 + b;
    x6 = a / x6 + b;
    x7 = a / x7 + b;
  }
  out[idx] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
`;
