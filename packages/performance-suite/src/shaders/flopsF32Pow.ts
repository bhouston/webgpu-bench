/**
 * fp32 pow-add FLOPS probe. Same shape as {@link flopsF32ScalarWgsl} but
 * `x = pow(x, e) + b` in place of the FMA. `e` is a per-lane exponent in
 * (0, 1): for any x > 0, x^e is a contraction towards 1, so adding a
 * positive `b` each step keeps the chain bounded and positive forever. 64
 * ops (32 pows + 32 adds) per loop iteration, same MAC-as-2 convention as
 * the multiply version.
 */
export const flopsF32PowWgsl = /* wgsl */ `
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
  let e: f32 = 0.4 + f32(idx & 15u) * 0.02;
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
    x0 = pow(x0, e) + b;
    x1 = pow(x1, e) + b;
    x2 = pow(x2, e) + b;
    x3 = pow(x3, e) + b;
    x4 = pow(x4, e) + b;
    x5 = pow(x5, e) + b;
    x6 = pow(x6, e) + b;
    x7 = pow(x7, e) + b;
    x0 = pow(x0, e) + b;
    x1 = pow(x1, e) + b;
    x2 = pow(x2, e) + b;
    x3 = pow(x3, e) + b;
    x4 = pow(x4, e) + b;
    x5 = pow(x5, e) + b;
    x6 = pow(x6, e) + b;
    x7 = pow(x7, e) + b;
    x0 = pow(x0, e) + b;
    x1 = pow(x1, e) + b;
    x2 = pow(x2, e) + b;
    x3 = pow(x3, e) + b;
    x4 = pow(x4, e) + b;
    x5 = pow(x5, e) + b;
    x6 = pow(x6, e) + b;
    x7 = pow(x7, e) + b;
    x0 = pow(x0, e) + b;
    x1 = pow(x1, e) + b;
    x2 = pow(x2, e) + b;
    x3 = pow(x3, e) + b;
    x4 = pow(x4, e) + b;
    x5 = pow(x5, e) + b;
    x6 = pow(x6, e) + b;
    x7 = pow(x7, e) + b;
  }
  out[idx] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
`;
