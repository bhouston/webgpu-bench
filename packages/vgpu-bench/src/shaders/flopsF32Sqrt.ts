/**
 * fp32 sqrt-add FLOPS probe. Same shape as {@link flopsF32ScalarWgsl} but
 * `x = sqrt(x) + b` in place of the FMA: sqrt(x) for x > 1 is a contraction
 * towards 1, so adding a positive `b` each step keeps the chain bounded and
 * positive forever. 64 ops (32 sqrts + 32 adds) per loop iteration, same
 * MAC-as-2 convention as the multiply version.
 */
export const flopsF32SqrtWgsl = /* wgsl */ `
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
    x0 = sqrt(x0) + b;
    x1 = sqrt(x1) + b;
    x2 = sqrt(x2) + b;
    x3 = sqrt(x3) + b;
    x4 = sqrt(x4) + b;
    x5 = sqrt(x5) + b;
    x6 = sqrt(x6) + b;
    x7 = sqrt(x7) + b;
    x0 = sqrt(x0) + b;
    x1 = sqrt(x1) + b;
    x2 = sqrt(x2) + b;
    x3 = sqrt(x3) + b;
    x4 = sqrt(x4) + b;
    x5 = sqrt(x5) + b;
    x6 = sqrt(x6) + b;
    x7 = sqrt(x7) + b;
    x0 = sqrt(x0) + b;
    x1 = sqrt(x1) + b;
    x2 = sqrt(x2) + b;
    x3 = sqrt(x3) + b;
    x4 = sqrt(x4) + b;
    x5 = sqrt(x5) + b;
    x6 = sqrt(x6) + b;
    x7 = sqrt(x7) + b;
    x0 = sqrt(x0) + b;
    x1 = sqrt(x1) + b;
    x2 = sqrt(x2) + b;
    x3 = sqrt(x3) + b;
    x4 = sqrt(x4) + b;
    x5 = sqrt(x5) + b;
    x6 = sqrt(x6) + b;
    x7 = sqrt(x7) + b;
  }
  out[idx] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
`;
