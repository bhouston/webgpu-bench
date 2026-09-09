/**
 * fp32 ln-add FLOPS probe. Same shape as {@link flopsF32ScalarWgsl} but
 * `x = log(x) + b` (natural log) in place of the FMA. `b` is kept large
 * (10-17) so it dominates the small log(x) term and the chain stays
 * comfortably positive forever — log of a negative/zero argument is
 * undefined, so this is the stabilization that matters here. 64 ops (32
 * logs + 32 adds) per loop iteration, same MAC-as-2 convention as the
 * multiply version.
 */
export const flopsF32LogWgsl = /* wgsl */ `
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
  let b: f32 = 10.0 + f32(idx & 7u);
  var x0: f32 = 10.0 + f32(idx & 255u) * 0.01;
  var x1: f32 = 11.0 + f32(idx & 255u) * 0.01;
  var x2: f32 = 12.0 + f32(idx & 255u) * 0.01;
  var x3: f32 = 13.0 + f32(idx & 255u) * 0.01;
  var x4: f32 = 14.0 + f32(idx & 255u) * 0.01;
  var x5: f32 = 15.0 + f32(idx & 255u) * 0.01;
  var x6: f32 = 16.0 + f32(idx & 255u) * 0.01;
  var x7: f32 = 17.0 + f32(idx & 255u) * 0.01;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x0 = log(x0) + b;
    x1 = log(x1) + b;
    x2 = log(x2) + b;
    x3 = log(x3) + b;
    x4 = log(x4) + b;
    x5 = log(x5) + b;
    x6 = log(x6) + b;
    x7 = log(x7) + b;
    x0 = log(x0) + b;
    x1 = log(x1) + b;
    x2 = log(x2) + b;
    x3 = log(x3) + b;
    x4 = log(x4) + b;
    x5 = log(x5) + b;
    x6 = log(x6) + b;
    x7 = log(x7) + b;
    x0 = log(x0) + b;
    x1 = log(x1) + b;
    x2 = log(x2) + b;
    x3 = log(x3) + b;
    x4 = log(x4) + b;
    x5 = log(x5) + b;
    x6 = log(x6) + b;
    x7 = log(x7) + b;
    x0 = log(x0) + b;
    x1 = log(x1) + b;
    x2 = log(x2) + b;
    x3 = log(x3) + b;
    x4 = log(x4) + b;
    x5 = log(x5) + b;
    x6 = log(x6) + b;
    x7 = log(x7) + b;
  }
  out[idx] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
`;
