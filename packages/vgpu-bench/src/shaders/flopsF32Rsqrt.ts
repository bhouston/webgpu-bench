/**
 * fp32 rsqrt-add FLOPS probe. Same shape as {@link flopsF32ScalarWgsl} but
 * `x = inverseSqrt(x) + b` in place of the FMA. inverseSqrt(x) is always
 * positive for x > 0, so `x` is bounded below by `b` from the second step
 * on — always positive, never blows up. This is the op behind every
 * `normalize()`: most GPUs have a dedicated fast-rsqrt path distinct from
 * sqrt-then-divide, which is what the gap against flops-f32-sqrt shows. 64
 * ops (32 rsqrts + 32 adds) per loop iteration, same MAC-as-2 convention as
 * the multiply version.
 */
export const flopsF32RsqrtWgsl = /* wgsl */ `
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
    x0 = inverseSqrt(x0) + b;
    x1 = inverseSqrt(x1) + b;
    x2 = inverseSqrt(x2) + b;
    x3 = inverseSqrt(x3) + b;
    x4 = inverseSqrt(x4) + b;
    x5 = inverseSqrt(x5) + b;
    x6 = inverseSqrt(x6) + b;
    x7 = inverseSqrt(x7) + b;
    x0 = inverseSqrt(x0) + b;
    x1 = inverseSqrt(x1) + b;
    x2 = inverseSqrt(x2) + b;
    x3 = inverseSqrt(x3) + b;
    x4 = inverseSqrt(x4) + b;
    x5 = inverseSqrt(x5) + b;
    x6 = inverseSqrt(x6) + b;
    x7 = inverseSqrt(x7) + b;
    x0 = inverseSqrt(x0) + b;
    x1 = inverseSqrt(x1) + b;
    x2 = inverseSqrt(x2) + b;
    x3 = inverseSqrt(x3) + b;
    x4 = inverseSqrt(x4) + b;
    x5 = inverseSqrt(x5) + b;
    x6 = inverseSqrt(x6) + b;
    x7 = inverseSqrt(x7) + b;
    x0 = inverseSqrt(x0) + b;
    x1 = inverseSqrt(x1) + b;
    x2 = inverseSqrt(x2) + b;
    x3 = inverseSqrt(x3) + b;
    x4 = inverseSqrt(x4) + b;
    x5 = inverseSqrt(x5) + b;
    x6 = inverseSqrt(x6) + b;
    x7 = inverseSqrt(x7) + b;
  }
  out[idx] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
`;
