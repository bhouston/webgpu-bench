/**
 * fp16 rsqrt-add FLOPS probe. Same shape as {@link flopsF32RsqrtWgsl} but
 * every operand and accumulator is `f16`, so the chain runs entirely in
 * half precision (only the final store converts back to f32).
 */
export const flopsF16RsqrtWgsl = /* wgsl */ `
enable f16;

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
  let b: f16 = f16(0.5) + f16(idx & 7u) * f16(0.001);
  var x0: f16 = f16(idx & 255u) * f16(0.0001) + f16(0.01);
  var x1: f16 = f16(idx & 255u) * f16(0.0001) + f16(0.02);
  var x2: f16 = f16(idx & 255u) * f16(0.0001) + f16(0.03);
  var x3: f16 = f16(idx & 255u) * f16(0.0001) + f16(0.04);
  var x4: f16 = f16(idx & 255u) * f16(0.0001) + f16(0.05);
  var x5: f16 = f16(idx & 255u) * f16(0.0001) + f16(0.06);
  var x6: f16 = f16(idx & 255u) * f16(0.0001) + f16(0.07);
  var x7: f16 = f16(idx & 255u) * f16(0.0001) + f16(0.08);
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
  out[idx] = f32(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7);
}
`;
