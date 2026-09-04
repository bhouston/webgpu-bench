/**
 * Raw fp16 FLOPS probe — same shape as {@link flopsF32ScalarWgsl} (eight
 * independent chains, unrolled 4x), but every operand and accumulator is
 * `f16`, so the FMA chains run entirely in half precision (only the final
 * store converts back to f32). 64 FLOPs per loop iteration.
 */
export const flopsF16ScalarWgsl = /* wgsl */ `
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
  let a: f16 = f16(1.0) - f16(idx & 15u) * f16(0.001);
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
    x0 = x0 * a + b;
    x1 = x1 * a + b;
    x2 = x2 * a + b;
    x3 = x3 * a + b;
    x4 = x4 * a + b;
    x5 = x5 * a + b;
    x6 = x6 * a + b;
    x7 = x7 * a + b;
    x0 = x0 * a + b;
    x1 = x1 * a + b;
    x2 = x2 * a + b;
    x3 = x3 * a + b;
    x4 = x4 * a + b;
    x5 = x5 * a + b;
    x6 = x6 * a + b;
    x7 = x7 * a + b;
    x0 = x0 * a + b;
    x1 = x1 * a + b;
    x2 = x2 * a + b;
    x3 = x3 * a + b;
    x4 = x4 * a + b;
    x5 = x5 * a + b;
    x6 = x6 * a + b;
    x7 = x7 * a + b;
    x0 = x0 * a + b;
    x1 = x1 * a + b;
    x2 = x2 * a + b;
    x3 = x3 * a + b;
    x4 = x4 * a + b;
    x5 = x5 * a + b;
    x6 = x6 * a + b;
    x7 = x7 * a + b;
  }
  out[idx] = f32(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7);
}
`;
