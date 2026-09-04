/**
 * Raw fp16 FLOPS probe — same shape as {@link flopsF32ScalarWgsl}, but every
 * operand and the accumulator itself are `f16`, so the FMA chain runs
 * entirely in half precision (only the final store converts back to f32).
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
  let a: f16 = f16(1.0) + f16(idx & 15u) * f16(0.001);
  let b: f16 = f16(1.0) - f16(idx & 7u) * f16(0.001);
  var x: f16 = f16(idx & 255u) * f16(0.0001);
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x = x * a + b;
  }
  out[idx] = f32(x);
}
`;
