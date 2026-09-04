/**
 * fp16 Vec SIMD FLOPS probe — {@link flopsF16ScalarWgsl}'s FMA chain held in
 * a `vec4<f16>` register, so every multiply/add is a 4-wide half-precision
 * SIMD op.
 */
export const flopsF16Vec4Wgsl = /* wgsl */ `
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
  let s: f16 = f16(idx & 15u) * f16(0.001);
  let one: f16 = f16(1.0);
  let a: vec4<f16> = vec4<f16>(one + s, one + f16(2.0) * s, one + f16(3.0) * s, one + f16(4.0) * s);
  let b: vec4<f16> = vec4<f16>(one - s, one - f16(2.0) * s, one - f16(3.0) * s, one - f16(4.0) * s);
  var x: vec4<f16> = vec4<f16>(f16(idx & 255u) * f16(0.0001));
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x = x * a + b;
  }
  out[idx] = f32(x.x + x.y + x.z + x.w);
}
`;
