/**
 * fp16 mat4 FLOPS probe — {@link flopsF32Mat4Wgsl}'s bounded
 * `x = m * x + c` recurrence, but `m`, `c`, and `x` are all `f16`, so the
 * mat4x4 * vec4 multiply runs entirely in half precision.
 */
export const flopsF16Mat4Wgsl = /* wgsl */ `
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
  let s: f16 = f16(idx & 15u) * f16(0.001) + f16(0.001);
  let one: f16 = f16(1.0);
  let three_s: f16 = f16(3.0) * s;
  let m: mat4x4<f16> = mat4x4<f16>(
    vec4<f16>(one - three_s, s, s, s),
    vec4<f16>(s, one - three_s, s, s),
    vec4<f16>(s, s, one - three_s, s),
    vec4<f16>(s, s, s, one - three_s),
  );
  let c: vec4<f16> = vec4<f16>(s, s, s, s);
  var x: vec4<f16> = vec4<f16>(f16(idx & 255u) * f16(0.0001));
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x = m * x + c;
  }
  out[idx] = f32(x.x + x.y + x.z + x.w);
}
`;
