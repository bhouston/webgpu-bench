/**
 * fp32 mat4 FLOPS probe: `x = m * x + c` chained in a register, `m` a
 * 4x4 matrix. `m` is a row-stochastic contraction (rows sum to 1, every
 * entry runtime-derived and non-zero) so the recurrence stays bounded for
 * any iteration count while still exercising a full, non-degenerate
 * mat4x4 * vec4 multiply every iteration.
 */
export const flopsF32Mat4Wgsl = /* wgsl */ `
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
  let s: f32 = f32(idx & 15u) * 0.001 + 0.0001;
  let m: mat4x4<f32> = mat4x4<f32>(
    vec4<f32>(1.0 - 3.0 * s, s, s, s),
    vec4<f32>(s, 1.0 - 3.0 * s, s, s),
    vec4<f32>(s, s, 1.0 - 3.0 * s, s),
    vec4<f32>(s, s, s, 1.0 - 3.0 * s),
  );
  let c: vec4<f32> = vec4<f32>(s, s, s, s);
  var x: vec4<f32> = vec4<f32>(f32(idx & 255u) * 0.0001);
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x = m * x + c;
  }
  out[idx] = x.x + x.y + x.z + x.w;
}
`;
