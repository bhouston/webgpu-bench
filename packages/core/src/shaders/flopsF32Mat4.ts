/**
 * fp32 mat4 FLOPS probe: `x = m * x + c` chained in a register, `m` a
 * 4x4 matrix. `m` has distinct runtime-derived entries and row sums below 1/2.
 * A rotating nonzero forcing vector keeps the recurrence bounded and moving
 * while exercising a full, non-degenerate
 * mat4x4 * vec4 multiply every iteration.
 */
export const flopsF32Mat4Wgsl = /* wgsl */ `
struct Params {
  threads: u32,
  iterations: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

fn weights(seed: u32) -> vec4<f32> {
  let h = seed * 0x9e3779b1u;
  let bytes = vec4<u32>(h & 255u, (h >> 8u) & 255u, (h >> 16u) & 255u, h >> 24u);
  return (vec4<f32>(bytes) + vec4<f32>(32.0)) * f32(1.0 / 4096.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.threads) {
    return;
  }
  let seed = idx * 4u;
  // Every entry is in [1/128, 1/8): each absolute row sum is < 1/2.
  let m = mat4x4<f32>(weights(seed + 1u), weights(seed + 2u), weights(seed + 3u), weights(seed + 4u));
  var c = vec4<f32>(0.125, 0.25, 0.375, 0.5);
  var x = vec4<f32>(0.1, 0.2, 0.3, 0.4) + vec4<f32>(f32(idx & 255u) * f32(0.0001));
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x = m * x + c;
    c = c.yzwx; // A four-step forcing cycle avoids a constant fixed point.
  }
  out[idx] = x.x + x.y + x.z + x.w;
}
`;
