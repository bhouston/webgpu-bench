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

fn weights(seed: u32) -> vec4<f16> {
  let h = seed * 0x9e3779b1u;
  let bytes = vec4<u32>(h & 255u, (h >> 8u) & 255u, (h >> 16u) & 255u, h >> 24u);
  return (vec4<f16>(bytes) + vec4<f16>(32.0)) * f16(1.0 / 4096.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.threads) {
    return;
  }
  let seed = idx * 4u;
  // Every entry is in [1/128, 1/8): each absolute row sum is < 1/2.
  let m = mat4x4<f16>(weights(seed + 1u), weights(seed + 2u), weights(seed + 3u), weights(seed + 4u));
  var c = vec4<f16>(0.125, 0.25, 0.375, 0.5);
  var x = vec4<f16>(0.1, 0.2, 0.3, 0.4) + vec4<f16>(f16(idx & 255u) * f16(0.0001));
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x = m * x + c;
    c = c.yzwx; // A four-step forcing cycle avoids a constant fixed point.
  }
  out[idx] = f32(x.x + x.y + x.z + x.w);
}
`;
