/**
 * fp16 vec4 FLOPS probe — a single FMA chain held in a `vec4<f16>` register
 * (4 independent half-precision lanes per step; only GPUs with packed-half
 * ALUs run this faster than the fp32 vec4 probe).
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
  let half: f16 = f16(0.5);
  let one: f16 = f16(1.0);
  let a: vec4<f16> = vec4<f16>(one - s, one - f16(2.0) * s, one - f16(3.0) * s, one - f16(4.0) * s);
  let b: vec4<f16> = vec4<f16>(half + s, half + f16(2.0) * s, half + f16(3.0) * s, half + f16(4.0) * s);
  var x: vec4<f16> = vec4<f16>(f16(idx & 255u) * f16(0.0001));
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x = x * a + b;
  }
  out[idx] = f32(x.x + x.y + x.z + x.w);
}
`;
