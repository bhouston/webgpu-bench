/**
 * Built-in interpolation: each thread does a straight-line "zoom in" scan
 * across a small `rgba32float` texture, advancing a quarter of a texel per
 * sample (so most samples interpolate the same or an adjacent texel pair,
 * like scrolling slowly across a magnified texture). The GPU's texture unit
 * (its fixed-function bilinear-filtering hardware) does the interpolation
 * via `textureSampleLevel`. Needs the `float32-filterable` feature — linear
 * filtering of 32-bit float textures isn't core WebGPU.
 */
export const textureInterpBuiltinWgsl = /* wgsl */ `
struct Params {
  threads: u32,
  iterations: u32,
  texelCount: u32,
};

const STEP: f32 = 0.25;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= params.threads) { return; }
  let phase = f32(t) / f32(params.threads);
  let stepNorm = STEP / f32(params.texelCount);
  var sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    let u = fract(phase + f32(i) * stepNorm);
    sum = sum + textureSampleLevel(tex, samp, vec2<f32>(u, 0.5), 0.0);
  }
  out[t] = sum.x + sum.y + sum.z + sum.w;
}
`;
