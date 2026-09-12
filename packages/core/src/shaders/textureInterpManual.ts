/**
 * Manual interpolation: the same "zoom in" scan and texture as
 * {@link textureInterpBuiltinWgsl}, but each thread fetches the two
 * neighbouring texels with unfiltered `textureLoad` and lerps them itself
 * with `mix`. No sampler, no filterable-format requirement — this is what
 * the built-in bilinear path is competing against.
 */
export const textureInterpManualWgsl = /* wgsl */ `
struct Params {
  threads: u32,
  iterations: u32,
  texelCount: u32,
};

const STEP: f32 = 0.25;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= params.threads) { return; }
  let phase = f32(t) / f32(params.threads);
  let stepNorm = STEP / f32(params.texelCount);
  var sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    let u = fract(phase + f32(i) * stepNorm);
    // Normalized filtering places texel centers at (i + 0.5) / width.
    let pos = u * f32(params.texelCount) - 0.5;
    let base = i32(floor(pos));
    let width = i32(params.texelCount);
    let i0 = u32((base + width) % width);
    let i1 = (i0 + 1u) % params.texelCount;
    let frac = pos - floor(pos);
    let a = textureLoad(tex, vec2<u32>(i0, 0u), 0);
    let b = textureLoad(tex, vec2<u32>(i1, 0u), 0);
    sum = sum + mix(a, b, frac);
  }
  out[t] = sum.x + sum.y + sum.z + sum.w;
}
`;
