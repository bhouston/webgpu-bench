/**
 * fp16 vec4 FLOPS probe — {@link flopsF32Vec4Wgsl}'s eight independent,
 * 4x-unrolled `vec4` FMA chains, but every operand and accumulator is `f16`.
 * 256 FLOPs per loop iteration. Only GPUs with packed-half ALUs run this
 * faster than the fp32 version.
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
  var x0: vec4<f16> = vec4<f16>(f16(idx & 255u) * f16(0.0001) + f16(0.01));
  var x1: vec4<f16> = vec4<f16>(f16(idx & 255u) * f16(0.0001) + f16(0.02));
  var x2: vec4<f16> = vec4<f16>(f16(idx & 255u) * f16(0.0001) + f16(0.03));
  var x3: vec4<f16> = vec4<f16>(f16(idx & 255u) * f16(0.0001) + f16(0.04));
  var x4: vec4<f16> = vec4<f16>(f16(idx & 255u) * f16(0.0001) + f16(0.05));
  var x5: vec4<f16> = vec4<f16>(f16(idx & 255u) * f16(0.0001) + f16(0.06));
  var x6: vec4<f16> = vec4<f16>(f16(idx & 255u) * f16(0.0001) + f16(0.07));
  var x7: vec4<f16> = vec4<f16>(f16(idx & 255u) * f16(0.0001) + f16(0.08));
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
  let t = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
  out[idx] = f32(t.x + t.y + t.z + t.w);
}
`;
