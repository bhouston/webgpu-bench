/**
 * fp32 vec4 FLOPS probe: a single FMA chain held in a `vec4<f32>` register.
 * On scalar-SIMT GPUs (Apple, NVIDIA, AMD) this compiles to 4 independent
 * scalar FMAs per step, so it measures 4-wide instruction-level parallelism
 * rather than a wider ALU. Same runtime-derived operands/loop count as the
 * scalar probe to defeat constant folding.
 */
export const flopsF32Vec4Wgsl = /* wgsl */ `
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
  let s: f32 = f32(idx & 15u) * 0.001;
  let a: vec4<f32> = vec4<f32>(1.0 - s, 1.0 - 2.0 * s, 1.0 - 3.0 * s, 1.0 - 4.0 * s);
  let b: vec4<f32> = vec4<f32>(0.5 + s, 0.5 + 2.0 * s, 0.5 + 3.0 * s, 0.5 + 4.0 * s);
  var x: vec4<f32> = vec4<f32>(f32(idx & 255u) * 0.0001);
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x = x * a + b;
  }
  out[idx] = x.x + x.y + x.z + x.w;
}
`;
