/**
 * Raw fp32 FLOPS probe: no matrices or vectors, just a long chain of scalar
 * fused-multiply-adds held in a register. `a`/`b`/the loop trip count are
 * all runtime values (derived from the thread id / a uniform), so the
 * compiler can't constant-fold or hoist the loop away. One tiny write per
 * thread at the end; everything else is pure ALU.
 */
export const flopsF32ScalarWgsl = /* wgsl */ `
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
  let a: f32 = 1.0 + f32(idx & 15u) * 0.001;
  let b: f32 = 1.0 - f32(idx & 7u) * 0.001;
  var x: f32 = f32(idx & 255u) * 0.0001;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x = x * a + b;
  }
  out[idx] = x;
}
`;
