/**
 * Raw fp32 FLOPS probe. Eight *independent* scalar FMA chains per thread,
 * with the loop body unrolled 4x, so the ALU always has work in flight:
 * a single dependent chain would measure FMA latency (plus loop overhead)
 * rather than throughput. `a`/`b`/the trip count are runtime values, and
 * every chain starts from a different value so the compiler can't merge
 * them. One tiny write per thread at the end; everything else is pure ALU.
 * 8 chains x 4 unrolled steps = 32 FMAs (64 FLOPs) per loop iteration.
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
  let a: f32 = 1.0 - f32(idx & 15u) * 0.001;
  let b: f32 = 0.5 + f32(idx & 7u) * 0.001;
  var x0: f32 = f32(idx & 255u) * 0.0001 + 0.01;
  var x1: f32 = f32(idx & 255u) * 0.0001 + 0.02;
  var x2: f32 = f32(idx & 255u) * 0.0001 + 0.03;
  var x3: f32 = f32(idx & 255u) * 0.0001 + 0.04;
  var x4: f32 = f32(idx & 255u) * 0.0001 + 0.05;
  var x5: f32 = f32(idx & 255u) * 0.0001 + 0.06;
  var x6: f32 = f32(idx & 255u) * 0.0001 + 0.07;
  var x7: f32 = f32(idx & 255u) * 0.0001 + 0.08;
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
  out[idx] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
`;
