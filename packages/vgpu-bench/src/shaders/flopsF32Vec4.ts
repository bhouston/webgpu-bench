/**
 * fp32 vec4 FLOPS probe: eight independent FMA chains, each held in a
 * `vec4<f32>` register, unrolled 4x — the same chain/unroll shape as
 * {@link flopsF32ScalarWgsl} so the two are directly comparable. (A single
 * vec4 chain measured FMA latency, not throughput: 4x slower than scalar on
 * an M3 purely from the missing ILP.) On scalar-SIMT GPUs each vec4 FMA is 4
 * scalar FMAs. 8 chains x 4 unrolled x 4 lanes = 128 FMAs (256 FLOPs) per
 * loop iteration.
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
  var x0: vec4<f32> = vec4<f32>(f32(idx & 255u) * 0.0001 + 0.01);
  var x1: vec4<f32> = vec4<f32>(f32(idx & 255u) * 0.0001 + 0.02);
  var x2: vec4<f32> = vec4<f32>(f32(idx & 255u) * 0.0001 + 0.03);
  var x3: vec4<f32> = vec4<f32>(f32(idx & 255u) * 0.0001 + 0.04);
  var x4: vec4<f32> = vec4<f32>(f32(idx & 255u) * 0.0001 + 0.05);
  var x5: vec4<f32> = vec4<f32>(f32(idx & 255u) * 0.0001 + 0.06);
  var x6: vec4<f32> = vec4<f32>(f32(idx & 255u) * 0.0001 + 0.07);
  var x7: vec4<f32> = vec4<f32>(f32(idx & 255u) * 0.0001 + 0.08);
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
  out[idx] = t.x + t.y + t.z + t.w;
}
`;
