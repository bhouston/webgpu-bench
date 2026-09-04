/**
 * i32<->f32 conversion FLOPS probe. Same shape as {@link flopsF32ScalarWgsl}
 * but each step is `xf = f32(xi) * a + b; xi = i32(xf)`: two conversions
 * plus the same FMA as the fp32 scalar test, so the delta against that test
 * isolates conversion cost. `a` < 1 keeps the round-tripped integer bounded.
 * 128 ops (32 x [f32-convert, multiply, add, i32-convert]) per loop
 * iteration.
 */
export const flopsI32F32ConvertWgsl = /* wgsl */ `
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
  let a: f32 = 0.9 + f32(idx & 15u) * 0.005;
  let b: f32 = 3.0 + f32(idx & 7u) * 0.1;
  var x0: i32 = i32(idx & 255u) + 100;
  var x1: i32 = i32(idx & 255u) + 200;
  var x2: i32 = i32(idx & 255u) + 300;
  var x3: i32 = i32(idx & 255u) + 400;
  var x4: i32 = i32(idx & 255u) + 500;
  var x5: i32 = i32(idx & 255u) + 600;
  var x6: i32 = i32(idx & 255u) + 700;
  var x7: i32 = i32(idx & 255u) + 800;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x0 = i32(f32(x0) * a + b);
    x1 = i32(f32(x1) * a + b);
    x2 = i32(f32(x2) * a + b);
    x3 = i32(f32(x3) * a + b);
    x4 = i32(f32(x4) * a + b);
    x5 = i32(f32(x5) * a + b);
    x6 = i32(f32(x6) * a + b);
    x7 = i32(f32(x7) * a + b);
    x0 = i32(f32(x0) * a + b);
    x1 = i32(f32(x1) * a + b);
    x2 = i32(f32(x2) * a + b);
    x3 = i32(f32(x3) * a + b);
    x4 = i32(f32(x4) * a + b);
    x5 = i32(f32(x5) * a + b);
    x6 = i32(f32(x6) * a + b);
    x7 = i32(f32(x7) * a + b);
    x0 = i32(f32(x0) * a + b);
    x1 = i32(f32(x1) * a + b);
    x2 = i32(f32(x2) * a + b);
    x3 = i32(f32(x3) * a + b);
    x4 = i32(f32(x4) * a + b);
    x5 = i32(f32(x5) * a + b);
    x6 = i32(f32(x6) * a + b);
    x7 = i32(f32(x7) * a + b);
    x0 = i32(f32(x0) * a + b);
    x1 = i32(f32(x1) * a + b);
    x2 = i32(f32(x2) * a + b);
    x3 = i32(f32(x3) * a + b);
    x4 = i32(f32(x4) * a + b);
    x5 = i32(f32(x5) * a + b);
    x6 = i32(f32(x6) * a + b);
    x7 = i32(f32(x7) * a + b);
  }
  out[idx] = f32(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7);
}
`;
