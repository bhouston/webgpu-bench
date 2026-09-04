/**
 * Raw int8-range FLOPS probe — same shape as {@link flopsF32ScalarWgsl}
 * (eight independent chains, unrolled 4x), but on `i32` (WGSL has no
 * first-class i8 type outside the packed-dot-product extension). Integer
 * overflow wraps (defined two's-complement behavior in WGSL), so the chains
 * stay bounded for any iteration count. 64 integer ops per loop iteration.
 */
export const flopsI8ScalarWgsl = /* wgsl */ `
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
  let a: i32 = 3 + i32(idx & 15u);
  let b: i32 = 5 + i32(idx & 7u);
  var x0: i32 = i32(idx & 255u) + 1;
  var x1: i32 = i32(idx & 255u) + 2;
  var x2: i32 = i32(idx & 255u) + 3;
  var x3: i32 = i32(idx & 255u) + 4;
  var x4: i32 = i32(idx & 255u) + 5;
  var x5: i32 = i32(idx & 255u) + 6;
  var x6: i32 = i32(idx & 255u) + 7;
  var x7: i32 = i32(idx & 255u) + 8;
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
  out[idx] = f32(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7);
}
`;
