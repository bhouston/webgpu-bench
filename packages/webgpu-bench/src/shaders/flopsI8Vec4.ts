/**
 * int8-range vec4 ops probe — {@link flopsF32Vec4Wgsl}'s eight independent,
 * 4x-unrolled `vec4` multiply-add chains, on `vec4<i32>`. Overflow wraps, so
 * the chains stay bounded. 256 integer ops per loop iteration.
 */
export const flopsI8Vec4Wgsl = /* wgsl */ `
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
  let s: i32 = i32(idx & 15u);
  let a: vec4<i32> = vec4<i32>(3 + s, 5 + s, 7 + s, 9 + s);
  let b: vec4<i32> = vec4<i32>(11 - s, 13 - s, 17 - s, 19 - s);
  var x0: vec4<i32> = vec4<i32>(i32(idx & 255u) + 1);
  var x1: vec4<i32> = vec4<i32>(i32(idx & 255u) + 2);
  var x2: vec4<i32> = vec4<i32>(i32(idx & 255u) + 3);
  var x3: vec4<i32> = vec4<i32>(i32(idx & 255u) + 4);
  var x4: vec4<i32> = vec4<i32>(i32(idx & 255u) + 5);
  var x5: vec4<i32> = vec4<i32>(i32(idx & 255u) + 6);
  var x6: vec4<i32> = vec4<i32>(i32(idx & 255u) + 7);
  var x7: vec4<i32> = vec4<i32>(i32(idx & 255u) + 8);
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
