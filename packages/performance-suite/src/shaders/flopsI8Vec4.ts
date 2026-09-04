/**
 * int8-range Vec SIMD FLOPS probe: {@link flopsI8ScalarWgsl}'s integer FMA
 * chain held in a `vec4<i32>` register, so every multiply/add is a 4-wide
 * integer SIMD op.
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
  var x: vec4<i32> = vec4<i32>(i32(idx & 255u));
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x = x * a + b;
  }
  out[idx] = f32(x.x + x.y + x.z + x.w);
}
`;
