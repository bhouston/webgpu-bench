/**
 * int8-range Mat SIMD FLOPS probe. WGSL has no `mat4x4<i32>` type (matrices
 * are f32/f16-only), so the 4x4 integer matrix is emulated as four
 * `vec4<i32>` rows and the matvec is built from `dot()` (which WGSL defines
 * for integer vectors too) — the same 4 dot-products + write pattern a
 * real int4x4 * int4 multiply would compile down to.
 */
export const flopsI8Mat4Wgsl = /* wgsl */ `
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
  let s: i32 = i32(idx & 15u) + 1;
  let m0: vec4<i32> = vec4<i32>(s, 1, 1, 1);
  let m1: vec4<i32> = vec4<i32>(1, s, 1, 1);
  let m2: vec4<i32> = vec4<i32>(1, 1, s, 1);
  let m3: vec4<i32> = vec4<i32>(1, 1, 1, s);
  let c: vec4<i32> = vec4<i32>(s, s, s, s);
  var x: vec4<i32> = vec4<i32>(i32(idx & 255u));
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x = vec4<i32>(dot(m0, x), dot(m1, x), dot(m2, x), dot(m3, x)) + c;
  }
  out[idx] = f32(x.x + x.y + x.z + x.w);
}
`;
