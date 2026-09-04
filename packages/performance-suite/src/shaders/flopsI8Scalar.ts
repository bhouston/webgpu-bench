/**
 * Raw int8-range FLOPS probe: same shape as {@link flopsF32ScalarWgsl}, but
 * the accumulator and operands are `i32` (WGSL has no first-class i8 type
 * outside the packed-dot-product extension). Integer overflow wraps
 * (defined two's-complement behavior in WGSL), so the chain stays bounded
 * for any iteration count with no damping needed.
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
  var x: i32 = i32(idx & 255u);
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x = x * a + b;
  }
  out[idx] = f32(x);
}
`;
