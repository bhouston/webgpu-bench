/**
 * int8 packed-dot-product FLOPS probe: uses the `packed_4x8_integer_dot_product`
 * extension's `dot4I8Packed` builtin directly (two u32s, each four packed
 * signed int8 lanes, one instruction) rather than emulating it — the same
 * builtin the `i8-packed` matvec kernel uses, run in a tight accumulation
 * loop instead of over a matrix so it isolates the instruction's peak
 * throughput. Operands are derived from the thread id so they can't be
 * constant-folded.
 */
export const flopsI8Dp4aWgsl = /* wgsl */ `
enable packed_4x8_integer_dot_product;

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
  let a: u32 = 0x01020304u ^ (idx * 0x9e3779b1u);
  let b: u32 = 0x05060708u ^ (idx * 0x85ebca6bu);
  var acc: i32 = i32(idx & 255u);
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    acc = acc + dot4I8Packed(a, b);
  }
  out[idx] = f32(acc);
}
`;
