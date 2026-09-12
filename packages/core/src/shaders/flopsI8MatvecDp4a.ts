/**
 * Register-resident matvec tile, packed int8: a 4-row x 8-column weight tile held in
 * registers (8 x u32 (4 packed int8 lanes each), hashed from the thread id so every thread has a
 * different tile), applied to an 8-wide input vector every iteration. Each
 * output row is two 4-wide dot4I8Packed() calls; the 4 outputs then become the next
 * 4 input lanes (the other 4 shift over), so the loop is genuinely
 * loop-carried and can't be hoisted. This is the same dot-product-and-
 * accumulate shape as a real matvec/GEMV inner loop, but with no storage
 * buffer traffic at all: it isolates how fast this dtype's dot products run
 * on the ALU once bandwidth is out of the picture. 32 MACs (64 FLOPs) per
 * loop iteration.
 * Uses the packed_4x8_integer_dot_product WGSL language extension (opted in
 * with `requires`, see flopsI8Dp4a.ts); the 4 outputs are
 * shifted down and re-packed with pack4xI8 to form the next input word.
 */
export const flopsI8MatvecDp4aWgsl = /* wgsl */ `
requires packed_4x8_integer_dot_product;

struct Params {
  threads: u32,
  iterations: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

// Hashes a seed into a u32 holding four packed int8 weights, exactly the
// storage format dot4I8Packed consumes.
fn weights(seed: u32) -> u32 {
  return seed * 0x9e3779b1u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.threads) {
    return;
  }
  let w00: u32 = weights(idx * 8u + 1u);
  let w01: u32 = weights(idx * 8u + 2u);
  let w10: u32 = weights(idx * 8u + 3u);
  let w11: u32 = weights(idx * 8u + 4u);
  let w20: u32 = weights(idx * 8u + 5u);
  let w21: u32 = weights(idx * 8u + 6u);
  let w30: u32 = weights(idx * 8u + 7u);
  let w31: u32 = weights(idx * 8u + 8u);
  var x0: u32 = (idx * 0x85ebca6bu) ^ 0x01020304u;
  var x1: u32 = (idx * 0x85ebca6bu) ^ 0x02040608u;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    let o0 = dot4I8Packed(w00, x0) + dot4I8Packed(w01, x1);
    let o1 = dot4I8Packed(w10, x0) + dot4I8Packed(w11, x1);
    let o2 = dot4I8Packed(w20, x0) + dot4I8Packed(w21, x1);
    let o3 = dot4I8Packed(w30, x0) + dot4I8Packed(w31, x1);
    x0 = x1;
    x1 = pack4xI8(vec4<i32>(o0, o1, o2, o3) >> vec4<u32>(4u));
  }
  out[idx] = f32(i32(x0) + i32(x1));
}
`;
