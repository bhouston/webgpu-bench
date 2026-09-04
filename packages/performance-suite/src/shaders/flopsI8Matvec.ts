/**
 * Register-resident matvec tile, int8-range weights unpacked to i32: a 4-row x 8-column weight tile held in
 * registers (8 x vec4<i32>, hashed from the thread id so every thread has a
 * different tile), applied to an 8-wide input vector every iteration. Each
 * output row is two 4-wide integer dot() calls; the 4 outputs then become the next
 * 4 input lanes (the other 4 shift over), so the loop is genuinely
 * loop-carried and can't be hoisted. This is the same dot-product-and-
 * accumulate shape as a real matvec/GEMV inner loop, but with no storage
 * buffer traffic at all: it isolates how fast this dtype's dot products run
 * on the ALU once bandwidth is out of the picture. 32 MACs (64 FLOPs) per
 * loop iteration.
 * The outputs are shifted back down into int8 range before being fed back
 * (integer wraparound keeps it bounded regardless).
 */
export const flopsI8MatvecWgsl = /* wgsl */ `
struct Params {
  threads: u32,
  iterations: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

// Hashes a seed into four int8-range weights, held unpacked as a vec4<i32>
// (what a dequantize-on-load kernel holds after unpack4xI8).
fn weights(seed: u32) -> vec4<i32> {
  return unpack4xI8(seed * 0x9e3779b1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.threads) {
    return;
  }
  let w00: vec4<i32> = weights(idx * 8u + 1u);
  let w01: vec4<i32> = weights(idx * 8u + 2u);
  let w10: vec4<i32> = weights(idx * 8u + 3u);
  let w11: vec4<i32> = weights(idx * 8u + 4u);
  let w20: vec4<i32> = weights(idx * 8u + 5u);
  let w21: vec4<i32> = weights(idx * 8u + 6u);
  let w30: vec4<i32> = weights(idx * 8u + 7u);
  let w31: vec4<i32> = weights(idx * 8u + 8u);
  var x0: vec4<i32> = vec4<i32>(i32(idx & 255u) + 1, 2, 3, 4);
  var x1: vec4<i32> = vec4<i32>(i32(idx & 255u) + 2, 4, 6, 8);
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    let o0 = dot(w00, x0) + dot(w01, x1);
    let o1 = dot(w10, x0) + dot(w11, x1);
    let o2 = dot(w20, x0) + dot(w21, x1);
    let o3 = dot(w30, x0) + dot(w31, x1);
    x0 = x1;
    x1 = vec4<i32>(o0, o1, o2, o3) >> vec4<u32>(4u);
  }
  out[idx] = f32(x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w);
}
`;
