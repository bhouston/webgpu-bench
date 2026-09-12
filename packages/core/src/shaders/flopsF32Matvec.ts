/**
 * Register-resident matvec tile, fp32: a 4-row x 8-column weight tile held in
 * registers (8 x vec4<f32>, hashed from the thread id so every thread has a
 * different tile), applied to an 8-wide input vector every iteration. Each
 * output row is two 4-wide dot() calls; the 4 outputs then become the next
 * 4 input lanes (the other 4 shift over), so the loop is genuinely
 * loop-carried and can't be hoisted. This is the same dot-product-and-
 * accumulate shape as a real matvec/GEMV inner loop, but with no storage
 * buffer traffic at all: it isolates how fast this dtype's dot products run
 * on the ALU once bandwidth is out of the picture. 32 multiplies + 32 adds (including forcing): 64 FLOPs per
 * loop iteration.
 */
export const flopsF32MatvecWgsl = /* wgsl */ `
struct Params {
  threads: u32,
  iterations: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

// Hashes a seed into four pseudo-random weights in about [-1/32, 1/32] (small
// enough that the 8-wide dot products below are a contraction, so the
// recurrence stays bounded for any iteration count).
fn weights(seed: u32) -> vec4<f32> {
  let h = seed * 0x9e3779b1u;
  let bytes = vec4<u32>(h & 0xffu, (h >> 8u) & 0xffu, (h >> 16u) & 0xffu, h >> 24u);
  return (vec4<f32>(bytes) - vec4<f32>(127.5)) * (1.0 / 4096.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.threads) {
    return;
  }
  let w00: vec4<f32> = weights(idx * 8u + 1u);
  let w01: vec4<f32> = weights(idx * 8u + 2u);
  let w10: vec4<f32> = weights(idx * 8u + 3u);
  let w11: vec4<f32> = weights(idx * 8u + 4u);
  let w20: vec4<f32> = weights(idx * 8u + 5u);
  let w21: vec4<f32> = weights(idx * 8u + 6u);
  let w30: vec4<f32> = weights(idx * 8u + 7u);
  let w31: vec4<f32> = weights(idx * 8u + 8u);
  var x0: vec4<f32> = vec4<f32>(f32(idx & 255u) * 0.001 + 0.1, 0.2, 0.3, 0.4);
  var x1: vec4<f32> = vec4<f32>(f32(idx & 255u) * 0.001 + 0.2, 0.4, 0.6, 0.8);
  var c = vec4<f32>(0.125, 0.25, 0.375, 0.5);
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    let o0 = dot(w00, x0) + dot(w01, x1) + c.x;
    let o1 = dot(w10, x0) + dot(w11, x1) + c.y;
    let o2 = dot(w20, x0) + dot(w21, x1) + c.z;
    let o3 = dot(w30, x0) + dot(w31, x1) + c.w;
    x0 = x1;
    x1 = vec4<f32>(o0, o1, o2, o3);
    c = c.yzwx; // Nonzero rotating forcing prevents decay to zero.
  }
  out[idx] = x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
}
`;
