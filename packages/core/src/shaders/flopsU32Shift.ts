/**
 * Variable-amount u32 shift probe. Eight independent lanes, unrolled 4x;
 * even lanes shift left, odd lanes shift right, and every step XORs a
 * per-thread constant back in so shifted-out bits are refilled and the lane
 * never collapses to zero. The shift amount comes from the loop counter
 * (`i & 31`, offset by 8 per unrolled step), so across any 32 iterations
 * every lane sees every one of the 32 possible amounts: the number reported
 * is the average over all of them. Counted as 1 op per lane step (the
 * shift; the XOR and amount arithmetic are not counted): 32 ops per loop
 * iteration.
 */
export const flopsU32ShiftWgsl = /* wgsl */ `
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
  let k: u32 = (idx * 2654435761u) | 1u;
  var x0: u32 = k;
  var x1: u32 = k + 1u;
  var x2: u32 = k + 2u;
  var x3: u32 = k + 3u;
  var x4: u32 = k + 4u;
  var x5: u32 = k + 5u;
  var x6: u32 = k + 6u;
  var x7: u32 = k + 7u;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    let s0 = i & 31u;
    let s1 = (i + 8u) & 31u;
    let s2 = (i + 16u) & 31u;
    let s3 = (i + 24u) & 31u;
    x0 = (x0 << s0) ^ k;
    x1 = (x1 >> s0) ^ k;
    x2 = (x2 << s0) ^ k;
    x3 = (x3 >> s0) ^ k;
    x4 = (x4 << s0) ^ k;
    x5 = (x5 >> s0) ^ k;
    x6 = (x6 << s0) ^ k;
    x7 = (x7 >> s0) ^ k;
    x0 = (x0 << s1) ^ k;
    x1 = (x1 >> s1) ^ k;
    x2 = (x2 << s1) ^ k;
    x3 = (x3 >> s1) ^ k;
    x4 = (x4 << s1) ^ k;
    x5 = (x5 >> s1) ^ k;
    x6 = (x6 << s1) ^ k;
    x7 = (x7 >> s1) ^ k;
    x0 = (x0 << s2) ^ k;
    x1 = (x1 >> s2) ^ k;
    x2 = (x2 << s2) ^ k;
    x3 = (x3 >> s2) ^ k;
    x4 = (x4 << s2) ^ k;
    x5 = (x5 >> s2) ^ k;
    x6 = (x6 << s2) ^ k;
    x7 = (x7 >> s2) ^ k;
    x0 = (x0 << s3) ^ k;
    x1 = (x1 >> s3) ^ k;
    x2 = (x2 << s3) ^ k;
    x3 = (x3 >> s3) ^ k;
    x4 = (x4 << s3) ^ k;
    x5 = (x5 >> s3) ^ k;
    x6 = (x6 << s3) ^ k;
    x7 = (x7 >> s3) ^ k;
  }
  out[idx] = f32(x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5 ^ x6 ^ x7);
}
`;
