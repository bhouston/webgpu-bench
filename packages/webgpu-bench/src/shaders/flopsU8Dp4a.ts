/**
 * Unsigned twin of {@link flopsI8Dp4aWgsl}: `dot4U8Packed` (four packed
 * unsigned int8 lanes per u32) in the same eight loop-carried accumulator
 * chains, so the two numbers compare directly. Some hardware has a native
 * unsigned dp4a and some only a signed one (or neither) — this shows which.
 * 8 dots x 4 MACs = 64 ops per loop iteration.
 */
export const flopsU8Dp4aWgsl = /* wgsl */ `
requires packed_4x8_integer_dot_product;

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
  let b: u32 = 0x05060708u ^ (idx * 0x85ebca6bu);
  var a0: u32 = 0x01020300u ^ (idx * 0x9e3779b1u);
  var acc0: u32 = (idx & 255u) + 0u;
  var a1: u32 = 0x01020301u ^ (idx * 0x9e3779b1u);
  var acc1: u32 = (idx & 255u) + 1u;
  var a2: u32 = 0x01020302u ^ (idx * 0x9e3779b1u);
  var acc2: u32 = (idx & 255u) + 2u;
  var a3: u32 = 0x01020303u ^ (idx * 0x9e3779b1u);
  var acc3: u32 = (idx & 255u) + 3u;
  var a4: u32 = 0x01020304u ^ (idx * 0x9e3779b1u);
  var acc4: u32 = (idx & 255u) + 4u;
  var a5: u32 = 0x01020305u ^ (idx * 0x9e3779b1u);
  var acc5: u32 = (idx & 255u) + 5u;
  var a6: u32 = 0x01020306u ^ (idx * 0x9e3779b1u);
  var acc6: u32 = (idx & 255u) + 6u;
  var a7: u32 = 0x01020307u ^ (idx * 0x9e3779b1u);
  var acc7: u32 = (idx & 255u) + 7u;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    acc0 = acc0 + dot4U8Packed(a0, b);
    a0 = a0 ^ acc0;
    acc1 = acc1 + dot4U8Packed(a1, b);
    a1 = a1 ^ acc1;
    acc2 = acc2 + dot4U8Packed(a2, b);
    a2 = a2 ^ acc2;
    acc3 = acc3 + dot4U8Packed(a3, b);
    a3 = a3 ^ acc3;
    acc4 = acc4 + dot4U8Packed(a4, b);
    a4 = a4 ^ acc4;
    acc5 = acc5 + dot4U8Packed(a5, b);
    a5 = a5 ^ acc5;
    acc6 = acc6 + dot4U8Packed(a6, b);
    a6 = a6 ^ acc6;
    acc7 = acc7 + dot4U8Packed(a7, b);
    a7 = a7 ^ acc7;
  }
  out[idx] = f32(acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7);
}
`;
