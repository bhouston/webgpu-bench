/**
 * int8 packed-dot-product FLOPS probe: uses the `packed_4x8_integer_dot_product`
 * extension's `dot4I8Packed` builtin directly (two u32s, each four packed
 * signed int8 lanes) in a tight accumulation loop, to
 * isolate the instruction's peak throughput. Eight independent accumulators
 * per thread so the number is throughput, not the latency of one dependent
 * chain. Each `a` is fed back from its own running accumulator every
 * iteration (not just `idx`-derived and fixed), so `dot4I8Packed(a, b)` is
 * genuinely loop-carried — a compiler that
 * noticed `a`/`b` were loop-invariant could otherwise hoist the call out of
 * the loop and replace all `iterations` of it with one call + a multiply,
 * which is exactly what happened before this fix (the loop collapsed to
 * ~free, and the dispatch finished faster than the timestamp-query clock's
 * resolution, reading back as a literal 0ns elapsed). 8 dots x 4 MACs =
 * 64 ops per loop iteration.
 *
 * `packed_4x8_integer_dot_product` is a WGSL *language extension* (gated by
 * `navigator.gpu.wgslLanguageFeatures`, checked as `ctx.info.supportsI8Dot`
 * in flopsCommon.ts), not an enable-extension tied to a GPUFeatureName like
 * `f16` — so it's opted into with the `requires` directive, not `enable`.
 * Using `enable` here fails shader compilation ("Expected 'clip_distances',
 * 'f16', or 'primitive_index'" — the compiler listing valid enable-extension
 * names), which used to go undetected and read back as a silent 0ns result.
 */
export const flopsI8Dp4aWgsl = /* wgsl */ `
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
  var acc0: i32 = i32(idx & 255u) + 0;
  var a1: u32 = 0x01020301u ^ (idx * 0x9e3779b1u);
  var acc1: i32 = i32(idx & 255u) + 1;
  var a2: u32 = 0x01020302u ^ (idx * 0x9e3779b1u);
  var acc2: i32 = i32(idx & 255u) + 2;
  var a3: u32 = 0x01020303u ^ (idx * 0x9e3779b1u);
  var acc3: i32 = i32(idx & 255u) + 3;
  var a4: u32 = 0x01020304u ^ (idx * 0x9e3779b1u);
  var acc4: i32 = i32(idx & 255u) + 4;
  var a5: u32 = 0x01020305u ^ (idx * 0x9e3779b1u);
  var acc5: i32 = i32(idx & 255u) + 5;
  var a6: u32 = 0x01020306u ^ (idx * 0x9e3779b1u);
  var acc6: i32 = i32(idx & 255u) + 6;
  var a7: u32 = 0x01020307u ^ (idx * 0x9e3779b1u);
  var acc7: i32 = i32(idx & 255u) + 7;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    acc0 = acc0 + dot4I8Packed(a0, b);
    a0 = a0 ^ u32(acc0);
    acc1 = acc1 + dot4I8Packed(a1, b);
    a1 = a1 ^ u32(acc1);
    acc2 = acc2 + dot4I8Packed(a2, b);
    a2 = a2 ^ u32(acc2);
    acc3 = acc3 + dot4I8Packed(a3, b);
    a3 = a3 ^ u32(acc3);
    acc4 = acc4 + dot4I8Packed(a4, b);
    a4 = a4 ^ u32(acc4);
    acc5 = acc5 + dot4I8Packed(a5, b);
    a5 = a5 ^ u32(acc5);
    acc6 = acc6 + dot4I8Packed(a6, b);
    a6 = a6 ^ u32(acc6);
    acc7 = acc7 + dot4I8Packed(a7, b);
    a7 = a7 ^ u32(acc7);
  }
  out[idx] = f32(acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7);
}
`;
