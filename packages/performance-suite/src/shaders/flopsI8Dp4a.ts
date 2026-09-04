/**
 * int8 packed-dot-product FLOPS probe: uses the `packed_4x8_integer_dot_product`
 * extension's `dot4I8Packed` builtin directly (two u32s, each four packed
 * signed int8 lanes, one instruction) in a tight accumulation loop, to
 * isolate the instruction's peak throughput. `a` is fed back from the
 * running accumulator every iteration (not just `idx`-derived and fixed),
 * so `dot4I8Packed(a, b)` is genuinely loop-carried — a compiler that
 * noticed `a`/`b` were loop-invariant could otherwise hoist the call out of
 * the loop and replace all `iterations` of it with one call + a multiply,
 * which is exactly what happened before this fix (the loop collapsed to
 * ~free, and the dispatch finished faster than the timestamp-query clock's
 * resolution, reading back as a literal 0ns elapsed).
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
  var a: u32 = 0x01020304u ^ (idx * 0x9e3779b1u);
  var acc: i32 = i32(idx & 255u);
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    acc = acc + dot4I8Packed(a, b);
    a = a ^ u32(acc);
  }
  out[idx] = f32(acc);
}
`;
