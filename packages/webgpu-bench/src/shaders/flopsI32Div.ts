/**
 * i32 divide-add ops probe. Same shape as {@link flopsI8ScalarWgsl} (eight
 * independent chains, unrolled 4x) but with the multiply replaced by integer
 * division: `x = a / x + b`. The divisor is the loop-carried value so the
 * division can't be strength-reduced against a loop-invariant `a` (see
 * flopsF32Div.ts). `x >= b >= 5` from the first step, so it never divides by
 * zero and stays within `[b, a / b + b]`. 64 ops (32 divides + 32
 * adds) per loop iteration, same MAC-as-2 convention as the other scalar
 * probes.
 */
export const flopsI32DivWgsl = /* wgsl */ `
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
  let a: i32 = 100003 + i32(idx & 255u) * 7;
  let b: i32 = 5 + i32(idx & 7u);
  var x0: i32 = i32(idx & 255u) + 1000;
  var x1: i32 = i32(idx & 255u) + 2000;
  var x2: i32 = i32(idx & 255u) + 3000;
  var x3: i32 = i32(idx & 255u) + 4000;
  var x4: i32 = i32(idx & 255u) + 5000;
  var x5: i32 = i32(idx & 255u) + 6000;
  var x6: i32 = i32(idx & 255u) + 7000;
  var x7: i32 = i32(idx & 255u) + 8000;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x0 = a / x0 + b;
    x1 = a / x1 + b;
    x2 = a / x2 + b;
    x3 = a / x3 + b;
    x4 = a / x4 + b;
    x5 = a / x5 + b;
    x6 = a / x6 + b;
    x7 = a / x7 + b;
    x0 = a / x0 + b;
    x1 = a / x1 + b;
    x2 = a / x2 + b;
    x3 = a / x3 + b;
    x4 = a / x4 + b;
    x5 = a / x5 + b;
    x6 = a / x6 + b;
    x7 = a / x7 + b;
    x0 = a / x0 + b;
    x1 = a / x1 + b;
    x2 = a / x2 + b;
    x3 = a / x3 + b;
    x4 = a / x4 + b;
    x5 = a / x5 + b;
    x6 = a / x6 + b;
    x7 = a / x7 + b;
    x0 = a / x0 + b;
    x1 = a / x1 + b;
    x2 = a / x2 + b;
    x3 = a / x3 + b;
    x4 = a / x4 + b;
    x5 = a / x5 + b;
    x6 = a / x6 + b;
    x7 = a / x7 + b;
  }
  out[idx] = f32(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7);
}
`;
