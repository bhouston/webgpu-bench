/**
 * i32 divide-add ops probe. Same shape as {@link flopsI8ScalarWgsl} (eight
 * independent chains, unrolled 4x) but with the multiply replaced by integer
 * division: `x = x / a + b`. `a` is always >= 2 so each step shrinks `x`
 * (bounded, no overflow) and never divides by zero. 64 ops (32 divides + 32
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
  let a: i32 = 2 + i32(idx & 3u);
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
    x0 = x0 / a + b;
    x1 = x1 / a + b;
    x2 = x2 / a + b;
    x3 = x3 / a + b;
    x4 = x4 / a + b;
    x5 = x5 / a + b;
    x6 = x6 / a + b;
    x7 = x7 / a + b;
    x0 = x0 / a + b;
    x1 = x1 / a + b;
    x2 = x2 / a + b;
    x3 = x3 / a + b;
    x4 = x4 / a + b;
    x5 = x5 / a + b;
    x6 = x6 / a + b;
    x7 = x7 / a + b;
    x0 = x0 / a + b;
    x1 = x1 / a + b;
    x2 = x2 / a + b;
    x3 = x3 / a + b;
    x4 = x4 / a + b;
    x5 = x5 / a + b;
    x6 = x6 / a + b;
    x7 = x7 / a + b;
    x0 = x0 / a + b;
    x1 = x1 / a + b;
    x2 = x2 / a + b;
    x3 = x3 / a + b;
    x4 = x4 / a + b;
    x5 = x5 / a + b;
    x6 = x6 / a + b;
    x7 = x7 / a + b;
  }
  out[idx] = f32(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7);
}
`;
