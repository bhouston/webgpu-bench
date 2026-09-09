/**
 * fp32 sin/cos FLOPS probe. Same shape as {@link flopsF32ScalarWgsl} but
 * `x = cos(sin(x))` in place of the FMA: sin then cos, both naturally
 * bounded to [-1, 1] so the chain needs no extra stabilization. 64 ops (32
 * sins + 32 coss) per loop iteration, same MAC-as-2 convention as the
 * multiply version.
 */
export const flopsF32SincosWgsl = /* wgsl */ `
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
  var x0: f32 = f32(idx & 255u) * 0.01 + 0.1;
  var x1: f32 = f32(idx & 255u) * 0.01 + 0.2;
  var x2: f32 = f32(idx & 255u) * 0.01 + 0.3;
  var x3: f32 = f32(idx & 255u) * 0.01 + 0.4;
  var x4: f32 = f32(idx & 255u) * 0.01 + 0.5;
  var x5: f32 = f32(idx & 255u) * 0.01 + 0.6;
  var x6: f32 = f32(idx & 255u) * 0.01 + 0.7;
  var x7: f32 = f32(idx & 255u) * 0.01 + 0.8;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x0 = cos(sin(x0));
    x1 = cos(sin(x1));
    x2 = cos(sin(x2));
    x3 = cos(sin(x3));
    x4 = cos(sin(x4));
    x5 = cos(sin(x5));
    x6 = cos(sin(x6));
    x7 = cos(sin(x7));
    x0 = cos(sin(x0));
    x1 = cos(sin(x1));
    x2 = cos(sin(x2));
    x3 = cos(sin(x3));
    x4 = cos(sin(x4));
    x5 = cos(sin(x5));
    x6 = cos(sin(x6));
    x7 = cos(sin(x7));
    x0 = cos(sin(x0));
    x1 = cos(sin(x1));
    x2 = cos(sin(x2));
    x3 = cos(sin(x3));
    x4 = cos(sin(x4));
    x5 = cos(sin(x5));
    x6 = cos(sin(x6));
    x7 = cos(sin(x7));
    x0 = cos(sin(x0));
    x1 = cos(sin(x1));
    x2 = cos(sin(x2));
    x3 = cos(sin(x3));
    x4 = cos(sin(x4));
    x5 = cos(sin(x5));
    x6 = cos(sin(x6));
    x7 = cos(sin(x7));
  }
  out[idx] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
`;
