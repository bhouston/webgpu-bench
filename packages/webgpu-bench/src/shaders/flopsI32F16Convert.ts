/**
 * i32<->fp16 round-trip FLOPS probe: `xi -> f32 -> pack2x16float ->
 * unpack2x16float -> f32*a+b -> xi`. There's no native int<->f16 builtin, so
 * this is what that conversion actually costs — chained through f32, like
 * the real path. `a` < 1 keeps the round-tripped integer bounded. Eight
 * independent i32 lanes, unrolled 4x; 6 ops/lane/step (f32-convert, pack,
 * unpack, multiply, add, i32-convert). 192 ops per loop iteration.
 */
export const flopsI32F16ConvertWgsl = /* wgsl */ `
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
  let a: f32 = 0.9 + f32(idx & 15u) * 0.005;
  let b: f32 = 3.0 + f32(idx & 7u) * 0.1;
  var x0: i32 = i32(idx & 255u) + 100;
  var x1: i32 = i32(idx & 255u) + 200;
  var x2: i32 = i32(idx & 255u) + 300;
  var x3: i32 = i32(idx & 255u) + 400;
  var x4: i32 = i32(idx & 255u) + 500;
  var x5: i32 = i32(idx & 255u) + 600;
  var x6: i32 = i32(idx & 255u) + 700;
  var x7: i32 = i32(idx & 255u) + 800;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x0 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x0)))).x * a + b);
    x1 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x1)))).x * a + b);
    x2 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x2)))).x * a + b);
    x3 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x3)))).x * a + b);
    x4 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x4)))).x * a + b);
    x5 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x5)))).x * a + b);
    x6 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x6)))).x * a + b);
    x7 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x7)))).x * a + b);
    x0 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x0)))).x * a + b);
    x1 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x1)))).x * a + b);
    x2 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x2)))).x * a + b);
    x3 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x3)))).x * a + b);
    x4 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x4)))).x * a + b);
    x5 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x5)))).x * a + b);
    x6 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x6)))).x * a + b);
    x7 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x7)))).x * a + b);
    x0 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x0)))).x * a + b);
    x1 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x1)))).x * a + b);
    x2 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x2)))).x * a + b);
    x3 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x3)))).x * a + b);
    x4 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x4)))).x * a + b);
    x5 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x5)))).x * a + b);
    x6 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x6)))).x * a + b);
    x7 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x7)))).x * a + b);
    x0 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x0)))).x * a + b);
    x1 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x1)))).x * a + b);
    x2 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x2)))).x * a + b);
    x3 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x3)))).x * a + b);
    x4 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x4)))).x * a + b);
    x5 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x5)))).x * a + b);
    x6 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x6)))).x * a + b);
    x7 = i32(unpack2x16float(pack2x16float(vec2<f32>(f32(x7)))).x * a + b);
  }
  out[idx] = f32(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7);
}
`;
