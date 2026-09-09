/**
 * fp32<->fp16 conversion FLOPS probe, using the core `pack2x16float` /
 * `unpack2x16float` builtins (round each of a vec2<f32>'s lanes to fp16 and
 * back) — these don't need the `shader-f16` device feature, unlike the
 * flops-f16-* kernels which compute *in* f16. Eight independent vec2<f32>
 * chains, unrolled 4x; each step packs, unpacks, then a vec2 FMA keeps the
 * chain moving and bounded. Counted as 2 ops per lane (one f32 -> f16
 * convert and one back; the FMA is not counted): 128 ops per loop iteration.
 */
export const flopsF32F16ConvertWgsl = /* wgsl */ `
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
  let a: vec2<f32> = vec2<f32>(0.97 + f32(idx & 15u) * 0.001);
  let b: vec2<f32> = vec2<f32>(0.5 + f32(idx & 7u) * 0.001);
  var x0: vec2<f32> = vec2<f32>(f32(idx & 255u) * 0.0001 + 0.01);
  var x1: vec2<f32> = vec2<f32>(f32(idx & 255u) * 0.0001 + 0.02);
  var x2: vec2<f32> = vec2<f32>(f32(idx & 255u) * 0.0001 + 0.03);
  var x3: vec2<f32> = vec2<f32>(f32(idx & 255u) * 0.0001 + 0.04);
  var x4: vec2<f32> = vec2<f32>(f32(idx & 255u) * 0.0001 + 0.05);
  var x5: vec2<f32> = vec2<f32>(f32(idx & 255u) * 0.0001 + 0.06);
  var x6: vec2<f32> = vec2<f32>(f32(idx & 255u) * 0.0001 + 0.07);
  var x7: vec2<f32> = vec2<f32>(f32(idx & 255u) * 0.0001 + 0.08);
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    x0 = unpack2x16float(pack2x16float(x0)) * a + b;
    x1 = unpack2x16float(pack2x16float(x1)) * a + b;
    x2 = unpack2x16float(pack2x16float(x2)) * a + b;
    x3 = unpack2x16float(pack2x16float(x3)) * a + b;
    x4 = unpack2x16float(pack2x16float(x4)) * a + b;
    x5 = unpack2x16float(pack2x16float(x5)) * a + b;
    x6 = unpack2x16float(pack2x16float(x6)) * a + b;
    x7 = unpack2x16float(pack2x16float(x7)) * a + b;
    x0 = unpack2x16float(pack2x16float(x0)) * a + b;
    x1 = unpack2x16float(pack2x16float(x1)) * a + b;
    x2 = unpack2x16float(pack2x16float(x2)) * a + b;
    x3 = unpack2x16float(pack2x16float(x3)) * a + b;
    x4 = unpack2x16float(pack2x16float(x4)) * a + b;
    x5 = unpack2x16float(pack2x16float(x5)) * a + b;
    x6 = unpack2x16float(pack2x16float(x6)) * a + b;
    x7 = unpack2x16float(pack2x16float(x7)) * a + b;
    x0 = unpack2x16float(pack2x16float(x0)) * a + b;
    x1 = unpack2x16float(pack2x16float(x1)) * a + b;
    x2 = unpack2x16float(pack2x16float(x2)) * a + b;
    x3 = unpack2x16float(pack2x16float(x3)) * a + b;
    x4 = unpack2x16float(pack2x16float(x4)) * a + b;
    x5 = unpack2x16float(pack2x16float(x5)) * a + b;
    x6 = unpack2x16float(pack2x16float(x6)) * a + b;
    x7 = unpack2x16float(pack2x16float(x7)) * a + b;
    x0 = unpack2x16float(pack2x16float(x0)) * a + b;
    x1 = unpack2x16float(pack2x16float(x1)) * a + b;
    x2 = unpack2x16float(pack2x16float(x2)) * a + b;
    x3 = unpack2x16float(pack2x16float(x3)) * a + b;
    x4 = unpack2x16float(pack2x16float(x4)) * a + b;
    x5 = unpack2x16float(pack2x16float(x5)) * a + b;
    x6 = unpack2x16float(pack2x16float(x6)) * a + b;
    x7 = unpack2x16float(pack2x16float(x7)) * a + b;
  }
  let s0 = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
  out[idx] = s0.x + s0.y;
}
`;
