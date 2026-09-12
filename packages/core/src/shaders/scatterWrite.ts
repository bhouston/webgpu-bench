/**
 * Scatter writes through a permutation of a power-of-two window. Each
 * destination has exactly one writer per dispatch; the odd multiplier is
 * invertible modulo the window size. Reports logical bytes stored once
 * across that window, rather than repeatedly racing on a smaller footprint.
 */
export const scatterWriteWgsl = /* wgsl */ `
struct Params {
  count: u32,
  threads: u32,
  mask: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  for (var i: u32 = t; i < params.count; i = i + params.threads) {
    let destination = (i * 2654435761u + 1013904223u) & params.mask;
    out[destination] = vec4<f32>(f32(i));
  }
}
`;
