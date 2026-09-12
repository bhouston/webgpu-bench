/**
 * Scatter-write probe: {@link streamWriteWgsl} with every store aimed at a
 * pseudo-random `vec4<f32>` inside a window of `mask + 1` elements at the
 * start of the buffer (same LCG indexing as {@link gatherReadWgsl}).
 * Threads will collide on the same element — a plain (non-atomic) data
 * race, which WGSL defines as leaving an indeterminate value, not as an
 * error — and the smaller the window the more the stores contend for the
 * same cache lines, so the three window sizes show how the write path
 * handles scattered, coalescing-hostile traffic.
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
  var h: u32 = t * 2654435761u;
  let v = vec4<f32>(f32(t), f32(t), f32(t), f32(t));
  for (var i: u32 = t; i < params.count; i = i + params.threads) {
    h = h * 1664525u + 1013904223u;
    out[(h >> 8u) & params.mask] = v;
  }
}
`;
