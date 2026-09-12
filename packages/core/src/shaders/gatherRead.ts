/**
 * Gather-read probe: the grid-stride loop of {@link streamReadWgsl}, but
 * every load hits a pseudo-random `vec4<f32>` inside a window of
 * `mask + 1` elements at the start of the buffer, instead of the next
 * coalesced one. The index comes from a per-thread LCG (top bits, since an
 * LCG's low bits cycle), so adjacent threads touch unrelated cache lines and
 * the loads within a thread stay independent of each other (throughput, not
 * latency). The same total number of vec4s is read as the linear test, so
 * the two compare directly; the window size controls the access footprint. Cache residency is
 * device- and workload-dependent.
 */
export const gatherReadWgsl = /* wgsl */ `
struct Params {
  count: u32,
  threads: u32,
  mask: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  var h: u32 = t * 2654435761u;
  var sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  for (var i: u32 = t; i < params.count; i = i + params.threads) {
    h = h * 1664525u + 1013904223u;
    sum = sum + data[(h >> 8u) & params.mask];
  }
  out[t] = sum.x + sum.y + sum.z + sum.w;
}
`;
