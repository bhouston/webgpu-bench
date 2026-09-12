/**
 * Read-bandwidth probe. Grid-stride loop: on step `i` every thread `t` loads
 * `data[t + i * threads]`, so adjacent threads in a SIMD group always touch
 * adjacent `vec4<f32>`s (fully coalesced) and the loads within a thread are
 * independent (the compiler can keep several in flight). Folds with plain
 * addition — the cheapest "prove you read it" op — and writes one scalar per
 * thread, so reads dominate. The previous one-row-per-thread layout had
 * adjacent threads 16 KB apart and only 4096 threads in total: fine on an
 * M3, badly under-reports on a wide discrete GPU.
 */
export const streamReadWgsl = /* wgsl */ `
struct Params {
  count: u32,
  threads: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  var sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  for (var i: u32 = t; i < params.count; i = i + params.threads) {
    sum = sum + data[i];
  }
  out[t] = sum.x + sum.y + sum.z + sum.w;
}
`;
