/**
 * Write-bandwidth probe: the grid-stride layout of {@link streamReadWgsl}
 * with the loads replaced by stores. No buffer reads at all; the only input
 * is the thread's own id.
 */
export const streamWriteWgsl = /* wgsl */ `
struct Params {
  count: u32,
  threads: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let v = vec4<f32>(f32(t), f32(t), f32(t), f32(t));
  for (var i: u32 = t; i < params.count; i = i + params.threads) {
    out[i] = v;
  }
}
`;
