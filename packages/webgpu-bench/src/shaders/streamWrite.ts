/**
 * Write-bandwidth probe: each thread stores a run of `vec4<f32>` values into
 * a large storage buffer. No buffer reads at all; the only input is the
 * thread's own id, so throughput here is close to the device's peak
 * storage-buffer write bandwidth.
 */
export const streamWriteWgsl = /* wgsl */ `
struct Params {
  rows: u32,
  cols4: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }
  let base = row * params.cols4;
  let v = vec4<f32>(f32(row), f32(row), f32(row), f32(row));
  for (var c: u32 = 0u; c < params.cols4; c = c + 1u) {
    out[base + c] = v;
  }
}
`;
