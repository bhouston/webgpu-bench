/**
 * Read-bandwidth probe: each thread streams a run of `vec4<f32>` values out
 * of a large storage buffer and folds them together with plain addition (no
 * second buffer, no multiply) — the cheapest possible "prove you read it"
 * op — then writes a single scalar. Reads dominate; writes are negligible
 * (one f32 per thread vs. `cols4` vec4s read), so throughput here is close
 * to the device's peak storage-buffer read bandwidth.
 */
export const streamReadWgsl = /* wgsl */ `
struct Params {
  rows: u32,
  cols4: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }
  var sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  let base = row * params.cols4;
  for (var c: u32 = 0u; c < params.cols4; c = c + 1u) {
    sum = sum + data[base + c];
  }
  out[row] = sum.x + sum.y + sum.z + sum.w;
}
`;
