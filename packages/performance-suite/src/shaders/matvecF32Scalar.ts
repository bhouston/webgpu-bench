/** Naive baseline: one thread per output row, scalar dot-product loop over f32. */
export const matvecF32ScalarWgsl = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> matrix: array<f32>;
@group(0) @binding(2) var<storage, read> vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }
  var sum: f32 = 0.0;
  let base = row * params.cols;
  for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
    sum = sum + matrix[base + c] * vec[c];
  }
  out[row] = sum;
}
`;
