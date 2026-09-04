/** f16 storage buffers, scalar dot-product loop, f32 accumulation (matches typical mixed-precision LLM inference). */
export const matvecF16ScalarWgsl = /* wgsl */ `
enable f16;

struct Params {
  rows: u32,
  cols: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> matrix: array<f16>;
@group(0) @binding(2) var<storage, read> vec: array<f16>;
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
    sum = sum + f32(matrix[base + c]) * f32(vec[c]);
  }
  out[row] = sum;
}
`;
