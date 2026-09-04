/** SIMD-style variant: one thread per output row, but the K dimension is walked 4 lanes at a time via vec4<f32>. */
export const matvecF32Vec4Wgsl = /* wgsl */ `
struct Params {
  rows: u32,
  cols4: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> matrix: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vec: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }
  var sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  let base = row * params.cols4;
  for (var c: u32 = 0u; c < params.cols4; c = c + 1u) {
    sum = sum + matrix[base + c] * vec[c];
  }
  out[row] = sum.x + sum.y + sum.z + sum.w;
}
`;
