/**
 * int8 quantized matvec using the `packed_4x8_integer_dot_product` WGSL
 * extension: matrix and vector are pre-packed on the CPU as arrays of u32,
 * each holding four signed int8 lanes, and `dot4I8Packed` does a 4-wide
 * dot product per instruction. A single per-tensor scale for the matrix and
 * one for the vector dequantize the integer accumulator back to f32
 * (symmetric quantization, matching typical LLM weight-only / W8A8 setups).
 */
export const matvecI8PackedWgsl = /* wgsl */ `
enable packed_4x8_integer_dot_product;

struct Params {
  rows: u32,
  cols4: u32,
  matScale: f32,
  vecScale: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> matrix: array<u32>;
@group(0) @binding(2) var<storage, read> vec: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }
  var acc: i32 = 0;
  let base = row * params.cols4;
  for (var c: u32 = 0u; c < params.cols4; c = c + 1u) {
    acc = acc + dot4I8Packed(matrix[base + c], vec[c]);
  }
  out[row] = f32(acc) * params.matScale * params.vecScale;
}
`;
