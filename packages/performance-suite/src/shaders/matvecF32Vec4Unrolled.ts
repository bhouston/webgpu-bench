/**
 * Same one-thread-per-row / vec4 SIMD approach as matvecF32Vec4, but with
 * the inner K loop manually unrolled 4x (16 floats per iteration). Matmul
 * kernels get their biggest wins from *register blocking* — reusing loaded
 * tiles across several output elements — but matvec has only one output
 * per row, so there's no data reuse to exploit that way (see README).
 * What still transfers from matmul-optimization write-ups is unrolling
 * itself: it turns loop-carried control flow into straight-line
 * fmul/fadd chains the compiler can pipeline, and issues 4 independent
 * global loads per iteration instead of 1.
 */
export const matvecF32Vec4UnrolledWgsl = /* wgsl */ `
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
  let base = row * params.cols4;
  let unrolledEnd = params.cols4 - (params.cols4 % 4u);

  var sum0: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var sum1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var sum2: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var sum3: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

  var c: u32 = 0u;
  loop {
    if (c >= unrolledEnd) {
      break;
    }
    sum0 = sum0 + matrix[base + c] * vec[c];
    sum1 = sum1 + matrix[base + c + 1u] * vec[c + 1u];
    sum2 = sum2 + matrix[base + c + 2u] * vec[c + 2u];
    sum3 = sum3 + matrix[base + c + 3u] * vec[c + 3u];
    c = c + 4u;
  }

  var sum = (sum0 + sum1) + (sum2 + sum3);
  loop {
    if (c >= params.cols4) {
      break;
    }
    sum = sum + matrix[base + c] * vec[c];
    c = c + 1u;
  }

  out[row] = sum.x + sum.y + sum.z + sum.w;
}
`;
