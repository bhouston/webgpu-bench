/**
 * Workgroup-tiled variant: one *workgroup* per output row (not one thread).
 * WG_SIZE threads split the K dimension between them (vec4-at-a-time),
 * accumulate partial sums into workgroup shared memory, and tree-reduce to
 * a single value. WG_SIZE is a pipeline-overridable constant so the same
 * shader can be tuned/swept across workgroup sizes without recompiling WGSL
 * source. An optional fused ReLU lets this same kernel be chained directly
 * into the multi-layer MLP benchmark without a separate activation dispatch.
 */
export const matvecF32Vec4SharedWgsl = /* wgsl */ `
struct Params {
  rows: u32,
  cols4: u32,
  applyRelu: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> matrix: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vec: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

override WG_SIZE: u32 = 128u;
const MAX_WG_SIZE: u32 = 256u;
var<workgroup> partial: array<f32, MAX_WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wgid.x;
  if (row >= params.rows) {
    return;
  }
  let base = row * params.cols4;
  var sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var c: u32 = lid.x;
  loop {
    if (c >= params.cols4) {
      break;
    }
    sum = sum + matrix[base + c] * vec[c];
    c = c + WG_SIZE;
  }
  partial[lid.x] = sum.x + sum.y + sum.z + sum.w;
  workgroupBarrier();

  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (lid.x < stride) {
      partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid.x == 0u) {
    var result = partial[0];
    if (params.applyRelu != 0u) {
      result = max(result, 0.0);
    }
    out[row] = result;
  }
}
`;
