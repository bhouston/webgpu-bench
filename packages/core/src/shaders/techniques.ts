/** Fixed-work technique comparisons. Paired variants share inputs and outputs. */
const params = /* wgsl */ `
struct Params { count: u32, rounds: u32, padded: u32, shards: u32 };
@group(0) @binding(0) var<uniform> params: Params;
`;

export function workgroupWgsl(size: number): string {
  return /* wgsl */ `${params}
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  var x = input[i];
  for (var k = 0u; k < params.rounds; k++) {
    x = x * 0.75 + 0.125;
  }
  output[i] = x;
}
`;
}

export type ParticleLayout = 'aos' | 'soa' | 'aosoa';
export function particleWgsl(layout: ParticleLayout): string {
  const address =
    layout === 'aos'
      ? 'i * 8u + field'
      : layout === 'soa'
        ? 'field * params.padded + i'
        : '(i / 32u) * 256u + field * 32u + i % 32u';
  return /* wgsl */ `${params}
// Eight scalar fields: position.xyz, velocity.xyz, mass, lifetime.
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
fn address(i: u32, field: u32) -> u32 { return ${address}; }
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  // Only positions change; other fields remain immutable in the input.
  for (var field = 0u; field < 3u; field++) {
    output[address(i, field)] = input[address(i, field)] + input[address(i, field + 3u)] * 0.125;
  }
}
`;
}

export type TileStrategy = 'direct' | 'shared' | 'shared-padded';
export function tileWgsl(strategy: TileStrategy): string {
  const stride = strategy === 'shared-padded' ? 17 : 16;
  const shared = strategy !== 'direct';
  return /* wgsl */ `${params}
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
${shared ? `var<workgroup> tile: array<f32, ${16 * stride}>;` : ''}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let base = group.x * 256u;
  let row = lane / 16u;
  let col = lane % 16u;
  ${
    shared
      ? `tile[row * ${stride}u + col] = input[base + lane];
  workgroupBarrier();`
      : ''
  }
  // Padding lanes must participate in the cooperative load and barrier.
  if (base + lane >= params.count) { return; }
  var sum = 0.0;
  // A fixed 3x3 box stencil, with wraparound within each 16x16 tile.
  for (var dy = 0u; dy < 3u; dy++) {
    for (var dx = 0u; dx < 3u; dx++) {
      let y = (row + dy + 15u) % 16u;
      let x = (col + dx + 15u) % 16u;
      sum += ${shared ? `tile[y * ${stride}u + x]` : 'input[base + y * 16u + x]'};
    }
  }
  output[base + lane] = sum * (1.0 / 9.0);
}
`;
}

export type ReductionStrategy = 'serial' | 'workgroup' | 'subgroup';
export function reductionWgsl(strategy: ReductionStrategy): string {
  const bindings = /* wgsl */ `${params}
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
`;
  if (strategy === 'serial')
    return /* wgsl */ `${bindings}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let segment = gid.x;
  if (segment * 256u >= params.count) { return; }
  var sum = 0u;
  for (var k = 0u; k < 256u; k++) {
    let i = segment * 256u + k;
    if (i < params.count) { sum += input[i]; }
  }
  output[segment] = sum;
}
`;
  if (strategy === 'subgroup')
    return /* wgsl */ `enable subgroups;
${bindings}
var<workgroup> total: atomic<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32,
        @builtin(subgroup_invocation_id) subgroupLane: u32) {
  if (lane == 0u) { atomicStore(&total, 0u); }
  workgroupBarrier();
  let i = group.x * 256u + lane;
  var value = 0u;
  if (i < params.count) { value = input[i]; }
  let partial = subgroupAdd(value);
  // All lanes participate, including zero-padded tails. No mapping between
  // workgroup indices and subgroup lanes or a fixed subgroup width is assumed.
  if (subgroupLane == 0u) { atomicAdd(&total, partial); }
  workgroupBarrier();
  if (lane == 0u) { output[group.x] = atomicLoad(&total); }
}
`;
  return /* wgsl */ `${bindings}
var<workgroup> partial: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let i = group.x * 256u + lane;
  var value = 0u;
  if (i < params.count) { value = input[i]; }
  partial[lane] = value;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (lane < stride) { partial[lane] += partial[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u) { output[group.x] = partial[0]; }
}
`;
}

export function branchVec4Wgsl(useSelect: boolean): string {
  const body = useSelect
    ? 'x = select(x * 0.75 + vec4<f32>(0.25), x * 0.5 - vec4<f32>(0.5), mask);'
    : ['x', 'y', 'z', 'w']
        .map(
          (c) => `if (mask.${c}) { x.${c} = x.${c} * 0.5 - 0.5; }
    else { x.${c} = x.${c} * 0.75 + 0.25; }`,
        )
        .join('\n    ');
  return /* wgsl */ `${params}
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.count) { return; }
  var x = input[gid.x];
  var h = gid.x * 2654435761u;
  for (var k = 0u; k < params.rounds; k++) {
    h = h * 1664525u + 1013904223u;
    let mask = ((vec4<u32>(h) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(1u)) != vec4<u32>(0u);
    ${body}
  }
  output[gid.x] = x;
}
`;
}

export type AtomicStrategy = 'direct' | 'workgroup' | 'sharded';
/** Three entry points make a complete, repeatable histogram: reset, main, merge. */
export function atomicWgsl(strategy: AtomicStrategy): string {
  return /* wgsl */ `${params}
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
${strategy === 'workgroup' ? 'var<workgroup> bins: array<atomic<u32>, 32>;' : ''}
@compute @workgroup_size(256)
fn reset(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < params.shards * 32u) { atomicStore(&scratch[gid.x], 0u); }
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  ${
    strategy === 'workgroup'
      ? `if (lane < 32u) { atomicStore(&bins[lane], 0u); }
  workgroupBarrier();`
      : ''
  }
  // Sixteen grid-stride updates per invocation for a full workgroup.
  for (var k = 0u; k < 16u; k++) {
    let i = group.x * 4096u + lane + k * 256u;
    if (i < params.count) {
      let bin = input[i];
      ${
        strategy === 'workgroup'
          ? 'atomicAdd(&bins[bin], 1u);'
          : `atomicAdd(&scratch[${strategy === 'sharded' ? '(group.x % params.shards) * 32u + ' : ''}bin], 1u);`
      }
    }
  }
  ${
    strategy === 'workgroup'
      ? `workgroupBarrier();
  if (lane < 32u) { atomicAdd(&scratch[lane], atomicLoad(&bins[lane])); }`
      : ''
  }
}
@compute @workgroup_size(32)
fn merge(@builtin(local_invocation_index) bin: u32) {
  var sum = 0u;
  for (var shard = 0u; shard < params.shards; shard++) {
    sum += atomicLoad(&scratch[shard * 32u + bin]);
  }
  output[bin] = sum;
}
`;
}

export const dependentChainWgsl = /* wgsl */ `${params}
@group(0) @binding(1) var<storage, read> next: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
// Exactly one active invocation and one chain, not one chain per GPU lane.
@compute @workgroup_size(1)
fn main() {
  var index = 0u;
  for (var hop = 0u; hop < params.rounds; hop++) { index = next[index]; }
  output[0] = index;
}
`;
