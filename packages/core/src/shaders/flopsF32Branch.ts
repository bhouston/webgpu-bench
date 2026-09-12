/**
 * Branch-divergence probes. The compiled cost of an `if` depends on
 * lane agreement as well as compiler lowering. Same
 * eight-chain, 4x-unrolled fp32 FMA body as {@link flopsF32ScalarWgsl}, but
 * each unrolled step is wrapped in an `if` whose two sides do identical
 * work with different constants (`a,b` vs `c,d`), so FLOPs per iteration
 * are fixed at 64 whichever side runs. `none` is the same body with no `if`
 * at all (identical to f32-fma-scalar), the baseline the others read against. Bodies are 8 multiply-adds each, but the compiler may still predicate
 * or select coefficients rather than emit a branch. Only `cond` differs:
 *
 * - none:      no branch.
 * - uniform:   `(i & 1u) == 0u`            every lane agrees; one side runs.
 * - coherent:  `((idx / 64u + i) & 1u)`    workgroups disagree, lanes within
 *              one agree; tests runtime detection of dynamic uniformity.
 * - divergent: `((idx + i) & 1u)`          adjacent lanes disagree; the penalty
 *              depends on how the compiler lowers the condition.
 */
const step = (a: string, b: string) =>
  ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'].map((x) => `      ${x} = ${x} * ${a} + ${b};`).join('\n');

function branchWgsl(cond: string | null): string {
  const branch = (k: number) =>
    cond === null
      ? step('a', 'b')
      : `    if (${cond.replaceAll('STEP', `(i + ${k}u)`)}) {
${step('a', 'b')}
    } else {
${step('c', 'd')}
    }`;
  return /* wgsl */ `
struct Params {
  threads: u32,
  iterations: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.threads) {
    return;
  }
  let a: f32 = 1.0 - f32(idx & 15u) * 0.001;
  let b: f32 = 0.5 + f32(idx & 7u) * 0.001;
  let c: f32 = 1.0 - f32(idx & 31u) * 0.0005;
  let d: f32 = 0.25 + f32(idx & 3u) * 0.001;
  var x0: f32 = f32(idx & 255u) * 0.0001 + 0.01;
  var x1: f32 = f32(idx & 255u) * 0.0001 + 0.02;
  var x2: f32 = f32(idx & 255u) * 0.0001 + 0.03;
  var x3: f32 = f32(idx & 255u) * 0.0001 + 0.04;
  var x4: f32 = f32(idx & 255u) * 0.0001 + 0.05;
  var x5: f32 = f32(idx & 255u) * 0.0001 + 0.06;
  var x6: f32 = f32(idx & 255u) * 0.0001 + 0.07;
  var x7: f32 = f32(idx & 255u) * 0.0001 + 0.08;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
${[0, 1, 2, 3].map(branch).join('\n')}
  }
  out[idx] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
`;
}

export const branchNoneWgsl = branchWgsl(null);
export const branchUniformWgsl = branchWgsl('(STEP & 1u) == 0u');
export const branchCoherentWgsl = branchWgsl('((idx / 64u + STEP) & 1u) == 0u');
export const branchDivergentWgsl = branchWgsl('((idx + STEP) & 1u) == 0u');
