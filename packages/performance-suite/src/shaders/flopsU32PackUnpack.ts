/**
 * Manual byte pack/unpack ops probe: no `pack4x*` (or `unpack4x*`) builtin, just
 * the shift+mask bit-twiddling those builtins compile to, so this measures
 * the raw integer ALU/shifter cost of doing it by hand. Eight independent
 * u32 lanes, each per step: unpack 4 bytes (4 mask + 3 shift = 7 ops),
 * increment+wrap each byte (4 add + 4 mask = 8 ops), repack (4 mask + 3
 * shift + 3 or = 10 ops) = 25 ops/lane/step, no unrolling (each step is
 * already many instructions deep). 200 ops per loop iteration.
 */
export const flopsU32PackUnpackWgsl = /* wgsl */ `
struct Params {
  threads: u32,
  iterations: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

fn unpack4(v: u32) -> vec4<u32> {
  return vec4<u32>(v & 0xFFu, (v >> 8u) & 0xFFu, (v >> 16u) & 0xFFu, (v >> 24u) & 0xFFu);
}

fn pack4(b: vec4<u32>) -> u32 {
  return (b.x & 0xFFu) | ((b.y & 0xFFu) << 8u) | ((b.z & 0xFFu) << 16u) | ((b.w & 0xFFu) << 24u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.threads) {
    return;
  }
  var v0: u32 = idx * 2654435761u;
  var v1: u32 = (idx + 1u) * 2654435761u;
  var v2: u32 = (idx + 2u) * 2654435761u;
  var v3: u32 = (idx + 3u) * 2654435761u;
  var v4: u32 = (idx + 4u) * 2654435761u;
  var v5: u32 = (idx + 5u) * 2654435761u;
  var v6: u32 = (idx + 6u) * 2654435761u;
  var v7: u32 = (idx + 7u) * 2654435761u;
  let one = vec4<u32>(1u);
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    v0 = pack4((unpack4(v0) + one) & vec4<u32>(0xFFu));
    v1 = pack4((unpack4(v1) + one) & vec4<u32>(0xFFu));
    v2 = pack4((unpack4(v2) + one) & vec4<u32>(0xFFu));
    v3 = pack4((unpack4(v3) + one) & vec4<u32>(0xFFu));
    v4 = pack4((unpack4(v4) + one) & vec4<u32>(0xFFu));
    v5 = pack4((unpack4(v5) + one) & vec4<u32>(0xFFu));
    v6 = pack4((unpack4(v6) + one) & vec4<u32>(0xFFu));
    v7 = pack4((unpack4(v7) + one) & vec4<u32>(0xFFu));
  }
  out[idx] = f32(v0 ^ v1 ^ v2 ^ v3 ^ v4 ^ v5 ^ v6 ^ v7);
}
`;
