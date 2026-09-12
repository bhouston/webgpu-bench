/**
 * Manual byte pack/unpack probe: no `pack4x*` (or `unpack4x*`) builtin, just
 * the shift+mask bit-twiddling those builtins compile to. Eight independent
 * u32 lanes, unrolled 4x; each step unpacks 4 bytes, increments each, and
 * repacks. Written with no redundant ops (the top byte needs no mask after
 * `>> 24u`, and `<< 24u` discards the bits a mask would) so the source is as
 * close as possible to what executes — but the compiler is still free to
 * reassociate shifts and masks, so each lane step counts as 2 ops (one
 * unpack + one pack of a whole u32), not an instruction count. 64 ops per
 * loop iteration.
 */
export const flopsU32PackUnpackWgsl = /* wgsl */ `
struct Params {
  threads: u32,
  iterations: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

// Unpack 4 bytes, add 1 to each (wrapping), repack.
fn step(v: u32) -> u32 {
  let b0 = (v & 0xFFu) + 1u;
  let b1 = ((v >> 8u) & 0xFFu) + 1u;
  let b2 = ((v >> 16u) & 0xFFu) + 1u;
  let b3 = (v >> 24u) + 1u;
  return (b0 & 0xFFu) | ((b1 & 0xFFu) << 8u) | ((b2 & 0xFFu) << 16u) | (b3 << 24u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.threads) {
    return;
  }
  var v0: u32 = (idx + 0u) * 2654435761u;
  var v1: u32 = (idx + 1u) * 2654435761u;
  var v2: u32 = (idx + 2u) * 2654435761u;
  var v3: u32 = (idx + 3u) * 2654435761u;
  var v4: u32 = (idx + 4u) * 2654435761u;
  var v5: u32 = (idx + 5u) * 2654435761u;
  var v6: u32 = (idx + 6u) * 2654435761u;
  var v7: u32 = (idx + 7u) * 2654435761u;
  for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
    v0 = step(v0);
    v1 = step(v1);
    v2 = step(v2);
    v3 = step(v3);
    v4 = step(v4);
    v5 = step(v5);
    v6 = step(v6);
    v7 = step(v7);
    v0 = step(v0);
    v1 = step(v1);
    v2 = step(v2);
    v3 = step(v3);
    v4 = step(v4);
    v5 = step(v5);
    v6 = step(v6);
    v7 = step(v7);
    v0 = step(v0);
    v1 = step(v1);
    v2 = step(v2);
    v3 = step(v3);
    v4 = step(v4);
    v5 = step(v5);
    v6 = step(v6);
    v7 = step(v7);
    v0 = step(v0);
    v1 = step(v1);
    v2 = step(v2);
    v3 = step(v3);
    v4 = step(v4);
    v5 = step(v5);
    v6 = step(v6);
    v7 = step(v7);
  }
  out[idx] = f32(v0 ^ v1 ^ v2 ^ v3 ^ v4 ^ v5 ^ v6 ^ v7);
}
`;
