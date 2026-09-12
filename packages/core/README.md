# webgpu-bench-core

[![npm](https://img.shields.io/npm/v/webgpu-bench-core.svg)](https://www.npmjs.com/package/webgpu-bench-core)
[![CI](https://github.com/bhouston/webgpu-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/bhouston/webgpu-bench/actions/workflows/ci.yml)
[![npm downloads](https://img.shields.io/npm/dm/webgpu-bench-core.svg)](https://www.npmjs.com/package/webgpu-bench-core)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://web3dsurvey.com/benchmark)

A microbenchmark suite for WebGPU: it isolates and benchmarks a device's raw ceilings — memory
**bandwidth** (read/write) and **FLOPS** (fp32, fp16, int8; scalar/vec4/mat4/matvec) — one operation at a
time in the browser, plus fixed-work shader technique comparisons. It is not a full app or game benchmark.

Try it live on the [Web3D Survey: GPU Benchmark page](https://web3dsurvey.com/benchmark).

This is a **micro-benchmark**, not a game or app benchmark: it tells you the raw cost of specific
operations (a memory read, an fp32 FMA, an int8 dot product, a divergent branch) on the user's GPU, so
you can make better-informed decisions when writing shaders — and understand how those costs differ
across devices.

Want to run the suite from the command line instead of a browser? See
[webgpu-bench](https://www.npmjs.com/package/webgpu-bench).

## Usage

`runSuite` (in `suite.ts`) is a scheduler: it round-robins the given benchmarks, times them, watches for
thermal throttling, and yields result rows. `BENCHMARKS` (in `catalog.ts`) is this package's own set of
definitions and the default; pass your own `BenchmarkDefinition`s, a filtered subset, or a mix:

```ts
import { runSuite, BENCHMARKS, type BenchmarkDefinition } from 'webgpu-bench-core';

const myKernel: BenchmarkDefinition = {
  id: 'my-kernel',
  label: 'My kernel',
  description: '...',
  source: myWgsl,
  category: 'compute',
  metric: { key: 'flops', unit: 'FLOP', name: 'Floating-point ops' },
  prepare: async ({ ctx, harness }) => {
    /* build a pipeline/bind group, return via prepareKernelBenchmark(...) */
  },
};

for await (const result of runSuite({
  benchmarks: [...BENCHMARKS.filter((b) => b.category === 'bandwidth'), myKernel],
})) {
  // ...
}
```

`index.ts` also exports `GpuContext`/`acquireGpuContext`, `createPipeline`, `prepareKernelBenchmark`, the
shared metric defs, and `generateMatVecData` for writing your own `prepare()`.

## What's measured

The raw throughput probes emphasize memory reads, writes, or ALU work. The technique comparisons below
measure complete useful tasks, including the memory and synchronization needed by each alternative.

| Benchmark                                               | What it tests                                                                                                                                                                                 |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `read-linear` / `write-linear`                          | Coalesced grid-stride streaming of a 4096×4096 f32 buffer, isolating read or write throughput                                                                                                 |
| `read-gather-16kb` / `-4mb` / `-64mb`                   | Same byte count, but each load hits a pseudo-random vec4 inside a 16 KB, 4 MB, or 64 MB (the whole default buffer) window                                                                     |
| `write-scatter-16kb` / `-4mb` / `-64mb`                 | Each vec4 in the clamped window is written once through a unique-destination permutation; reported bytes scale with the window                                                                |
| `f32-fma-scalar` / `-vec4` / `-mat4` / `-matvec`        | Unrolled independent FMA chains in fp32, at scalar/vec4/mat4/register-resident-matvec granularity                                                                                             |
| `f16-fma-*`                                             | The same four shapes in `f16` (skipped without `shader-f16`)                                                                                                                                  |
| `i32-mad-*`                                             | The same four shapes as 32-bit integer multiply-add (WGSL has no `i8` arithmetic)                                                                                                             |
| `i8-dp4a` / `i8-dp4a-matvec` / `u8-dp4a`                | Packed int8 dot products via `dot4I8Packed` / `dot4U8Packed` (skipped without the feature)                                                                                                    |
| `u32-shift`                                             | Variable-amount u32 shifts (left and right) cycling through all 32 amounts; the average shift throughput                                                                                      |
| `branch-none` / `-uniform` / `-coherent` / `-divergent` | The fp32 scalar FMA chains with no branch, or wrapped in an if/else with equal work per side whose condition agrees across all lanes, per workgroup, or flips per lane (SIMT divergence cost) |

The raw ALU probes use runtime loop counts and calibrate them per GPU (see below) to keep dispatches
short. Runtime inputs discourage constant folding, but do not guarantee a particular compiler lowering.

## Shader technique comparisons

These 18 `algorithm` benchmarks use one fixed workload per family. Only the named technique changes;
there are no parameter sweeps. They report useful work per second, so compare rates within a family.
The existing `computeThreads`, `computeIterations`, and matrix-size options do not resize these fixed
comparisons. Sampling calibrates batch repetitions, not their workload or loop counts.

| IDs                                                             | Fixed comparison                                                                                                                                      | Metric              |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `workgroup-64`, `workgroup-128`, `workgroup-256`                | 65,536 scalar elements, one element per invocation, 16 identical arithmetic steps; only workgroup size changes                                        | elements/s          |
| `layout-aos`, `layout-soa`, `layout-aosoa`                      | Position updates for 65,536 particles with eight scalar fields; AoSoA uses 32-particle blocks; reads positions and velocities, preserves other fields | particles/s         |
| `tile-direct`, `tile-shared`, `tile-shared-padded`              | Same 3×3 box stencil on 256 independent 16×16 tiles, wrapping within each tile; direct storage reads versus shared-memory row strides 16 and 17       | outputs/s           |
| `reduction-serial`, `reduction-workgroup`, `reduction-subgroup` | Same 256 independent sums of 256 bounded u32 values; serial per segment, shared-memory tree, or subgroup sums combined with workgroup atomics         | input elements/s    |
| `branch-vec4-select`, `branch-vec4-if`                          | Same 65,536 vec4 inputs, 64 steps, seeded component masks, and bounded arithmetic; vector select versus four scalar if/else statements                | component choices/s |
| `atomic-direct`, `atomic-workgroup`, `atomic-sharded`           | Same 65,536 keys and 32-bin histogram, about half targeting bin zero; direct global atomics, workgroup aggregation, or eight global shards            | input updates/s     |
| `read-dependent-chain`                                          | Exactly one invocation follows 4,096 dependent links through a seeded single-cycle permutation of a 4 MiB buffer                                      | hops/s              |

Within each family, variants consume the same logical inputs, produce the same logical outputs, and
count the same useful work. Particle buffers are decoded to a common field order when checking
equivalence. Integer sums/histograms must agree exactly; floating-point results are checked with a
small tolerance. Vitest checks both CPU references and direct agreement between variants.

A repeatable higher rate therefore identifies the faster implementation **for this workload on this
GPU and runtime**. It is useful evidence for choosing a technique in a similar shader. It does not
establish the best technique for every shader on that GPU: workgroup results depend on the shader's
resource needs, tile results on reuse, histogram results on contention, and branch results on the
conditions and work in each arm. Layout conversion/upload costs are excluded, so these tests assume
data is already in the chosen layout. Reductions compare complete segment sums, not full-array sums.
Compare results using the same timing method, and treat small or inconsistent differences as
inconclusive. The dependent chain is a standalone cost probe, with no alternative variant to rank.

The histogram measurements include reset and final merge on every repetition, so repeated batches
compute the same complete result. A repetition is three dispatches for histograms and one for the other
families; `innerIterations` counts repetitions. Tile loading and barriers are included. CPU data
packing/upload and correctness readback are outside timing. The layout tests update positions in the
original layout from immutable inputs; they do not include conversion to a common output layout.

`reduction-subgroup` requests the optional `subgroups` feature and reports `skipped` if unsupported.
It makes no assumption about subgroup width or the mapping to workgroup indices. The three reductions
produce final **segment** sums, not a single global sum. See the
[WGSL subgroup operations](https://www.w3.org/TR/WGSL/#subgroup-builtin-functions).

These are specific operating points, not universal technique rankings. The branch test does not
force the compiler to emit native branches. The histogram has one fixed contention distribution.
For the dependent chain, `1e9 / hopsPerSecond` gives amortized ns/hop, including loop and dispatch cost;
repeated measurements reuse the same footprint and starting point. It is not a physical memory-cycle
measurement. Validate shader outputs and tails with `techniques.browser.test.ts` before interpreting
performance changes.

## Sampling methodology

Benchmarks run round-robin, not one-after-another, to avoid thermal throttling from skewing later
results. Per benchmark, the suite:

1. Prepares all kernels up front, so unsupported ones resolve as `skipped` immediately.
2. Calibrates per-dispatch work for kernels with a work knob toward `targetDispatchMs` (10ms); fixed-work
   comparisons keep their logical problem unchanged and calibrate only the number of batched repetitions.
3. Takes short (~100ms) timed measurements in random order, with an idle gap between them.
4. Reports the best (fastest) run — noise only ever slows a measurement down, never speeds it up.
5. Converges once the best stops improving (`minRounds`/`stableRounds`), up to a `maxRounds` cap.
6. Discards runs >20% slower than the current best as thermal noise (`throttledMs`, not `stats`).
7. Pauses and retries if the device is broadly throttled; gives up after `maxCooldowns` and flags the row.

All of the above are tunable via `SuiteOptions`. Timing prefers GPU `timestamp-query` and falls back to
wall-clock, cross-checking one against the other every measurement (Safari's timestamps are unreliable
enough to need this). `BenchmarkResult.stats` also has mean/median/stddev/ci95 if you want more than the
best-run headline number.

Requires a WebGPU-capable browser (recent Chrome/Edge desktop). `f16`/`int8`-dot-product benchmarks report
as "skipped" rather than failing when a device lacks the feature.

## Author

Created by [Ben Houston](https://ben3d.ca) and sponsored by [Land of Assets](https://landofassets.com).
