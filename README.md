# webgpu-bench

[![CI](https://github.com/bhouston/webgpu-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/bhouston/webgpu-bench/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/webgpu-bench.svg)](https://www.npmjs.com/package/webgpu-bench)
[![npm downloads](https://img.shields.io/npm/dm/webgpu-bench.svg)](https://www.npmjs.com/package/webgpu-bench)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://web3dsurvey.com/benchmark)

A microbenchmark suite for WebGPU: it isolates and benchmarks a device's raw ceilings — memory
**bandwidth** (read/write) and **FLOPS** (fp32, fp16, int8; scalar/vec4/mat4/matvec) — one operation at a
time in the browser, not a full app or game.

Try it live on the [Web3D Survey: GPU Benchmark page](https://web3dsurvey.com/benchmark).

## Command line

Run the whole suite on a machine's GPU without a browser (WebGPU via [Dawn](https://github.com/dawn-gpu/node-webgpu)):

```sh
npx webgpu-bench-cli                          # or: npm i -g webgpu-bench-cli && webgpu-bench
webgpu-bench --filter 'f16-*'           # glob on benchmark ids
webgpu-bench --no-report                      # don't submit the run to web3dsurvey.com
webgpu-bench --json                           # machine-readable output
```

Progress is printed as a percentage on stderr; the results table goes to stdout once the run completes.
By default a finished run is submitted to [Web3D Survey](https://web3dsurvey.com/benchmark) and its result
page URL is printed.

## Usage

`runSuite` (in `suite.ts`) is a scheduler: it round-robins the given benchmarks, times them, watches for
thermal throttling, and yields result rows. `BENCHMARKS` (in `catalog.ts`) is this package's own set of
definitions and the default; pass your own `BenchmarkDefinition`s, a filtered subset, or a mix:

```ts
import { runSuite, BENCHMARKS, type BenchmarkDefinition } from 'webgpu-bench';

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

Each benchmark isolates one resource — memory read, memory write, or ALU — keeping the others near zero,
so numbers compare directly against a device's published bandwidth/FLOPS specs.

| Benchmark                                        | What it tests                                                                                                                     |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `read-linear` / `write-linear`                   | Coalesced grid-stride streaming of a 4096×4096 f32 buffer, isolating read or write throughput                                     |
| `read-gather-16kb` / `-4mb` / `-64mb`            | Same byte count, but each load hits a pseudo-random vec4 inside a 16 KB (L1), 4 MB (L2), or 64 MB (DRAM, the whole buffer) window |
| `write-scatter-16kb` / `-4mb` / `-64mb`          | Same byte count, but each store lands on a pseudo-random vec4 inside the same three windows                                       |
| `f32-fma-scalar` / `-vec4` / `-mat4` / `-matvec` | Unrolled independent FMA chains in fp32, at scalar/vec4/mat4/register-resident-matvec granularity                                 |
| `f16-fma-*`                                      | The same four shapes in `f16` (skipped without `shader-f16`)                                                                      |
| `i32-mad-*`                                      | The same four shapes as 32-bit integer multiply-add (WGSL has no `i8` arithmetic)                                                 |
| `i8-dp4a` / `i8-dp4a-matvec` / `u8-dp4a`         | Packed int8 dot products via `dot4I8Packed` / `dot4U8Packed` (skipped without the feature)                                        |
| `u32-shift`                                      | Variable-amount u32 shifts (left and right) cycling through all 32 amounts; the average shift throughput                          |

Loop trip counts are runtime values so the shader compiler can't fold them away, and are calibrated per
GPU (see below) so each dispatch stays short.

## Sampling methodology

Benchmarks run round-robin, not one-after-another, to avoid thermal throttling from skewing later
results. Per benchmark, the suite:

1. Prepares all kernels up front, so unsupported ones resolve as `skipped` immediately.
2. Calibrates dispatch size so one dispatch takes ~`targetDispatchMs` (10ms) — never freezing the tab.
3. Takes short (~100ms) timed measurements in random order, with an idle gap between them.
4. Reports the best (fastest) run — noise only ever slows a measurement down, never speeds it up.
5. Converges once the best stops improving (`minRounds`/`stableRounds`), up to a `maxRounds` cap.
6. Discards runs >20% slower than the current best as thermal noise (`throttledMs`, not `stats`).
7. Pauses and retries if the device is broadly throttled; gives up after `maxCooldowns` and flags the row.

All of the above are tunable via `SuiteOptions`. Timing prefers GPU `timestamp-query` and falls back to
wall-clock, cross-checking one against the other every measurement (Safari's timestamps are unreliable
enough to need this). `BenchmarkResult.stats` also has mean/median/stddev/ci95 if you want more than the
best-run headline number.

## Monorepo layout

- `packages/webgpu-bench` — the benchmark suite (WGSL shaders, data generation, timing harness,
  orchestration). Framework-agnostic, browser-only, consumed as TypeScript source.
- `packages/website` — TanStack Start + Tailwind + shadcn/ui app that runs the suite client-side.
  Dev server on **port 3500**.

## Development

```bash
pnpm install
pnpm dev        # starts the website on http://localhost:3500
pnpm build      # typecheck webgpu-bench + build the website
pnpm lint
pnpm tsc
```

Requires a WebGPU-capable browser (recent Chrome/Edge desktop). `f16`/`int8`-dot-product benchmarks report
as "skipped" rather than failing when a device lacks the feature.

## Author

Created by [Ben Houston](https://ben3d.ca) and sponsored by [Land of Assets](https://landofassets.com).
