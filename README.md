# webgpu-bench (monorepo)

[![CI](https://github.com/bhouston/webgpu-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/bhouston/webgpu-bench/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://web3dsurvey.com/benchmark)

A microbenchmark suite for WebGPU: it isolates and benchmarks a device's raw ceilings — memory
**bandwidth** (read/write) and **FLOPS** (fp32, fp16, int8; scalar/vec4/mat4/matvec) — one operation at a
time in the browser, not a full app or game.

Try it live on the [Web3D Survey: GPU Benchmark page](https://web3dsurvey.com/benchmark).

## Packages

- [`packages/webgpu-bench`](packages/webgpu-bench) — the benchmark suite (WGSL shaders, data generation,
  timing harness, orchestration). Framework-agnostic, browser-only, consumed as TypeScript source. See its
  [README](packages/webgpu-bench/README.md).
- [`packages/cli`](packages/cli) — `webgpu-bench` command-line runner (WebGPU via Dawn, no browser
  needed). See its [README](packages/cli/README.md).

## Development

```bash
pnpm install
pnpm build
pnpm lint
pnpm tsc
pnpm test
```

Requires a WebGPU-capable browser (recent Chrome/Edge desktop) for the browser tests. `f16`/`int8`-dot-
product benchmarks report as "skipped" rather than failing when a device lacks the feature.

## Author

Created by [Ben Houston](https://ben3d.ca) and sponsored by [Land of Assets](https://landofassets.com).
