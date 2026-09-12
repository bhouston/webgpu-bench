# webgpu-bench

[![npm](https://img.shields.io/npm/v/webgpu-bench.svg)](https://www.npmjs.com/package/webgpu-bench)
[![CI](https://github.com/bhouston/webgpu-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/bhouston/webgpu-bench/actions/workflows/ci.yml)
[![npm downloads](https://img.shields.io/npm/dm/webgpu-bench.svg)](https://www.npmjs.com/package/webgpu-bench)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://web3dsurvey.com/benchmark)

Run the [webgpu-bench-core](https://www.npmjs.com/package/webgpu-bench-core) GPU benchmark suite from the
command line, no browser needed (WebGPU via [Dawn](https://github.com/dawn-gpu/node-webgpu)).

This is a **micro-benchmark**, not a game or app benchmark: it tells you the raw cost of specific
operations (a memory read, an fp32 FMA, an int8 dot product, a divergent branch) on your GPU, so you can
make better-informed decisions when writing shaders — and understand how those costs differ across
devices.

```sh
npx webgpu-bench                        # or: npm i -g webgpu-bench && webgpu-bench
webgpu-bench --filter 'f16-*'           # glob on benchmark ids
webgpu-bench --no-report                # don't submit the run to web3dsurvey.com
webgpu-bench --json                     # machine-readable output
```

Progress is printed as a percentage on stderr; the results table goes to stdout once the run completes.
By default a finished run is submitted to [Web3D Survey](https://web3dsurvey.com/benchmark) and its result
page URL is printed.

## Example output

```
webgpu-bench v0.9.0
Vendor        apple
Architecture  metal-3
Description   Metal driver on macOS Version 27.0 (Build 26A428)
Features      core-features-and-limits, shader-f16, timestamp-query
Limits        maxBufferSize=4294967295, maxStorageBufferBindingSize=4294967295, maxComputeWorkgroupSizeX=1024, maxComputeInvocationsPerWorkgroup=1024, maxComputeWorkgroupStorageSize=32768
Supports      f16 yes, packed i8 dot yes, GPU timestamps yes
Host          Mac OS 27.0.0 arm64, Node.js 26.3.0, Dawn

Running 51 benchmarks…
[████████████████████████████████████████] 100%

branch-coherent      1.84 TFLOP/s
branch-divergent     1.02 TFLOP/s
branch-none          2.95 TFLOP/s
branch-uniform       1.99 TFLOP/s
f16-div              853.08 GFLOP/s
f16-fma-mat4         3.25 TFLOP/s
f16-fma-matvec       2.50 TFLOP/s
f16-fma-scalar       3.12 TFLOP/s
f16-fma-vec4         3.22 TFLOP/s
f16-log              853.08 GFLOP/s
f16-pow              427.23 GFLOP/s
f16-rsqrt            850.47 GFLOP/s
f16-sincos           226.51 GFLOP/s
f16-sqrt             844.63 GFLOP/s
f32-clamp            1.51 TOP/s
f32-div              851.96 GFLOP/s
f32-f16-convert      2.61 TOP/s
f32-fma-builtin      2.93 TFLOP/s
f32-fma-mat4         2.66 TFLOP/s
f32-fma-matvec       1.76 TFLOP/s
f32-fma-scalar       2.95 TFLOP/s
f32-fma-vec4         2.57 TFLOP/s
f32-log              853.08 GFLOP/s
f32-minmax           3.02 TOP/s
f32-pow              425.23 GFLOP/s
f32-rsqrt            853.08 GFLOP/s
f32-select           973.07 GOP/s
f32-sincos           175.90 GFLOP/s
f32-sqrt             844.73 GFLOP/s
i32-div              153.72 GOP/s
i32-f32-convert      427.10 GOP/s
i32-mad-mat4         837.51 GOP/s
i32-mad-matvec       830.51 GOP/s
i32-mad-scalar       849.54 GOP/s
i32-mad-vec4         851.12 GOP/s
i8-dp4a              612.16 GOP/s
i8-dp4a-matvec       556.90 GOP/s
read-gather-16kb     774.07 GB/s
read-gather-4mb      65.41 GB/s
read-gather-64mb     23.68 GB/s
read-linear          92.38 GB/s
u32-countonebits     425.95 GOP/s
u32-firstleadingbit  358.07 GOP/s
u32-packunpack       607.37 GOP/s
u32-reversebits      426.60 GOP/s
u32-shift            424.42 GOP/s
u8-dp4a              781.47 GOP/s
write-linear         90.50 GB/s
write-scatter-16kb   795.87 GB/s
write-scatter-4mb    58.11 GB/s
write-scatter-64mb   20.10 GB/s

Reported to https://web3dsurvey.com/benchmark/33e2e606-baec-43d2-abab-422567824435
```

## Author

Created by [Ben Houston](https://ben3d.ca) and sponsored by [Land of Assets](https://landofassets.com).
