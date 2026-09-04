# webgpu-profiler

A benchmark suite for the operation that dominates large-language-model inference — **matrix &times;
vector multiplication** — comparing WebGPU compute-shader strategies head to head in the browser, built
on [`vgpu`](https://github.com/vercel-labs/vgpu).

Run it live: click **Run benchmark suite** on the website (`packages/website`) and watch results stream
into a table as each kernel finishes.

## What it measures

Most kernels multiply the same deterministically-generated matrix and vector (default 4096 &times; 4096,
padded to a multiple of 4), so results are directly comparable:

| Benchmark | What it tests |
| --- | --- |
| `read-bandwidth` | Streams the matrix buffer in via `vec4<f32>` loads folded with addition only (no multiply, one scalar written per thread) — a read-bandwidth-bound probe of peak storage-buffer read throughput |
| `write-bandwidth` | Streams computed `vec4<f32>` values out into a matrix-sized buffer with no buffer reads at all — a write-bandwidth-bound probe of peak storage-buffer write throughput |
| `f32-scalar` | Baseline: one thread per output row, scalar dot-product loop, f32 |
| `f32-vec4` | Same, but the K dimension is walked 4 lanes at a time via `vec4<f32>` |
| `f32-vec4-shared` | One **workgroup** per row; threads split K and tree-reduce partial sums in workgroup shared memory |
| `f16-scalar` / `f16-vec4` | Same shapes, storage buffers in `f16` (skipped if the device lacks `shader-f16`) |
| `i8-packed` | Symmetric per-tensor int8 quantization, `dot4I8Packed` 4-wide integer dot products (skipped without `packed_4x8_integer_dot_product`) |
| `f32-vec4-unrolled` | `f32-vec4`, but the inner K loop is manually unrolled 4x |
| Workgroup sweep | The tiled/shared-memory kernel run at every power-of-two workgroup size the device allows, to find the actual best size for *this* GPU rather than assuming one |
| MLP multi-layer | Chains 4 matvec layers with fused ReLU, all dispatches in a single command-buffer submit — only the final layer's output is ever read back |
| `flops-f32-scalar` | Raw fp32 FLOPS: a long chain of scalar fused multiply-adds held in a register, no vectors/matrices, ~no memory traffic |
| `flops-f32-vec4` | fp32 Vec SIMD FLOPS: the same FMA chain, but on a `vec4<f32>` register (4-wide SIMD) |
| `flops-f32-mat4` | fp32 Mat SIMD FLOPS: `x = m * x + c` chained with a `mat4x4<f32>` (a bounded contraction, so it stays numerically stable over any iteration count) |
| `flops-f16-scalar` / `flops-f16-vec4` / `flops-f16-mat4` | Same three raw-FLOPS shapes, but entirely in `f16` (skipped if the device lacks `shader-f16`) |
| `flops-i8-scalar` | Raw int8-range FLOPS: scalar `i32` multiply-add chain (WGSL has no first-class `i8` type) |
| `flops-i8-vec4` | Same integer FMA chain on a `vec4<i32>` register |
| `flops-i8-mat4` | A 4x4 integer matvec emulated as four `vec4<i32>` rows combined with `dot()` (WGSL has no `mat4x4<i32>`) |
| `flops-i8-dp4a` | `dot4I8Packed` from the `packed_4x8_integer_dot_product` extension, accumulated in a tight loop to isolate its peak throughput (skipped without the feature) |

The bandwidth and raw-FLOPS benchmarks are deliberately *not* matvec-shaped: they isolate one resource
(memory read, memory write, or ALU throughput) by keeping the other two as close to zero as the WebGPU
compute model allows, so they report the ceilings the matvec kernels above are being measured against.
The raw-FLOPS kernels' loop trip count and per-lane operands are runtime values (derived from a uniform
and the thread id), so the shader compiler can't constant-fold or hoist the FMA chain away. The integer
variants rely on i32/u32 wraparound (defined, trap-free two's-complement behavior in WGSL) to stay bounded
instead of the float variants' damped/contraction operands.

### Why not more matmul-style tiling?

Classic WebGPU matmul optimization (see [nuss-and-bolts' matmul kernel writeup](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel))
gets its biggest wins from **register/tile blocking**: each thread computes an NxN block of the output so a
loaded matrix tile is reused across multiple output elements, cutting global memory traffic. Matvec doesn't
have that reuse opportunity — every matrix element is read exactly once no matter how you tile it, since
there's only one output element (the row's dot product) per row, not a 2D output tile. So matvec is
memory-bandwidth-bound almost from the start (arithmetic intensity ~1 FLOP/byte), and the wins that *do*
transfer are the ones this suite tests: wide/coalesced loads (`vec4`), enough parallelism per row to
saturate memory bandwidth (the workgroup-tiled kernel + its size sweep), and loop unrolling to turn
loop-carried dependencies into independent instruction streams the compiler can pipeline. Reducing
precision (f16, int8) helps for the same memory-bound reason — it directly cuts the bytes that have to move.

Each benchmark: a calibration pass estimates how many dispatches need batching to take ~300ms, then 3
warmup measurements, then **10 timed measurements**, reporting mean/min/max/median/stddev. GPU timing
uses `timestamp-query` when available (pure device-side time, no CPU/driver overhead); otherwise it falls
back to wall-clock around `queue.onSubmittedWorkDone()`.

## Monorepo layout

- `packages/performance-suite` — the benchmark suite itself (WGSL shaders, data generation, timing
  harness, orchestration). Framework-agnostic, browser-only, consumed as TypeScript source.
- `packages/website` — TanStack Start + Tailwind CSS + shadcn/ui app that runs the suite client-side and
  renders results incrementally. Dev server on **port 3500**.

## Development

```bash
pnpm install
pnpm dev        # starts the website on http://localhost:3500
pnpm build      # typecheck performance-suite + build the website
pnpm lint
pnpm tsc
```

Requires a WebGPU-capable browser (recent Chrome/Edge desktop). `f16`/`int8`/timestamp-query benchmarks
report themselves as "skipped" with an explanation when the browser/GPU doesn't support the underlying
WebGPU feature, rather than failing the whole run.

## Deployment

`packages/website/Dockerfile` builds the whole workspace and serves the TanStack Start (Nitro) production
server on port 3500. `.github/workflows/deploy.yml` deploys it to Cloud Run on push to `main` (reuses the
generic `deploy-service.yml` workflow).
