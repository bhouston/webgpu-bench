# webgpu-profiler

A focused benchmark suite for the raw ceilings of a WebGPU device: memory **bandwidth** (read and write)
and **FLOPS** (fp32, fp16, and int8, at scalar / vec4 / mat4 / register-resident-matvec granularity), run head to head in the
browser, built on [`vgpu`](https://github.com/vercel-labs/vgpu).

Run it live: click **Run benchmark suite** on the website (`packages/website`) and watch results stream
into a table as each kernel finishes.

## What it measures

Every benchmark deliberately isolates a single resource — memory read, memory write, or ALU throughput —
by keeping the other two as close to zero as the WebGPU compute model allows, rather than modeling a
specific real-world op. That makes each number directly comparable to the device's published
bandwidth/FLOPS specs.

| Benchmark | What it tests |
| --- | --- |
| `read-bandwidth` | Streams a large buffer in via `vec4<f32>` loads folded with addition only (no multiply, one scalar written per thread) — a read-bandwidth-bound probe of peak storage-buffer read throughput |
| `write-bandwidth` | Streams computed `vec4<f32>` values out into a large buffer with no buffer reads at all — a write-bandwidth-bound probe of peak storage-buffer write throughput |
| `flops-f32-scalar` | Raw fp32 FLOPS: eight *independent* scalar FMA chains per thread, unrolled 4x, so the ALU always has work in flight (a single dependent chain would measure FMA latency + loop overhead, not throughput) |
| `flops-f32-vec4` | One FMA chain held in a `vec4<f32>` register. On scalar-SIMT GPUs (Apple, NVIDIA, AMD) that is 4 independent scalar FMAs per step — it measures 4-wide instruction-level parallelism, not a wider ALU |
| `flops-f32-mat4` | `x = m * x + c` chained with a `mat4x4<f32>` (a bounded contraction, so it stays numerically stable over any iteration count) |
| `flops-f32-matvec` | Register-resident matvec tile: a 4x8 f32 weight tile held in registers, applied to an 8-wide input every iteration via `dot()`, outputs feeding back as the next inputs. The dot-product-accumulate shape of a GEMV inner loop with zero buffer traffic |
| `flops-f16-scalar` / `-vec4` / `-mat4` / `-matvec` | The same four shapes, entirely in `f16` (skipped if the device lacks `shader-f16`) |
| `flops-i8-scalar` / `-vec4` / `-mat4` | The scalar / vec4 shapes on `i32` (WGSL has no first-class `i8` type), and a 4x4 integer matvec emulated as four `vec4<i32>` rows combined with `dot()` (WGSL has no `mat4x4<i32>`) |
| `flops-i8-matvec` | The register-resident matvec tile with int8-range weights/inputs held unpacked as `vec4<i32>` and integer `dot()` accumulation — the no-extension int8 path |
| `flops-i8-matvec-dp4a` | The same tile with weights/inputs kept packed four int8 lanes per `u32`, each 4-wide dot product a single `dot4I8Packed` (skipped without `packed_4x8_integer_dot_product`) |
| `flops-i8-dp4a` | `dot4I8Packed` accumulated in a tight loop to isolate the instruction's peak throughput (skipped without the feature) |

The two bandwidth benchmarks stream the same deterministically-generated buffer (default 4096 &times;
4096 f32, padded to a multiple of 4). The raw-FLOPS kernels' loop trip count and per-lane operands are
runtime values (derived from a uniform and the thread id), so the shader compiler can't constant-fold or
hoist the FMA chain away. Each kernel picks its own trip count (256 for the unrolled scalar chains, 512
for the matvec tiles, 1024 otherwise) so every dispatch takes several ms and per-thread setup, the final
store, and dispatch overhead are negligible; `computeIterations` overrides all of them. The integer variants rely on i32/u32 wraparound (defined, trap-free
two's-complement behavior in WGSL) to stay bounded instead of the float variants' damped/contraction
operands.

Each benchmark: a calibration pass estimates how many dispatches need batching to take ~300ms (and
doubles as an untimed warm-up), then 1 more discarded warmup measurement, then **adaptive sampling**:
timed measurements are taken one at a time and, after each one (from the 3rd on), the harness checks
whether the 95% Student-t confidence interval on the mean is within ±3% of the mean. As soon as it is,
sampling stops; if the timings are too noisy to converge it stops at a hard cap of 10 runs and the row
is flagged `max-runs` (shown as "noisy" in the table). On a quiet GPU most kernels settle in 3–4 runs,
which is what makes the suite several times faster than a fixed 10-run loop. `minRuns`, `maxRuns`,
`precision`, and `warmups` are all `SuiteOptions`.

The reported number is the median (used for the throughput numbers, since one slow run from a
still-ramping GPU clock skews a mean but not a median) plus stddev (mean/min/max/ci95 are still in the
raw `BenchmarkResult.stats` if you need them — just not shown in the table). GPU timing uses
`timestamp-query` when available (pure device-side time, no CPU/driver overhead); otherwise it falls back
to wall-clock around `queue.onSubmittedWorkDone()`.

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

Requires a WebGPU-capable browser (recent Chrome/Edge desktop). `f16`/`int8`-dot-product benchmarks report
themselves as "skipped" with an explanation when the browser/GPU doesn't support the underlying WebGPU
feature, rather than failing the whole run.

## Deployment

`packages/website/Dockerfile` builds the whole workspace and serves the TanStack Start (Nitro) production
server on port 3500. `.github/workflows/deploy.yml` deploys it to Cloud Run on push to `main` (reuses the
generic `deploy-service.yml` workflow).
