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

| Benchmark                                          | What it tests                                                                                                                                                                                                                                           |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `read-bandwidth`                                   | Streams a large buffer in via `vec4<f32>` loads folded with addition only (no multiply, one scalar written per thread) — a read-bandwidth-bound probe of peak storage-buffer read throughput                                                            |
| `write-bandwidth`                                  | Streams computed `vec4<f32>` values out into a large buffer with no buffer reads at all — a write-bandwidth-bound probe of peak storage-buffer write throughput                                                                                         |
| `flops-f32-scalar`                                 | Raw fp32 FLOPS: eight _independent_ scalar FMA chains per thread, unrolled 4x, so the ALU always has work in flight (a single dependent chain would measure FMA latency + loop overhead, not throughput)                                                |
| `flops-f32-vec4`                                   | One FMA chain held in a `vec4<f32>` register. On scalar-SIMT GPUs (Apple, NVIDIA, AMD) that is 4 independent scalar FMAs per step — it measures 4-wide instruction-level parallelism, not a wider ALU                                                   |
| `flops-f32-mat4`                                   | `x = m * x + c` chained with a `mat4x4<f32>` (a bounded contraction, so it stays numerically stable over any iteration count)                                                                                                                           |
| `flops-f32-matvec`                                 | Register-resident matvec tile: a 4x8 f32 weight tile held in registers, applied to an 8-wide input every iteration via `dot()`, outputs feeding back as the next inputs. The dot-product-accumulate shape of a GEMV inner loop with zero buffer traffic |
| `flops-f16-scalar` / `-vec4` / `-mat4` / `-matvec` | The same four shapes, entirely in `f16` (skipped if the device lacks `shader-f16`)                                                                                                                                                                      |
| `flops-i8-scalar` / `-vec4` / `-mat4`              | The scalar / vec4 shapes on `i32` (WGSL has no first-class `i8` type), and a 4x4 integer matvec emulated as four `vec4<i32>` rows combined with `dot()` (WGSL has no `mat4x4<i32>`)                                                                     |
| `flops-i8-matvec`                                  | The register-resident matvec tile with int8-range weights/inputs held unpacked as `vec4<i32>` and integer `dot()` accumulation — the no-extension int8 path                                                                                             |
| `flops-i8-matvec-dp4a`                             | The same tile with weights/inputs kept packed four int8 lanes per `u32`, each 4-wide dot product a single `dot4I8Packed` (skipped without `packed_4x8_integer_dot_product`)                                                                             |
| `flops-i8-dp4a`                                    | `dot4I8Packed` accumulated in a tight loop to isolate the instruction's peak throughput (skipped without the feature)                                                                                                                                   |

The two bandwidth benchmarks stream the same deterministically-generated buffer (default 4096 &times;
4096 f32, padded to a multiple of 4). The raw-FLOPS kernels' loop trip count and per-lane operands are
runtime values (derived from a uniform and the thread id), so the shader compiler can't constant-fold or
hoist the FMA chain away. Each kernel has its own ceiling on the trip count (256 for the unrolled scalar
chains, 512 for the matvec tiles, 1024 otherwise); the actual count is calibrated per GPU so a single
dispatch takes ~10ms (see below), which keeps per-thread setup, the final store, and dispatch overhead
negligible without ever freezing the display. `computeIterations` pins it. The integer variants rely on i32/u32 wraparound (defined, trap-free
two's-complement behavior in WGSL) to stay bounded instead of the float variants' damped/contraction
operands.

## How it samples (and why the number is the _best_ run)

Benchmarks aren't run one after another to completion. That's the naive approach, and on a phone it
fails in a specific way: a few seconds of saturated GPU is enough to hit thermal throttling, so every
benchmark after the first couple measures a slowed-down device, and the numbers depend on the run
order. Instead the suite:

1. **Prepares every kernel up front** (shader compile, buffers, bind groups) so rows that can't run —
   missing `shader-f16` / `packed_4x8_integer_dot_product`, a failed shader compile — resolve as
   `skipped` / `error` immediately and the table has its final shape before any timing starts.
2. **Sizes each dispatch for this GPU.** A GPU dispatch can't be preempted, so a long one freezes
   the display for its whole duration. The raw-FLOPS kernels' loop trip count is therefore not fixed:
   on its first visit each kernel is probed with a tiny trip count (16) and ramped up — never more than
   4x per step — until one dispatch takes about `targetDispatchMs` (10ms), capped at the kernel's own
   default (256–1024). A desktop lands near the cap; a phone lands on a much smaller count; neither
   ever runs a dispatch longer than a few tens of ms, even during calibration. `computeIterations`
   pins the count and disables this.
3. **Samples round-robin, in a fresh random order each round.** Each round takes one short timed measurement (~100ms, `targetMs`: a batch
   of those ~10ms dispatches in one command buffer) of every still-active kernel, with an idle gap
   (`idleMs`, 100ms) between measurements so the GPU duty-cycles instead of running flat out. The
   calibration probes double as warm-up; one further discarded warmup measurement (`warmups`) follows.
4. **Reports the best run.** Every noise source a benchmark meets — throttling, clock ramp, compositor
   frames, other apps — only ever makes a run _slower_, so the minimum is the least-contaminated
   estimate of the device's capability. Throughput/bandwidth are derived from `stats.min`; the table
   updates after every measurement and its numbers only ever improve.
5. **Converges when the best stops improving.** A kernel retires once it has at least `minRounds` (3)
   kept measurements and its best hasn't improved by more than `improvementTolerance` (1%) over the
   last `stableRounds` (2) of them. On a cool GPU that's 3 measurements per kernel; a GPU still
   ramping its clock keeps sampling, up to `maxRounds` (10, flagged `max-rounds`).
6. **Discards throttled runs.** A measurement more than `throttleThreshold` (20%) slower than that
   kernel's best is thermal noise, not information: it's recorded in `throttledMs` for diagnostics
   but never feeds `stats` or convergence.
7. **Pauses the whole suite when the device is throttling.** If half the kernels in a round come back
   throttled (`throttledFraction`, at least two of them), the suite sleeps
   for `cooldownMs` (3s) and tries again — throttling recovers, so a later round can still beat the
   current best. After `maxCooldowns` (3) pauses it gives up: whatever is still unconverged is
   reported with its best run so far and `stopReason: 'throttled'` (shown as a warning in the table).

`onProgress` reports round boundaries and cooldowns; all of the knobs above are `SuiteOptions`.
GPU timing uses `timestamp-query` when available (pure device-side time, no CPU/driver overhead);
otherwise it falls back to wall-clock around `queue.onSubmittedWorkDone()`. "Available" is checked, not
trusted: every timestamp reading is compared with wall clock (minus the fixed submit/readback overhead,
measured with empty submits at calibration) and a kernel is demoted to wall-clock timing once the GPU
reading comes in under half of that twice running. Safari needs this — outside a cross-origin-isolated
context its timestamps are quantized to the point of reporting ~1ms for a 70ms batch — and all batch
sizing uses wall clock regardless, since a batch sized from a bogus timestamp is a multi-second command
buffer that hangs the GPU process. Two further hygiene rules: a measurement during which the tab was
hidden is dropped (background tabs get a throttled event loop and a lower-priority GPU queue) and the
suite waits for the page to be visible again; and the idle gap between measurements ends on
`requestIdleCallback`, so the results table's re-render and the compositor frame it triggers land in the
gap rather than inside the next measurement. Mean / median / stddev /
ci95 over the kept runs are still in `BenchmarkResult.stats` if you want a sustained number — the
table just shows the best.

## Monorepo layout

- `packages/webgpu-bench` — the benchmark suite itself (WGSL shaders, data generation, timing
  harness, orchestration). Framework-agnostic, browser-only, consumed as TypeScript source.
- `packages/website` — TanStack Start + Tailwind CSS + shadcn/ui app that runs the suite client-side and
  renders results incrementally. Dev server on **port 3500**.

## Development

```bash
pnpm install
pnpm dev        # starts the website on http://localhost:3500
pnpm build      # typecheck webgpu-bench + build the website
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
