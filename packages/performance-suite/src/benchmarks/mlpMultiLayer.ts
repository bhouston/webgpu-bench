import type { GpuContext } from '../gpu/context.ts';
import { generateMatVecData, padToMultipleOf4 } from '../data/generate.ts';
import { createUniformBuffer, createStorageBuffer, createEmptyStorageBuffer } from '../gpu/buffers.ts';
import { createPipeline } from './common.ts';
import { runTimedBenchmark } from '../gpu/benchmarkRunner.ts';
import { matvecF32Vec4SharedWgsl } from '../shaders/matvecF32Vec4Shared.ts';
import type { BenchmarkResult } from '../types.ts';

export interface MlpLayerSpec {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  rows: number;
  cols: number;
}

/**
 * Chains several matvec "layers" entirely on the GPU: layer i's output
 * buffer is layer i+1's input buffer, all dispatches for one forward pass
 * are recorded into a single compute pass / single command-buffer submit,
 * and the CPU only reads back the final layer's output — never the
 * intermediates. ReLU is fused into every layer but the last (using the
 * same tiled/shared-memory kernel the workgroup sweep tunes), so there's no
 * separate activation dispatch either.
 */
export async function benchmarkMlpMultiLayer(
  ctx: GpuContext,
  opts: {
    inputCols: number;
    layerRows: number[];
    wgSize?: number;
    seed?: number;
    targetMs?: number;
    warmups?: number;
    runs?: number;
  },
): Promise<BenchmarkResult> {
  const { device } = ctx;
  const wgSize = opts.wgSize ?? 128;
  const seed = opts.seed ?? 9001;
  const pipeline = createPipeline(device, `mlp-layer-wg${wgSize}`, matvecF32Vec4SharedWgsl, { WG_SIZE: wgSize });

  const inputCols = padToMultipleOf4(opts.inputCols);
  const initial = generateMatVecData(opts.layerRows[0]!, inputCols, seed);
  let inputBuf = createStorageBuffer(device, initial.vector, 'mlp-input');
  let prevCols = inputCols;

  const layers: MlpLayerSpec[] = [];
  let totalFlops = 0;
  let totalBytes = 0;

  for (let i = 0; i < opts.layerRows.length; i++) {
    const rows = padToMultipleOf4(opts.layerRows[i]!);
    const cols = prevCols;
    const layerData = i === 0 ? initial : generateMatVecData(rows, cols, seed + i);
    const matrixBuf = createStorageBuffer(device, layerData.matrix, `mlp-weights-${i}`);
    const outBuf = createEmptyStorageBuffer(device, rows * 4, `mlp-out-${i}`);
    const isLast = i === opts.layerRows.length - 1;
    const paramsBuf = createUniformBuffer(
      device,
      new Uint32Array([rows, cols / 4, isLast ? 0 : 1, 0]),
      `mlp-params-${i}`,
    );
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: matrixBuf } },
        { binding: 2, resource: { buffer: inputBuf } },
        { binding: 3, resource: { buffer: outBuf } },
      ],
    });
    layers.push({ pipeline, bindGroup, rows, cols });
    totalFlops += 2 * rows * cols;
    totalBytes += layerData.matrix.byteLength;

    inputBuf = outBuf;
    prevCols = rows;
  }

  const timed = await runTimedBenchmark({
    device,
    useTimestamps: ctx.info.supportsTimestampQuery,
    targetMs: opts.targetMs,
    warmups: opts.warmups,
    runs: opts.runs,
    encode: (pass, iterations) => {
      for (let i = 0; i < iterations; i++) {
        for (const layer of layers) {
          pass.setPipeline(layer.pipeline);
          pass.setBindGroup(0, layer.bindGroup);
          pass.dispatchWorkgroups(layer.rows, 1, 1);
        }
      }
    },
  });

  const seconds = timed.stats.mean / 1000;
  const gflops = totalFlops / seconds / 1e9;
  const gbps = totalBytes / seconds / 1e9;
  const finalRows = opts.layerRows[opts.layerRows.length - 1]!;

  return {
    id: 'mlp-multilayer',
    label: `MLP, ${opts.layerRows.length} layers (fused, no readbacks)`,
    description: `Chains ${opts.layerRows.length} matvec layers (sizes ${opts.layerRows.join(' -> ')}) with fused ReLU, all in one command-buffer submit; only the final layer's output is read back.`,
    category: 'multilayer',
    status: 'ok',
    rows: finalRows,
    cols: inputCols,
    innerIterations: timed.innerIterations,
    timesMs: timed.timesMs,
    stats: timed.stats,
    gflops,
    gbps,
    timingMethod: timed.timingMethod,
  };
}
