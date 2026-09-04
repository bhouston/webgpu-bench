import type { GpuContext } from '../gpu/context.ts';
import type { GeneratedData } from '../data/generate.ts';
import { createUniformBuffer, createStorageBuffer, createEmptyStorageBuffer } from '../gpu/buffers.ts';
import { createPipeline, prepareKernelBenchmark, type HarnessConfig, type PreparedBenchmark } from './common.ts';
import { streamReadWgsl } from '../shaders/streamRead.ts';
import { streamWriteWgsl } from '../shaders/streamWrite.ts';

/**
 * Read-bandwidth test: streams the (already-allocated, matrix-sized) input
 * buffer through each thread with nothing but addition, writing back a
 * single scalar per thread. Reads vastly outweigh writes, so `gbps` here is
 * a read-bandwidth-bound number close to the device's peak.
 */
export async function prepareReadBandwidth(
  ctx: GpuContext,
  data: GeneratedData,
  harness: HarnessConfig = {},
): Promise<PreparedBenchmark> {
  const { device } = ctx;
  const cols4 = data.cols / 4;
  const pipeline = await createPipeline(device, 'stream-read', streamReadWgsl);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([data.rows, cols4]), 'params');
  const dataBuf = createStorageBuffer(device, data.matrix, 'data');
  const outBuf = createEmptyStorageBuffer(device, data.rows * 4, 'out');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: dataBuf } },
      { binding: 2, resource: { buffer: outBuf } },
    ],
  });

  return prepareKernelBenchmark({
    id: 'read-bandwidth',
    category: 'bandwidth',
    ctx,
    rows: data.rows,
    cols: data.cols,
    amountPerOp: data.matrix.byteLength,
    workgroupsPerIteration: [Math.ceil(data.rows / 64), 1, 1],
    pipeline,
    bindGroup,
    ...harness,
  });
}

/**
 * Write-bandwidth test: streams computed values out into a matrix-sized
 * output buffer. No buffer reads at all, so `gbps` here is a
 * write-bandwidth-bound number close to the device's peak.
 */
export async function prepareWriteBandwidth(
  ctx: GpuContext,
  data: GeneratedData,
  harness: HarnessConfig = {},
): Promise<PreparedBenchmark> {
  const { device } = ctx;
  const cols4 = data.cols / 4;
  const pipeline = await createPipeline(device, 'stream-write', streamWriteWgsl);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([data.rows, cols4]), 'params');
  const outBuf = createEmptyStorageBuffer(device, data.matrix.byteLength, 'out');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: outBuf } },
    ],
  });

  return prepareKernelBenchmark({
    id: 'write-bandwidth',
    category: 'bandwidth',
    ctx,
    rows: data.rows,
    cols: data.cols,
    amountPerOp: data.matrix.byteLength,
    workgroupsPerIteration: [Math.ceil(data.rows / 64), 1, 1],
    pipeline,
    bindGroup,
    ...harness,
  });
}
