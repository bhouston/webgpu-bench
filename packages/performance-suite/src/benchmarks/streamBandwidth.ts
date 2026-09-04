import type { GpuContext } from '../gpu/context.ts';
import type { GeneratedData } from '../data/generate.ts';
import { createUniformBuffer, createStorageBuffer, createEmptyStorageBuffer } from '../gpu/buffers.ts';
import { createPipeline, runKernelBenchmark, type HarnessConfig } from './common.ts';
import { streamReadWgsl } from '../shaders/streamRead.ts';
import { streamWriteWgsl } from '../shaders/streamWrite.ts';
import type { BenchmarkResult } from '../types.ts';

/**
 * Read-bandwidth test: streams the (already-allocated, matrix-sized) input
 * buffer through each thread with nothing but addition, writing back a
 * single scalar per thread. Reads vastly outweigh writes, so `gbps` here is
 * a read-bandwidth-bound number close to the device's peak.
 */
export async function benchmarkReadBandwidth(
  ctx: GpuContext,
  data: GeneratedData,
  harness: HarnessConfig = {},
): Promise<BenchmarkResult> {
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

  return runKernelBenchmark({
    id: 'read-bandwidth',
    label: 'Read bandwidth',
    description: 'One thread per row; streams a large buffer in via vec4<f32> loads and addition only, writes one scalar. Read-bandwidth-bound.',
    category: 'bandwidth',
    ctx,
    rows: data.rows,
    cols: data.cols,
    bytes: data.matrix.byteLength,
    // One add per element read — negligible compute, reported for completeness.
    flops: data.rows * data.cols,
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
export async function benchmarkWriteBandwidth(
  ctx: GpuContext,
  data: GeneratedData,
  harness: HarnessConfig = {},
): Promise<BenchmarkResult> {
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

  return runKernelBenchmark({
    id: 'write-bandwidth',
    label: 'Write bandwidth',
    description: 'One thread per row; stores computed vec4<f32> values into a large buffer with no buffer reads. Write-bandwidth-bound.',
    category: 'bandwidth',
    ctx,
    rows: data.rows,
    cols: data.cols,
    bytes: data.matrix.byteLength,
    // No arithmetic beyond forming the value to store.
    flops: 0,
    workgroupsPerIteration: [Math.ceil(data.rows / 64), 1, 1],
    pipeline,
    bindGroup,
    ...harness,
  });
}
