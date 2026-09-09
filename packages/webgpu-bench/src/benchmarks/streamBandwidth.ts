import type { GpuContext } from '../gpu/context.ts';
import type { GeneratedData } from '../data/generate.ts';
import { createUniformBuffer, createStorageBuffer, createEmptyStorageBuffer } from '../gpu/buffers.ts';
import { createPipeline, prepareKernelBenchmark, type HarnessConfig, type PreparedBenchmark } from './common.ts';
import { streamReadWgsl } from '../shaders/streamRead.ts';
import { streamWriteWgsl } from '../shaders/streamWrite.ts';

const WORKGROUP_SIZE = 256;
/**
 * vec4 loads/stores per thread. Threads are sized from the buffer, not the
 * GPU: WebGPU limits say nothing about core count or memory bus, so instead
 * every device gets a shape that's good everywhere — coalesced grid-stride
 * access, a few independent accesses per thread to hide latency, and enough
 * threads (256K for the default 64 MB buffer) to fill a wide discrete GPU,
 * while a small mobile GPU just drains the same workgroups over more waves.
 * On an M3 this is within a few percent of the best thread count in a sweep.
 */
const VEC4S_PER_THREAD = 16;

function bandwidthShape(data: GeneratedData): { count: number; threads: number; workgroups: number } {
  const count = data.matrix.length / 4;
  const workgroups = Math.max(1, Math.ceil(count / VEC4S_PER_THREAD / WORKGROUP_SIZE));
  return { count, threads: workgroups * WORKGROUP_SIZE, workgroups };
}

/**
 * Read-bandwidth test: streams the (already-allocated, matrix-sized) input
 * buffer through a coalesced grid-stride loop with nothing but addition,
 * writing back a single scalar per thread. Reads vastly outweigh writes, so
 * `gbps` here is a read-bandwidth-bound number close to the device's peak.
 */
export async function prepareReadBandwidth(
  ctx: GpuContext,
  data: GeneratedData,
  harness: HarnessConfig = {},
): Promise<PreparedBenchmark> {
  const { device } = ctx;
  const { count, threads, workgroups } = bandwidthShape(data);
  const pipeline = await createPipeline(device, 'stream-read', streamReadWgsl);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([count, threads]), 'params');
  const dataBuf = createStorageBuffer(device, data.matrix, 'data');
  const outBuf = createEmptyStorageBuffer(device, threads * 4, 'out');
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
    workgroupsPerIteration: [workgroups, 1, 1],
    pipeline,
    bindGroup,
    ...harness,
  });
}

/**
 * Write-bandwidth test: streams computed values out into a matrix-sized
 * output buffer through the same coalesced grid-stride loop. No buffer reads
 * at all, so `gbps` here is a write-bandwidth-bound number close to the
 * device's peak.
 */
export async function prepareWriteBandwidth(
  ctx: GpuContext,
  data: GeneratedData,
  harness: HarnessConfig = {},
): Promise<PreparedBenchmark> {
  const { device } = ctx;
  const { count, threads, workgroups } = bandwidthShape(data);
  const pipeline = await createPipeline(device, 'stream-write', streamWriteWgsl);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([count, threads]), 'params');
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
    workgroupsPerIteration: [workgroups, 1, 1],
    pipeline,
    bindGroup,
    ...harness,
  });
}
