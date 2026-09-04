import type { GpuContext } from '../gpu/context.ts';
import type { GeneratedData } from '../data/generate.ts';
import { createUniformBuffer, createStorageBuffer, createEmptyStorageBuffer } from '../gpu/buffers.ts';
import { createPipeline, runKernelBenchmark, type HarnessConfig } from './common.ts';
import { matvecF32Vec4SharedWgsl } from '../shaders/matvecF32Vec4Shared.ts';
import type { BenchmarkResult } from '../types.ts';

/** Builds (but does not run) the tiled, workgroup-shared-memory kernel — reused by the workgroup-size sweep and the MLP. */
export function buildF32Vec4SharedPipeline(
  ctx: GpuContext,
  data: GeneratedData,
  wgSize: number,
): { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; paramsBuf: GPUBuffer; outBuf: GPUBuffer } {
  const { device } = ctx;
  const cols4 = data.cols / 4;
  const pipeline = createPipeline(device, `matvec-f32-vec4-shared-wg${wgSize}`, matvecF32Vec4SharedWgsl, {
    WG_SIZE: wgSize,
  });
  const paramsBuf = createUniformBuffer(device, new Uint32Array([data.rows, cols4, 0, 0]), 'params');
  const matrixBuf = createStorageBuffer(device, data.matrix, 'matrix');
  const vectorBuf = createStorageBuffer(device, data.vector, 'vector');
  const outBuf = createEmptyStorageBuffer(device, data.rows * 4, 'out');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: matrixBuf } },
      { binding: 2, resource: { buffer: vectorBuf } },
      { binding: 3, resource: { buffer: outBuf } },
    ],
  });
  return { pipeline, bindGroup, paramsBuf, outBuf };
}

export async function benchmarkF32Vec4Shared(
  ctx: GpuContext,
  data: GeneratedData,
  wgSize = 128,
  harness: HarnessConfig = {},
): Promise<BenchmarkResult> {
  const { pipeline, bindGroup } = buildF32Vec4SharedPipeline(ctx, data, wgSize);

  return runKernelBenchmark({
    id: `f32-vec4-shared-wg${wgSize}`,
    label: `f32 vec4 tiled, workgroup=${wgSize}`,
    description:
      'One workgroup per output row; threads split the K dimension (vec4 lanes) and tree-reduce partial sums in workgroup shared memory.',
    category: 'layout',
    ctx,
    rows: data.rows,
    cols: data.cols,
    bytesPerOp: data.matrix.byteLength + data.vector.byteLength,
    workgroupsPerIteration: [data.rows, 1, 1],
    pipeline,
    bindGroup,
    ...harness,
  });
}
