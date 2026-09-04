import type { GpuContext } from '../gpu/context.ts';
import type { GeneratedData } from '../data/generate.ts';
import { createUniformBuffer, createStorageBuffer, createEmptyStorageBuffer } from '../gpu/buffers.ts';
import { createPipeline, runKernelBenchmark, type HarnessConfig } from './common.ts';
import { matvecF32Vec4Wgsl } from '../shaders/matvecF32Vec4.ts';
import type { BenchmarkResult } from '../types.ts';

export async function benchmarkF32Vec4(
  ctx: GpuContext,
  data: GeneratedData,
  harness: HarnessConfig = {},
): Promise<BenchmarkResult> {
  const { device } = ctx;
  const cols4 = data.cols / 4;
  const pipeline = createPipeline(device, 'matvec-f32-vec4', matvecF32Vec4Wgsl);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([data.rows, cols4]), 'params');
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

  return runKernelBenchmark({
    id: 'f32-vec4',
    label: 'f32 vec4 (SIMD)',
    description: 'One thread per output row; K dimension walked 4 lanes at a time via vec4<f32> loads/multiplies.',
    category: 'layout',
    ctx,
    rows: data.rows,
    cols: data.cols,
    bytesPerOp: data.matrix.byteLength + data.vector.byteLength,
    workgroupsPerIteration: [Math.ceil(data.rows / 64), 1, 1],
    pipeline,
    bindGroup,
    ...harness,
  });
}
