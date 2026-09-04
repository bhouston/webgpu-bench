import type { GpuContext } from '../gpu/context.ts';
import type { GeneratedData } from '../data/generate.ts';
import { toF16Buffer } from '../data/generate.ts';
import { createUniformBuffer, createStorageBuffer, createEmptyStorageBuffer } from '../gpu/buffers.ts';
import { createPipeline, runKernelBenchmark, skippedResult, type HarnessConfig } from './common.ts';
import { matvecF16ScalarWgsl } from '../shaders/matvecF16Scalar.ts';
import { matvecF16Vec4Wgsl } from '../shaders/matvecF16Vec4.ts';
import type { BenchmarkResult } from '../types.ts';

export async function benchmarkF16Scalar(
  ctx: GpuContext,
  data: GeneratedData,
  harness: HarnessConfig = {},
): Promise<BenchmarkResult> {
  if (!ctx.info.supportsF16) {
    return skippedResult(
      'f16-scalar',
      'f16 scalar',
      'Scalar dot-product loop over f16 storage buffers.',
      'dtype',
      data.rows,
      data.cols,
      'Device/browser does not support the "shader-f16" WebGPU feature.',
    );
  }
  const { device } = ctx;
  const pipeline = createPipeline(device, 'matvec-f16-scalar', matvecF16ScalarWgsl);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([data.rows, data.cols]), 'params');
  const matrixBuf = createStorageBuffer(device, toF16Buffer(data.matrix), 'matrix');
  const vectorBuf = createStorageBuffer(device, toF16Buffer(data.vector), 'vector');
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
    id: 'f16-scalar',
    label: 'f16 scalar',
    description: 'One thread per output row; scalar dot-product loop over f16 storage buffers, f32 accumulation.',
    category: 'dtype',
    ctx,
    rows: data.rows,
    cols: data.cols,
    bytesPerOp: data.matrix.length * 2 + data.vector.length * 2,
    workgroupsPerIteration: [Math.ceil(data.rows / 64), 1, 1],
    pipeline,
    bindGroup,
    ...harness,
  });
}

export async function benchmarkF16Vec4(
  ctx: GpuContext,
  data: GeneratedData,
  harness: HarnessConfig = {},
): Promise<BenchmarkResult> {
  if (!ctx.info.supportsF16) {
    return skippedResult(
      'f16-vec4',
      'f16 vec4 (SIMD)',
      'K dimension walked 4 lanes at a time via vec4<f16>.',
      'layout',
      data.rows,
      data.cols,
      'Device/browser does not support the "shader-f16" WebGPU feature.',
    );
  }
  const { device } = ctx;
  const cols4 = data.cols / 4;
  const pipeline = createPipeline(device, 'matvec-f16-vec4', matvecF16Vec4Wgsl);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([data.rows, cols4]), 'params');
  const matrixBuf = createStorageBuffer(device, toF16Buffer(data.matrix), 'matrix');
  const vectorBuf = createStorageBuffer(device, toF16Buffer(data.vector), 'vector');
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
    id: 'f16-vec4',
    label: 'f16 vec4 (SIMD)',
    description: 'One thread per output row; K dimension walked 4 lanes at a time via vec4<f16>, f32 accumulation.',
    category: 'layout',
    ctx,
    rows: data.rows,
    cols: data.cols,
    bytesPerOp: data.matrix.length * 2 + data.vector.length * 2,
    workgroupsPerIteration: [Math.ceil(data.rows / 64), 1, 1],
    pipeline,
    bindGroup,
    ...harness,
  });
}
