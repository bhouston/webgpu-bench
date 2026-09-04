import type { GpuContext } from '../gpu/context.ts';
import type { GeneratedData } from '../data/generate.ts';
import { quantizeInt8, packInt8x4 } from '../data/generate.ts';
import { createUniformBuffer, createStorageBuffer, createEmptyStorageBuffer } from '../gpu/buffers.ts';
import { createPipeline, runKernelBenchmark, skippedResult, type HarnessConfig } from './common.ts';
import { matvecI8PackedWgsl } from '../shaders/matvecI8Packed.ts';
import type { BenchmarkResult } from '../types.ts';

export async function benchmarkI8Packed(
  ctx: GpuContext,
  data: GeneratedData,
  harness: HarnessConfig = {},
): Promise<BenchmarkResult> {
  if (!ctx.info.supportsI8Dot) {
    return skippedResult(
      'i8-packed',
      'int8 packed dot product',
      'Symmetric per-tensor int8 quantization, dot4I8Packed 4-wide integer dot products.',
      'dtype',
      data.rows,
      data.cols,
      'Device/browser does not support the "packed_4x8_integer_dot_product" WebGPU feature.',
    );
  }
  const { device } = ctx;
  const cols4 = data.cols / 4;

  const matQ = quantizeInt8(data.matrix);
  const vecQ = quantizeInt8(data.vector);
  const matrixPacked = packInt8x4(matQ.data);
  const vectorPacked = packInt8x4(vecQ.data);

  const pipeline = createPipeline(device, 'matvec-i8-packed', matvecI8PackedWgsl);

  const paramsData = new ArrayBuffer(16);
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, data.rows, true);
  paramsView.setUint32(4, cols4, true);
  paramsView.setFloat32(8, matQ.scale, true);
  paramsView.setFloat32(12, vecQ.scale, true);
  const paramsBuf = createUniformBuffer(device, new Uint8Array(paramsData), 'params');

  const matrixBuf = createStorageBuffer(device, matrixPacked, 'matrix');
  const vectorBuf = createStorageBuffer(device, vectorPacked, 'vector');
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
    id: 'i8-packed',
    label: 'int8 packed dot product',
    description:
      'One thread per output row; symmetric per-tensor int8 quantization, dot4I8Packed 4-wide integer dot products, dequantized on write-out.',
    category: 'dtype',
    ctx,
    rows: data.rows,
    cols: data.cols,
    bytesPerOp: matrixPacked.byteLength + vectorPacked.byteLength,
    workgroupsPerIteration: [Math.ceil(data.rows / 64), 1, 1],
    pipeline,
    bindGroup,
    ...harness,
  });
}
