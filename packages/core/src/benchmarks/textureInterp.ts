import type { GpuContext } from '../gpu/context.ts';
import type { WorkKnob } from '../gpu/benchmarkRunner.ts';
import { createUniformBuffer, createEmptyStorageBuffer } from '../gpu/buffers.ts';
import { createFloat32Texture } from '../gpu/textures.ts';
import { mulberry32 } from '../data/generate.ts';
import {
  createPipeline,
  prepareKernelBenchmark,
  skippedResult,
  type BenchmarkMeta,
  type PreparedBenchmark,
} from './common.ts';
import type { FlopsHarnessConfig } from './flopsCommon.ts';
import { textureInterpBuiltinWgsl } from '../shaders/textureInterpBuiltin.ts';
import { textureInterpManualWgsl } from '../shaders/textureInterpManual.ts';

/** Texels in the small texture both kernels sweep across. Small enough to comfortably fit any GPU's 2D width limit. */
const TEXEL_COUNT = 4096;
const DEFAULT_THREADS = 1024 * 1024;
const DEFAULT_ITERATIONS = 256;
const DEFAULT_MIN_ITERATIONS = 16;

function generateTexelData(texelCount: number, seed = 4242): Float32Array {
  const rand = mulberry32(seed);
  const data = new Float32Array(texelCount * 4);
  for (let i = 0; i < data.length; i++) data[i] = rand() * 2 - 1;
  return data;
}

interface TextureInterpSpec {
  id: string;
  wgsl: string;
  /** Builtin path samples through a filtering sampler at binding 2 (out moves to binding 3); the manual path has no sampler and `textureLoad`s the texture directly. */
  useSampler: boolean;
  requiresFloat32Filterable?: boolean;
}

/**
 * Shared runner for the two texture-interpolation kernels: same texture,
 * same "zoom in" linear scan, same Params{threads,iterations,texelCount}
 * uniform — only the WGSL (and whether a sampler is bound) differs.
 */
async function prepareTextureInterp(
  ctx: GpuContext,
  spec: TextureInterpSpec,
  harness: FlopsHarnessConfig = {},
): Promise<PreparedBenchmark> {
  const threads = harness.threads ?? DEFAULT_THREADS;
  const maxIterations = harness.iterations ?? DEFAULT_ITERATIONS;
  const metaAtWork = (iterations: number): BenchmarkMeta => ({
    id: spec.id,
    category: 'bandwidth',
    rows: threads,
    cols: iterations,
    // Two vec4<f32> texel fetches per sample (the interpolation pair) — the same accounting for both kernels so they compare directly.
    amountPerOp: threads * iterations * 2 * 16,
  });
  const meta = metaAtWork(maxIterations);

  if (spec.requiresFloat32Filterable && !ctx.info.supportsFloat32Filterable) {
    return {
      kind: 'skipped',
      result: skippedResult(meta, 'Device/browser does not support the "float32-filterable" WebGPU feature.'),
    };
  }

  const { device } = ctx;
  const pipeline = await createPipeline(device, spec.id, spec.wgsl);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([threads, maxIterations, TEXEL_COUNT]), 'params');
  const texture = createFloat32Texture(device, generateTexelData(TEXEL_COUNT), TEXEL_COUNT, 'texel-data');
  const outBuf = createEmptyStorageBuffer(device, threads * 4, 'out');
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: paramsBuf } },
    { binding: 1, resource: texture.createView() },
  ];
  if (spec.useSampler) {
    const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'repeat' });
    entries.push({ binding: 2, resource: sampler }, { binding: 3, resource: { buffer: outBuf } });
  } else {
    entries.push({ binding: 2, resource: { buffer: outBuf } });
  }
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });

  // An explicit `iterations` pins the trip count; otherwise let the sampler size it.
  const work: WorkKnob | undefined =
    harness.iterations === undefined
      ? {
          min: Math.min(maxIterations, DEFAULT_MIN_ITERATIONS),
          max: maxIterations,
          apply: (iterations) =>
            device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([threads, iterations, TEXEL_COUNT])),
        }
      : undefined;

  return prepareKernelBenchmark({
    ...meta,
    ctx,
    workgroupsPerIteration: [Math.ceil(threads / 256), 1, 1],
    pipeline,
    bindGroup,
    work,
    metaAtWork,
    targetMs: harness.targetMs,
    targetDispatchMs: harness.targetDispatchMs,
    warmups: harness.warmups,
    maxIterations: harness.maxIterations,
  });
}

export function prepareTextureInterpBuiltin(ctx: GpuContext, harness?: FlopsHarnessConfig): Promise<PreparedBenchmark> {
  return prepareTextureInterp(
    ctx,
    { id: 'texture-interp-builtin', wgsl: textureInterpBuiltinWgsl, useSampler: true, requiresFloat32Filterable: true },
    harness,
  );
}

export function prepareTextureInterpManual(ctx: GpuContext, harness?: FlopsHarnessConfig): Promise<PreparedBenchmark> {
  return prepareTextureInterp(
    ctx,
    { id: 'texture-interp-manual', wgsl: textureInterpManualWgsl, useSampler: false },
    harness,
  );
}
