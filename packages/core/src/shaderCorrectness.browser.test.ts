import { afterAll, beforeAll, expect, test } from 'vitest';
import { acquireGpuContext, type GpuContext } from './gpu/context.ts';
import { createPipeline } from './benchmarks/common.ts';
import { createStorageBuffer, createUniformBuffer, readFloat32 } from './gpu/buffers.ts';
import { createFloat32Texture } from './gpu/textures.ts';
import { scatterWriteWgsl } from './shaders/scatterWrite.ts';
import { prepareScatterWrite } from './benchmarks/streamBandwidth.ts';
import { generateMatVecData } from './data/generate.ts';
import { textureInterpBuiltinWgsl } from './shaders/textureInterpBuiltin.ts';
import { textureInterpManualWgsl } from './shaders/textureInterpManual.ts';
import { BENCHMARKS } from './catalog.ts';

let ctx: GpuContext;
beforeAll(async () => {
  ctx = await acquireGpuContext();
});
afterAll(() => {
  ctx?.device.destroy();
});

async function dispatch(source: string, buffers: GPUBuffer[], workgroups: number, extra: GPUBindGroupEntry[] = []) {
  const pipeline = await createPipeline(ctx.device, 'correctness', source);
  const bindGroup = ctx.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [...buffers.map((buffer, binding) => ({ binding, resource: { buffer } })), ...extra],
  });
  const encoder = ctx.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  ctx.device.queue.submit([encoder.finish()]);
}

// Check complete destination coverage, including rounded dispatches and multiple
// grid-stride trips. A sentinel beyond the window catches accidental writes.
for (const count of [1, 16, 1024, 8192]) {
  test(`scatter writes a permutation of ${count} vec4s exactly once`, async () => {
    const threads = count >= 8192 ? 512 : 256;
    const params = createUniformBuffer(ctx.device, new Uint32Array([count, threads, count - 1]));
    const initial = new Float32Array((count + 1) * 4).fill(-1);
    const output = createStorageBuffer(ctx.device, initial);
    try {
      await dispatch(scatterWriteWgsl, [params, output], threads / 256);
      const result = await readFloat32(ctx.device, output, initial.length);
      const destinations = new Set<number>();
      for (let i = 0; i < count; i++) {
        const dest = (Math.imul(i, 2654435761) + 1013904223) & (count - 1);
        destinations.add(dest);
        expect(Array.from(result.slice(dest * 4, dest * 4 + 4))).toEqual([i, i, i, i]);
      }
      expect(destinations.size).toBe(count);
      expect(Array.from(result.slice(count * 4))).toEqual([-1, -1, -1, -1]);
    } finally {
      params.destroy();
      output.destroy();
    }
  });
}

test('scatter accounts for the clamped window rather than the full buffer', async () => {
  const data = generateMatVecData(32, 48);
  for (const windowBytes of [256, 16384]) {
    const prepared = await prepareScatterWrite(ctx, data, {}, 'scatter-check', windowBytes);
    expect(prepared.kind).toBe('kernel');
    if (prepared.kind === 'kernel') {
      expect(prepared.meta.amountPerOp).toBe(Math.min(windowBytes, 4096));
    }
  }
});

function weights(seed: number, mat4: boolean): number[] {
  const h = Math.imul(seed, 0x9e3779b1) >>> 0;
  return [h & 255, (h >>> 8) & 255, (h >>> 16) & 255, h >>> 24].map((v) => (mat4 ? v + 32 : v - 127.5) / 4096);
}

// Independent CPU matrix operations, using double precision with tolerances for
// WGSL rounding/contraction. Test actual benchmark output, not a toy kernel.
function matrixReference(idx: number, iterations: number, mat4: boolean): number {
  let c = [0.125, 0.25, 0.375, 0.5];
  let x = mat4
    ? [0.1, 0.2, 0.3, 0.4].map((v) => v + (idx & 255) * 0.0001)
    : [(idx & 255) * 0.001 + 0.1, 0.2, 0.3, 0.4, (idx & 255) * 0.001 + 0.2, 0.4, 0.6, 0.8];
  const columns = Array.from({ length: 4 }, (_, col) => weights(idx * 4 + col + 1, true));
  const rows = Array.from({ length: 4 }, (_, row) => [
    ...weights(idx * 8 + row * 2 + 1, false),
    ...weights(idx * 8 + row * 2 + 2, false),
  ]);
  for (let i = 0; i < iterations; i++) {
    const next = c.map(
      (v, row) => v + x.reduce((sum, value, col) => sum + value * (mat4 ? columns[col]![row]! : rows[row]![col]!), 0),
    );
    x = mat4 ? next : [...x.slice(4), ...next];
    c = [c[1]!, c[2]!, c[3]!, c[0]!];
  }
  return x.reduce((sum, value) => sum + value, 0);
}

for (const precision of ['f32', 'f16']) {
  for (const shape of ['mat4', 'matvec']) {
    test(`${precision} ${shape} matches reference and stays nonzero through calibration range`, async (t) => {
      if (precision === 'f16' && !ctx.info.supportsF16) t.skip();
      const source = BENCHMARKS.find((b) => b.id === `${precision}-fma-${shape}`)!.source;
      const threads = 65; // Exercise the early return in the final workgroup.
      const params = createUniformBuffer(ctx.device, new Uint32Array([threads, 1]));
      const output = createStorageBuffer(ctx.device, new Float32Array(threads + 1).fill(-99));
      try {
        for (const iterations of [1, 16, 17, 128, 512, 1024]) {
          ctx.device.queue.writeBuffer(params, 0, new Uint32Array([threads, iterations]));
          await dispatch(source, [params, output], 2);
          const result = await readFloat32(ctx.device, output, threads + 1);
          for (let idx = 0; idx < threads; idx++) {
            const expected = matrixReference(idx, iterations, shape === 'mat4');
            expect(Math.abs(result[idx]! - expected), `idx=${idx}, iterations=${iterations}`).toBeLessThan(
              precision === 'f16' ? 0.015 : 0.00001,
            );
            expect(result[idx]!).toBeGreaterThan(0.1);
            expect(result[idx]!).toBeLessThan(8);
          }
          expect(result[threads]).toBe(-99);
        }
      } finally {
        params.destroy();
        output.destroy();
      }
    });
  }
}

for (const builtin of [false, true]) {
  test(`${builtin ? 'sampled' : 'manual'} texture interpolation matches repeat-wrapped CPU reference`, async (t) => {
    if (builtin && !ctx.info.supportsFloat32Filterable) t.skip();
    // 32 phases across 8 texels cover seams, exact centers and quarter weights.
    // 33 threads additionally exercise non-exact fractions and a partial group.
    for (const threads of [32, 33]) {
      const width = 8;
      const texels = Float32Array.from({ length: width * 4 }, (_, i) => ((i * 13) % 31) / 32 - 0.5);
      const texture = createFloat32Texture(ctx.device, texels, width);
      const params = createUniformBuffer(ctx.device, new Uint32Array([threads, 1, width]));
      const output = createStorageBuffer(ctx.device, new Float32Array(threads + 1).fill(-99));
      const sampler = ctx.device.createSampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'repeat' });
      const extra: GPUBindGroupEntry[] = [{ binding: 1, resource: texture.createView() }];
      if (builtin) extra.push({ binding: 2, resource: sampler });
      extra.push({ binding: builtin ? 3 : 2, resource: { buffer: output } });
      try {
        for (const iterations of [1, 16, 256]) {
          ctx.device.queue.writeBuffer(params, 0, new Uint32Array([threads, iterations, width]));
          await dispatch(builtin ? textureInterpBuiltinWgsl : textureInterpManualWgsl, [params], 1, extra);
          const result = await readFloat32(ctx.device, output, threads + 1);
          for (let idx = 0; idx < threads; idx++) {
            let expected = 0;
            for (let i = 0; i < iterations; i++) {
              const u = (idx / threads + (i * 0.25) / width) % 1;
              const pos = u * width - 0.5;
              const left = Math.floor(pos);
              const weight = pos - left;
              for (let channel = 0; channel < 4; channel++) {
                expected +=
                  texels[((left + width) % width) * 4 + channel]! * (1 - weight) +
                  texels[((left + 1 + width) % width) * 4 + channel]! * weight;
              }
            }
            // Filtering weights may have limited precision; use a per-sample
            // tolerance for hardware filtering, tighter for manual arithmetic.
            expect(Math.abs(result[idx]! - expected)).toBeLessThan(iterations * (builtin ? 0.005 : 0.00001));
          }
          expect(result[threads]).toBe(-99);
        }
      } finally {
        params.destroy();
        output.destroy();
        texture.destroy();
      }
    }
  });
}

// Broad numerical smoke coverage complements the detailed references above.
// This intentionally runs in Vitest only: no readback in timed benchmarks.
for (const benchmark of BENCHMARKS.filter((b) => b.category === 'compute')) {
  test(`${benchmark.id} produces finite outputs at the maximum configured loop count`, async (t) => {
    const prepared = await benchmark.prepare({ ctx, data: generateMatVecData(4, 4), harness: {}, computeThreads: 257 });
    if (prepared.kind === 'skipped') t.skip();
    if (prepared.kind !== 'kernel') return;
    const params = createUniformBuffer(ctx.device, new Uint32Array([257, prepared.meta.cols]));
    const output = createStorageBuffer(ctx.device, new Float32Array(257).fill(Number.NaN));
    try {
      await dispatch(benchmark.source, [params, output], 5);
      const values = await readFloat32(ctx.device, output, 257);
      expect(values.every(Number.isFinite)).toBe(true);
    } finally {
      params.destroy();
      output.destroy();
    }
  });
}
