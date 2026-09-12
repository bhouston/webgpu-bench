import { afterAll, beforeAll, expect, test } from 'vitest';
import { acquireGpuContext, type GpuContext } from './gpu/context.ts';
import { readFloat32 } from './gpu/buffers.ts';
import { KernelSampler } from './gpu/benchmarkRunner.ts';
import {
  TECHNIQUE_BENCHMARKS,
  TECHNIQUE_SPECS,
  buildTechnique,
  techniqueData,
  techniqueId,
  particleAddress,
  type TechniqueSpec,
} from './benchmarks/techniques.ts';
import { BENCHMARKS } from './catalog.ts';

let ctx: GpuContext;
beforeAll(async () => {
  ctx = await acquireGpuContext();
});
afterAll(() => {
  ctx?.device.destroy();
});

function reference(spec: TechniqueSpec, data: ReturnType<typeof techniqueData>): number[] {
  const { input, count, padded, rounds } = data;
  if (spec.family === 'layout') {
    const result = Array.from(input);
    for (let i = 0; i < count; i++)
      for (let field = 0; field < 3; field++) {
        const p = particleAddress(spec.variant, i, field, padded);
        result[p] = input[p]! + input[particleAddress(spec.variant, i, field + 3, padded)]! * 0.125;
      }
    return result;
  }
  if (spec.family === 'atomic') {
    const bins = Array<number>(32).fill(0);
    for (let i = 0; i < count; i++) bins[input[i]!]!++;
    return bins;
  }
  if (spec.family === 'reduction') {
    const sums = Array<number>(Math.ceil(count / 256)).fill(0);
    for (let i = 0; i < count; i++) sums[Math.floor(i / 256)]! += input[i]!;
    return sums;
  }
  if (spec.family === 'dependent') {
    let index = 0;
    for (let hop = 0; hop < rounds; hop++) index = input[index]!;
    return [index];
  }
  if (spec.family === 'tile')
    return Array.from({ length: count }, (_, i) => {
      const base = Math.floor(i / 256) * 256;
      const row = Math.floor((i % 256) / 16);
      const col = i % 16;
      let sum = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          sum += input[base + ((row + dy + 16) % 16) * 16 + ((col + dx + 16) % 16)]!;
        }
      return sum / 9;
    });
  if (spec.family === 'branch') {
    const result = Array.from(input.slice(0, count * 4));
    for (let i = 0; i < count; i++) {
      let h = Math.imul(i, 2654435761) >>> 0;
      for (let k = 0; k < rounds; k++) {
        h = (Math.imul(h, 1664525) + 1013904223) >>> 0;
        for (let c = 0; c < 4; c++) {
          const x = result[i * 4 + c]!;
          result[i * 4 + c] = (h >>> (c * 8)) & 1 ? x * 0.5 - 0.5 : x * 0.75 + 0.25;
        }
      }
    }
    return result;
  }
  return Array.from(input.slice(0, count), (v) => {
    for (let k = 0; k < rounds; k++) v = v * 0.75 + 0.125;
    return v;
  });
}

async function execute(built: Awaited<ReturnType<typeof buildTechnique>>, repeats: number) {
  if (built.prepared.kind !== 'kernel' || !built.output) throw new Error('Expected supported technique');
  const encoder = ctx.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  built.prepared.harness.encode(pass, repeats);
  pass.end();
  ctx.device.queue.submit([encoder.finish()]);
  const floats = await readFloat32(ctx.device, built.output, built.data.initialOutput.length);
  return Array.from(built.data.integer ? new Uint32Array(floats.buffer) : floats);
}

for (const spec of TECHNIQUE_SPECS) {
  test(`${techniqueId(spec)} matches CPU reference, including tails and repeated batches`, async (t) => {
    if (spec.family === 'reduction' && spec.variant === 'subgroup' && !ctx.device.features.has('subgroups')) t.skip();
    // Tiny cases, multiple workgroups/tiles, and a partial final group. Atomic
    // cases span more than eight groups to exercise shard reuse as well.
    for (const count of spec.family === 'atomic' ? [1, 257, 36865] : [1, 777]) {
      const built = await buildTechnique(ctx, spec, {}, count);
      try {
        const expected = reference(spec, built.data);
        for (const repetitions of [1, 3]) {
          const actual = await execute(built, repetitions);
          expect(actual.length).toBe(expected.length);
          let maxError = 0;
          for (let i = 0; i < actual.length; i++) {
            expect(Number.isFinite(actual[i])).toBe(true);
            maxError = Math.max(maxError, Math.abs(actual[i]! - expected[i]!));
          }
          expect(maxError, `count=${count}, repetitions=${repetitions}`).toBeLessThan(built.data.integer ? 0.5 : 1e-6);
        }
      } finally {
        built.destroy();
      }
    }
  });
}

test('catalog registers exactly the requested 18 distinct comparisons with fixed useful-work metrics', async () => {
  expect(TECHNIQUE_BENCHMARKS).toHaveLength(18);
  expect(new Set(BENCHMARKS.map((b) => b.id)).size).toBe(BENCHMARKS.length);
  for (const definition of TECHNIQUE_BENCHMARKS) {
    expect(BENCHMARKS).toContain(definition);
    expect(definition.metric.unit).not.toBe('FLOP');
  }
  for (const spec of TECHNIQUE_SPECS) {
    const built = await buildTechnique(ctx, spec, {}, 777);
    try {
      if (built.prepared.kind !== 'kernel') continue;
      expect(built.prepared.harness.work).toBeUndefined();
      expect(built.prepared.meta.amountPerOp).toBe(
        spec.family === 'dependent' ? 4096 : spec.family === 'branch' ? 777 * 64 * 4 : 777,
      );
    } finally {
      built.destroy();
    }
  }
});

test('dependent input is a single reproducible cycle visiting the whole footprint', () => {
  const spec = { family: 'dependent', variant: 'chain' } as const;
  const data = techniqueData(spec, 8192);
  expect(data.input).toEqual(techniqueData(spec, 8192).input);
  const seen = new Set<number>();
  let index = 0;
  for (let hop = 0; hop < 8192; hop++) {
    seen.add(index);
    index = data.input[index]!;
  }
  expect(seen.size).toBe(8192);
  expect(index).toBe(0);
});

test('subgroup reduction skips before pipeline creation on a device without subgroups', async () => {
  // A real device requested without optional features exercises the fallback
  // even on machines where acquireGpuContext enables subgroup support.
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No adapter available for subgroup fallback test');
  const device = await adapter.requestDevice();
  try {
    const built = await buildTechnique({ ...ctx, device }, { family: 'reduction', variant: 'subgroup' }, {}, 1);
    expect(built.prepared.kind).toBe('skipped');
    if (built.prepared.kind === 'skipped') expect(built.prepared.result.message).toContain('subgroups');
  } finally {
    device.destroy();
  }
});

// Exercise the actual default sizes and timing encoder, including all three
// histogram stages. Keep this short: correctness tests don't rank performance.
for (const spec of TECHNIQUE_SPECS) {
  test(`${techniqueId(spec)} default workload produces a positive timed sample`, async (t) => {
    const built = await buildTechnique(ctx, spec, { targetMs: 5, warmups: 0, maxIterations: 32 });
    try {
      if (built.prepared.kind === 'skipped') t.skip();
      if (built.prepared.kind !== 'kernel') return;
      const sampler = new KernelSampler(built.prepared.harness);
      await sampler.calibrate();
      const ms = await sampler.sample();
      expect(Number.isFinite(ms)).toBe(true);
      expect(ms).toBeGreaterThan(0);
      sampler.destroy();
    } finally {
      built.destroy();
    }
  });
}

// Cross-variant checks protect the comparison contract, in addition to each
// kernel's CPU reference: identical logical data, work accounting and results.
for (const family of ['workgroup', 'layout', 'tile', 'reduction', 'branch', 'atomic'] as const) {
  test(`${family} variants perform equivalent useful work on identical logical inputs`, async () => {
    let baseline: { input: number[]; output: number[]; amount: number; unit: string } | undefined;
    for (const spec of TECHNIQUE_SPECS.filter((candidate) => candidate.family === family)) {
      const built = await buildTechnique(ctx, spec, {}, 777);
      try {
        if (built.prepared.kind === 'skipped') continue;
        const canonical = (values: ArrayLike<number>) => {
          if (spec.family !== 'layout') return Array.from(values);
          return Array.from(
            { length: built.data.count * 8 },
            (_, index) => values[particleAddress(spec.variant, Math.floor(index / 8), index % 8, built.data.padded)]!,
          );
        };
        const input = canonical(built.data.input);
        const output = canonical(await execute(built, 2));
        const amount = built.prepared.meta.amountPerOp;
        const unit = TECHNIQUE_BENCHMARKS.find((b) => b.id === techniqueId(spec))!.metric.unit;
        if (!baseline) {
          baseline = { input, output, amount, unit };
          continue;
        }
        expect(input).toEqual(baseline.input);
        expect(amount).toBe(baseline.amount);
        expect(unit).toBe(baseline.unit);
        expect(output.length).toBe(baseline.output.length);
        let maxError = 0;
        for (let i = 0; i < output.length; i++) {
          maxError = Math.max(maxError, Math.abs(output[i]! - baseline.output[i]!));
        }
        if (built.data.integer) expect(maxError).toBe(0);
        else expect(maxError).toBeLessThan(1e-6);
      } finally {
        built.destroy();
      }
    }
    expect(baseline).toBeDefined();
  });
}
