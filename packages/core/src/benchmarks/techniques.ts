import type { GpuContext } from '../gpu/context.ts';
import type { MetricDef } from '../types.ts';
import { createStorageBuffer, createUniformBuffer } from '../gpu/buffers.ts';
import { mulberry32 } from '../data/generate.ts';
import {
  createPipeline,
  skippedResult,
  type BenchmarkDefinition,
  type HarnessConfig,
  type PreparedBenchmark,
} from './common.ts';
import {
  workgroupWgsl,
  particleWgsl,
  tileWgsl,
  reductionWgsl,
  branchVec4Wgsl,
  atomicWgsl,
  dependentChainWgsl,
  type ParticleLayout,
  type TileStrategy,
  type ReductionStrategy,
  type AtomicStrategy,
} from '../shaders/techniques.ts';

export type TechniqueSpec =
  | { family: 'workgroup'; variant: 64 | 128 | 256 }
  | { family: 'layout'; variant: ParticleLayout }
  | { family: 'tile'; variant: TileStrategy }
  | { family: 'reduction'; variant: ReductionStrategy }
  | { family: 'branch'; variant: 'select' | 'if' }
  | { family: 'atomic'; variant: AtomicStrategy }
  | { family: 'dependent'; variant: 'chain' };

export const TECHNIQUE_SPECS: readonly TechniqueSpec[] = [
  { family: 'workgroup', variant: 64 },
  { family: 'workgroup', variant: 128 },
  { family: 'workgroup', variant: 256 },
  { family: 'layout', variant: 'aos' },
  { family: 'layout', variant: 'soa' },
  { family: 'layout', variant: 'aosoa' },
  { family: 'tile', variant: 'direct' },
  { family: 'tile', variant: 'shared' },
  { family: 'tile', variant: 'shared-padded' },
  { family: 'reduction', variant: 'serial' },
  { family: 'reduction', variant: 'workgroup' },
  { family: 'reduction', variant: 'subgroup' },
  { family: 'branch', variant: 'select' },
  { family: 'branch', variant: 'if' },
  { family: 'atomic', variant: 'direct' },
  { family: 'atomic', variant: 'workgroup' },
  { family: 'atomic', variant: 'sharded' },
  { family: 'dependent', variant: 'chain' },
];

export function techniqueId(spec: TechniqueSpec): string {
  return spec.family === 'branch'
    ? `branch-vec4-${spec.variant}`
    : spec.family === 'dependent'
      ? 'read-dependent-chain'
      : `${spec.family}-${spec.variant}`;
}

export function techniqueSource(spec: TechniqueSpec): string {
  switch (spec.family) {
    case 'workgroup':
      return workgroupWgsl(spec.variant);
    case 'layout':
      return particleWgsl(spec.variant);
    case 'tile':
      return tileWgsl(spec.variant);
    case 'reduction':
      return reductionWgsl(spec.variant);
    case 'branch':
      return branchVec4Wgsl(spec.variant === 'select');
    case 'atomic':
      return atomicWgsl(spec.variant);
    case 'dependent':
      return dependentChainWgsl;
  }
}

const metric = (key: string, unit: string, name: string): MetricDef => ({ key, unit, name });
const METRICS: Record<TechniqueSpec['family'], MetricDef> = {
  workgroup: metric('elements', 'element', 'Transformed elements'),
  layout: metric('particles', 'particle', 'Updated particles'),
  tile: metric('outputs', 'output', 'Stencil outputs'),
  reduction: metric('elements', 'element', 'Reduced input elements'),
  branch: metric('choices', 'choice', 'Component choices'),
  atomic: metric('updates', 'update', 'Histogram inputs'),
  dependent: metric('hops', 'hop', 'Dependent memory hops'),
};

function description(spec: TechniqueSpec): string {
  switch (spec.family) {
    case 'workgroup':
      return `65,536 scalar elements, one per invocation, with 16 multiply-add steps each. Only workgroup size (${spec.variant}) changes; the input, arithmetic and output count match. Reports elements/s.`;
    case 'layout':
      return `Identical position.xyz += velocity.xyz * 0.125 updates for 65,536 eight-field particles stored as ${spec.variant}. AoSoA uses 32-particle blocks. Reads immutable input and writes positions in the same layout; other fields are preserved. Reports particles/s, excluding CPU packing.`;
    case 'tile':
      return `The same 3x3 box stencil on 256 independent 16x16 tiles, wrapping within each tile. ${spec.variant === 'direct' ? 'Loads all nine neighbors directly from storage.' : `Cooperatively stages the tile in workgroup memory with row stride ${spec.variant === 'shared' ? 16 : 17}; includes loading and synchronization.`} Reports outputs/s. A fixed reuse pattern, not a reuse sweep.`;
    case 'reduction':
      return `Sums 65,536 bounded u32 inputs into 256 independent 256-element segment sums. ${spec.variant === 'serial' ? 'One invocation sums each segment serially.' : spec.variant === 'workgroup' ? 'One 256-invocation workgroup per segment uses a shared-memory tree.' : 'Uses subgroupAdd with one workgroup atomic addition per subgroup; requires subgroups, assumes no subgroup width.'} Reports input elements/s; all variants produce complete segment sums.`;
    case 'branch':
      return `65,536 vec4 values, 64 steps with the same seeded component masks and bounded arithmetic. ${spec.variant === 'select' ? 'One vector select per step.' : 'Four scalar if/else statements per step.'} Reports component choices/s. Measures source alternatives as compiled; does not guarantee native branches.`;
    case 'atomic':
      return `A complete 32-bin histogram of 65,536 identical seeded keys (about half target bin zero). ${spec.variant === 'direct' ? 'Direct global atomic updates.' : spec.variant === 'workgroup' ? 'Aggregates in workgroup-local atomics before global merging.' : 'Updates eight global shards, then sums them.'} Includes reset, 16 inputs per invocation, and final merge in every timed repetition. Reports histogram inputs/s.`;
    case 'dependent':
      return 'One active invocation follows one dependent chain for 4,096 hops through a seeded single-cycle permutation of 1,048,576 u32 entries (4 MiB). Every address comes from the preceding load. Reports hops/s; reciprocal rate is amortized ns/hop, including loop/dispatch cost, not raw memory latency. Repeated batches reuse the same chain.';
  }
}

/** CPU fixture construction is outside timing; smaller counts are for correctness tests. */
export function techniqueData(spec: TechniqueSpec, count = spec.family === 'dependent' ? 1_048_576 : 65_536) {
  if (!Number.isSafeInteger(count) || count < 1 || count > 1_048_576)
    throw new Error('Technique count must be an integer in [1, 1048576].');
  const padded = Math.ceil(count / 256) * 256;
  const random = mulberry32(0x5eed);
  const rounds =
    spec.family === 'branch' ? 64 : spec.family === 'workgroup' ? 16 : spec.family === 'dependent' ? 4096 : 1;
  const shards = spec.family === 'atomic' && spec.variant === 'sharded' ? 8 : 1;
  const integer = spec.family === 'atomic' || spec.family === 'reduction' || spec.family === 'dependent';
  let input: Float32Array | Uint32Array;
  let initialOutput: Float32Array | Uint32Array;
  if (spec.family === 'layout') {
    input = new Float32Array(padded * 8);
    for (let i = 0; i < count; i++)
      for (let field = 0; field < 8; field++) {
        input[particleAddress(spec.variant, i, field, padded)] = random() * 2 - 1;
      }
    initialOutput = input.slice();
  } else if (spec.family === 'dependent') {
    // Fisher-Yates order wired into one cycle: no short cycles or arithmetic
    // next-address formula in the shader. Start at zero on every repetition.
    const order = Uint32Array.from({ length: count }, (_, i) => i);
    for (let i = count - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      const old = order[i]!;
      order[i] = order[j]!;
      order[j] = old;
    }
    input = new Uint32Array(count);
    for (let i = 0; i < count; i++) input[order[i]!] = order[(i + 1) % count]!;
    initialOutput = new Uint32Array(1).fill(0xffffffff);
  } else if (integer) {
    input = new Uint32Array(padded);
    for (let i = 0; i < count; i++)
      input[i] =
        spec.family === 'atomic' ? (random() < 0.5 ? 0 : 1 + Math.floor(random() * 31)) : Math.floor(random() * 16);
    initialOutput = new Uint32Array(spec.family === 'atomic' ? 32 : padded / 256).fill(0xffffffff);
  } else {
    const width = spec.family === 'branch' ? 4 : 1;
    input = new Float32Array(padded * width);
    for (let i = 0; i < count * width; i++) input[i] = random() * 2 - 1;
    initialOutput = new Float32Array(count * width).fill(Number.NaN);
  }
  return { count, padded, rounds, shards, integer, input, initialOutput };
}

export function particleAddress(layout: ParticleLayout, i: number, field: number, padded: number): number {
  return layout === 'aos'
    ? i * 8 + field
    : layout === 'soa'
      ? field * padded + i
      : Math.floor(i / 32) * 256 + field * 32 + (i % 32);
}

/** Exposes the actual timed output/resources to Vitest, without benchmark readback. */
export async function buildTechnique(
  ctx: GpuContext,
  spec: TechniqueSpec,
  harness: HarnessConfig = {},
  count?: number,
) {
  const id = techniqueId(spec);
  const data = techniqueData(spec, count);
  const meta = {
    id,
    category: 'algorithm' as const,
    rows: data.count,
    cols: spec.family === 'reduction' ? 256 : data.rounds,
    amountPerOp:
      spec.family === 'dependent' ? data.rounds : data.count * (spec.family === 'branch' ? data.rounds * 4 : 1),
  };
  if (spec.family === 'reduction' && spec.variant === 'subgroup' && !ctx.device.features.has('subgroups')) {
    return {
      prepared: {
        kind: 'skipped',
        result: skippedResult(meta, 'Device/browser does not support the "subgroups" WebGPU feature.'),
      } as PreparedBenchmark,
      data,
      output: undefined,
      destroy() {},
    };
  }
  const resources: GPUBuffer[] = [];
  const keep = (buffer: GPUBuffer) => {
    resources.push(buffer);
    return buffer;
  };
  const destroy = () => {
    for (const buffer of resources) buffer.destroy();
  };
  try {
    const params = keep(
      createUniformBuffer(ctx.device, new Uint32Array([data.count, data.rounds, data.padded, data.shards])),
    );
    const input = keep(createStorageBuffer(ctx.device, data.input));
    const output = keep(createStorageBuffer(ctx.device, data.initialOutput));
    const source = techniqueSource(spec);
    const stages: { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; workgroups: number }[] = [];
    const addStage = async (entry: string, bindings: [number, GPUBuffer][], workgroups: number) => {
      const pipeline = await createPipeline(ctx.device, `${id}-${entry}`, source, undefined, entry);
      const bindGroup = ctx.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
      });
      stages.push({ pipeline, bindGroup, workgroups });
    };
    if (spec.family === 'atomic') {
      const scratch = keep(createStorageBuffer(ctx.device, new Uint32Array(data.shards * 32)));
      await addStage(
        'reset',
        [
          [0, params],
          [2, scratch],
        ],
        1,
      );
      await addStage(
        'main',
        [
          [0, params],
          [1, input],
          [2, scratch],
        ],
        Math.ceil(data.count / 4096),
      );
      await addStage(
        'merge',
        [
          [0, params],
          [2, scratch],
          [3, output],
        ],
        1,
      );
    } else {
      const workgroups =
        spec.family === 'dependent'
          ? 1
          : spec.family === 'reduction'
            ? spec.variant === 'serial'
              ? Math.ceil(Math.ceil(data.count / 256) / 64)
              : Math.ceil(data.count / 256)
            : Math.ceil(data.count / (spec.family === 'workgroup' ? spec.variant : spec.family === 'tile' ? 256 : 128));
      // The chain doesn't read count/padded, but does read the rounds uniform.
      await addStage(
        'main',
        [
          [0, params],
          [1, input],
          [2, output],
        ],
        workgroups,
      );
    }
    const prepared: PreparedBenchmark = {
      kind: 'kernel',
      meta,
      harness: {
        ...harness,
        device: ctx.device,
        useTimestamps: ctx.info.supportsTimestampQuery,
        // A fixed logical problem per repetition. Only batch repetitions are
        // calibrated, so paired variants never drift to different workloads.
        encode(pass, repetitions) {
          for (let i = 0; i < repetitions; i++)
            for (const stage of stages) {
              pass.setPipeline(stage.pipeline);
              pass.setBindGroup(0, stage.bindGroup);
              pass.dispatchWorkgroups(stage.workgroups);
            }
        },
      },
    };
    return { prepared, data, output, destroy };
  } catch (error) {
    destroy();
    throw error;
  }
}

export const TECHNIQUE_BENCHMARKS: readonly BenchmarkDefinition[] = TECHNIQUE_SPECS.map((spec) => ({
  id: techniqueId(spec),
  label: techniqueId(spec).replaceAll('-', ' '),
  description: description(spec),
  source: techniqueSource(spec),
  category: 'algorithm',
  metric: METRICS[spec.family],
  prepare: async ({ ctx, harness }) => (await buildTechnique(ctx, spec, harness)).prepared,
}));
