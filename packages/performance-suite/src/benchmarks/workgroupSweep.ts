import type { GpuContext } from '../gpu/context.ts';
import type { GeneratedData } from '../data/generate.ts';
import { benchmarkF32Vec4Shared } from './matvecF32Vec4Shared.ts';
import type { HarnessConfig } from './common.ts';
import type { BenchmarkResult } from '../types.ts';

const CANDIDATE_WG_SIZES = [32, 64, 128, 256];

/** Picks the workgroup sizes worth trying: powers of two (required by the tree reduction), capped by device limits. */
export function candidateWorkgroupSizes(ctx: GpuContext): number[] {
  const maxSize = ctx.info.limits['maxComputeWorkgroupSizeX'] ?? 256;
  const maxInvocations = ctx.info.limits['maxComputeInvocationsPerWorkgroup'] ?? 256;
  const cap = Math.min(maxSize, maxInvocations, 256);
  return CANDIDATE_WG_SIZES.filter((n) => n <= cap);
}

/**
 * Sweeps the tiled, workgroup-shared-memory kernel across a handful of
 * workgroup sizes so the suite reports which one this specific device
 * actually performs best at, rather than assuming a fixed size.
 */
export async function* runWorkgroupSweep(
  ctx: GpuContext,
  data: GeneratedData,
  harness: HarnessConfig = {},
): AsyncGenerator<BenchmarkResult> {
  for (const wgSize of candidateWorkgroupSizes(ctx)) {
    const result = await benchmarkF32Vec4Shared(ctx, data, wgSize, harness);
    yield { ...result, category: 'workgroup' as const };
  }
}
