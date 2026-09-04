import type { Stats } from './types.ts';

export function computeStats(values: readonly number[]): Stats {
  if (values.length === 0) {
    throw new Error('computeStats: values must be non-empty');
  }
  const sorted = values.toSorted((a, b) => a - b);
  const sum = values.reduce((a, b) => a + b, 0);
  const mean = sum / values.length;
  const min = sorted[0]!;
  const max = sorted[sorted.length - 1]!;
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
  const variance =
    values.length > 1
      ? values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (values.length - 1)
      : 0;
  const stddev = Math.sqrt(variance);
  return { mean, min, max, median, stddev };
}
