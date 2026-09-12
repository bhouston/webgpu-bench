import chalk from 'chalk';
import type { BenchmarkResult, BenchmarkDefinition } from 'webgpu-bench';

/** `f16-*` -> /^f16-.*$/ ; `*` matches anything, `?` one character. */
export function globToRegExp(glob: string): RegExp {
  const escaped = glob
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*/g, '.*')
    .replace(/\?/g, '.');
  return new RegExp(`^${escaped}$`);
}

const SI = ['', 'k', 'M', 'G', 'T', 'P'];

/** 1.544e12, 'FLOP' -> '1.54 TFLOP/s' */
export function formatRate(value: number, unit: string): string {
  let i = 0;
  while (value >= 1000 && i < SI.length - 1) {
    value /= 1000;
    i++;
  }
  return `${value.toFixed(2)} ${SI[i]}${unit}/s`;
}

/** One line per benchmark: id, then the headline rate — or, for a skipped/failed row, why. */
export function formatTable(results: BenchmarkResult[], defs: readonly BenchmarkDefinition[]): string {
  const width = Math.max(...results.map((r) => r.id.length));
  return results
    .map((r) => {
      const unit = defs.find((d) => d.id === r.id)?.metric.unit ?? '';
      const value =
        r.status === 'ok' && r.metricValue !== undefined
          ? chalk.green(formatRate(r.metricValue, unit))
          : r.status === 'skipped'
            ? chalk.yellow(`skipped: ${r.message ?? ''}`)
            : chalk.red(`${r.status}: ${r.message ?? ''}`);
      return `${chalk.cyan(r.id.padEnd(width))}  ${value}`;
    })
    .join('\n');
}
