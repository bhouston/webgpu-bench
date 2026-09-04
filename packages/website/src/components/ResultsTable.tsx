import { humanizeUnit } from 'humanize-units';

import type { BenchmarkResult } from '@webgpu-profiler/performance-suite';

import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

function formatMs(value: number): string {
  // `value` is in ms; humanizeUnit works off the base (seconds) unit so it can
  // pick µs/ms/s as appropriate via the standard SI prefixes.
  return humanizeUnit(value / 1000, { postfix: 's', unitSeparator: ' ', significantDigits: 3 });
}

function formatStddevPct(mean: number, stddev: number): string {
  if (!Number.isFinite(mean) || mean <= 0) return '—';
  return `${((stddev / mean) * 100).toFixed(1)}%`;
}

function formatRuns(r: BenchmarkResult): string {
  if (r.status !== 'ok') return '—';
  // Sampling is adaptive: a row that ran to the cap without converging is
  // flagged so a noisy number isn't mistaken for a settled one.
  return r.stopReason === 'max-runs' ? `${r.timesMs.length} (noisy)` : String(r.timesMs.length);
}

function formatThroughput(value: number | undefined, unit: string): string {
  if (value === undefined || !Number.isFinite(value)) return '—';
  // `value` is already in giga-units (GFLOP/s, GB/s); rescale to the base
  // unit so humanizeUnit can re-derive the right SI prefix (G, T, ...).
  return humanizeUnit(value * 1e9, { postfix: `${unit}/s`, unitSeparator: ' ', significantDigits: 3 });
}

const CATEGORY_LABEL: Record<string, string> = {
  bandwidth: 'Bandwidth',
  compute: 'Raw compute',
};

function StatusBadge({ status }: { status: BenchmarkResult['status'] }) {
  switch (status) {
    case 'running':
      return <Badge variant="secondary">running…</Badge>;
    case 'ok':
      return <Badge variant="success">ok</Badge>;
    case 'skipped':
      return <Badge variant="warning">skipped</Badge>;
    case 'error':
      return <Badge variant="destructive">error</Badge>;
  }
}

export function ResultsTable({ results }: { results: BenchmarkResult[] }) {
  // A median of exactly 0 (or negative/NaN) isn't a real timing — it means
  // the GPU timer read back garbage (e.g. a dispatch that never ran, or
  // finished faster than the timestamp-query clock's resolution). Excluding
  // it here stops one bad reading from crowning itself "fastest".
  const fastestMedian = Math.min(
    ...results.filter((r) => r.status === 'ok' && r.stats && r.stats.median > 0).map((r) => r.stats!.median),
    Number.POSITIVE_INFINITY,
  );

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Benchmark</TableHead>
          <TableHead>Category</TableHead>
          <TableHead>Status</TableHead>
          <TableHead className="text-right">Median</TableHead>
          <TableHead className="text-right">StdDev %</TableHead>
          <TableHead className="text-right">Runs</TableHead>
          <TableHead className="text-right">Throughput</TableHead>
          <TableHead className="text-right">Bandwidth</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {results.map((r) => {
          const isFastest = r.status === 'ok' && r.stats?.median === fastestMedian;
          return (
            <TableRow key={r.id} className={isFastest ? 'bg-success/5' : undefined}>
              <TableCell className="font-medium">
                <div>{r.label}</div>
                {r.description ? <div className="text-xs text-muted-foreground">{r.description}</div> : null}
                {r.message ? <div className="text-xs text-muted-foreground italic">{r.message}</div> : null}
              </TableCell>
              <TableCell>{CATEGORY_LABEL[r.category] ?? r.category}</TableCell>
              <TableCell>
                <StatusBadge status={r.status} />
              </TableCell>
              <TableCell className="text-right tabular-nums">{r.stats ? formatMs(r.stats.median) : '—'}</TableCell>
              <TableCell className="text-right tabular-nums">
                {r.stats ? formatStddevPct(r.stats.mean, r.stats.stddev) : '—'}
              </TableCell>
              <TableCell
                className={`text-right tabular-nums ${r.stopReason === 'max-runs' ? 'text-warning' : ''}`}
                title={
                  r.stopReason === 'max-runs'
                    ? 'Timings did not converge before the run cap; treat with suspicion.'
                    : 'Timed measurements taken before the 95% confidence interval converged.'
                }
              >
                {formatRuns(r)}
              </TableCell>
              <TableCell className="text-right tabular-nums">{formatThroughput(r.gflops, 'FLOP')}</TableCell>
              <TableCell className="text-right tabular-nums">{formatThroughput(r.gbps, 'B')}</TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
