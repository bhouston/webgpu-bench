import type { BenchmarkResult } from '@webgpu-profiler/performance-suite';

import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

function formatMs(value: number): string {
  if (value < 1) return `${(value * 1000).toFixed(1)} µs`;
  return `${value.toFixed(3)} ms`;
}

function formatThroughput(value: number | undefined, unit: string): string {
  if (value === undefined || !Number.isFinite(value)) return '—';
  if (value >= 1000) return `${(value / 1000).toFixed(2)} T${unit}/s`;
  return `${value.toFixed(1)} G${unit}/s`;
}

const CATEGORY_LABEL: Record<string, string> = {
  bandwidth: 'Bandwidth',
  dtype: 'Data type',
  layout: 'Memory layout',
  workgroup: 'Workgroup size',
  multilayer: 'Multi-layer MLP',
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
  const fastestMean = Math.min(
    ...results.filter((r) => r.status === 'ok' && r.stats).map((r) => r.stats!.mean),
    Number.POSITIVE_INFINITY,
  );

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Benchmark</TableHead>
          <TableHead>Category</TableHead>
          <TableHead>Status</TableHead>
          <TableHead className="text-right">Mean</TableHead>
          <TableHead className="text-right">Min</TableHead>
          <TableHead className="text-right">Median</TableHead>
          <TableHead className="text-right">Max</TableHead>
          <TableHead className="text-right">StdDev</TableHead>
          <TableHead className="text-right">Throughput</TableHead>
          <TableHead className="text-right">Bandwidth</TableHead>
          <TableHead className="text-right">Speedup</TableHead>
          <TableHead>Timing</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {results.map((r) => {
          const isFastest = r.status === 'ok' && r.stats?.mean === fastestMean;
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
              <TableCell className="text-right tabular-nums">{r.stats ? formatMs(r.stats.mean) : '—'}</TableCell>
              <TableCell className="text-right tabular-nums">{r.stats ? formatMs(r.stats.min) : '—'}</TableCell>
              <TableCell className="text-right tabular-nums">{r.stats ? formatMs(r.stats.median) : '—'}</TableCell>
              <TableCell className="text-right tabular-nums">{r.stats ? formatMs(r.stats.max) : '—'}</TableCell>
              <TableCell className="text-right tabular-nums">{r.stats ? formatMs(r.stats.stddev) : '—'}</TableCell>
              <TableCell className="text-right tabular-nums">{formatThroughput(r.gflops, 'FLOP')}</TableCell>
              <TableCell className="text-right tabular-nums">{formatThroughput(r.gbps, 'B')}</TableCell>
              <TableCell className="text-right tabular-nums">
                {r.status === 'ok' && r.stats && Number.isFinite(fastestMean)
                  ? `${(r.stats.mean / fastestMean).toFixed(2)}x`
                  : '—'}
              </TableCell>
              <TableCell className="text-xs text-muted-foreground">{r.timingMethod === 'gpu-timestamp' ? 'GPU' : 'CPU'}</TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
