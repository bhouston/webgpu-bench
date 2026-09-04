import { humanizeUnit } from 'humanize-units';

import type { BenchmarkResult } from '@webgpu-profiler/performance-suite';

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

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

/**
 * What goes in a value cell. Rows that can't produce a number say why in the
 * cell itself instead of a separate status column; rows still being sampled
 * show their best-so-far (dimmed) since the best only ever improves.
 */
function ValueCell({ r, value, unit }: { r: BenchmarkResult; value: number | undefined; unit: string }) {
  let text: string;
  let className = 'text-right tabular-nums';
  switch (r.status) {
    case 'error':
      text = 'error';
      className += ' text-destructive';
      break;
    case 'skipped':
      text = 'skipped';
      className += ' text-muted-foreground';
      break;
    case 'running':
      text = value === undefined ? 'measuring…' : formatThroughput(value, unit);
      className += ' text-muted-foreground';
      break;
    case 'ok':
      text = formatThroughput(value, unit);
      if (r.stopReason === 'throttled') className += ' text-warning';
      break;
  }
  return (
    <TableCell className={className} title={r.status === 'ok' && r.stopReason === 'throttled' ? r.message : undefined}>
      {text}
    </TableCell>
  );
}

export function ResultsTable({ results }: { results: BenchmarkResult[] }) {
  // Highlight the fastest finished benchmark (by best run). A best of
  // exactly 0 isn't a real timing, so it can't crown itself.
  const fastestBest = Math.min(
    ...results.filter((r) => r.status === 'ok' && r.stats && r.stats.min > 0).map((r) => r.stats!.min),
    Number.POSITIVE_INFINITY,
  );

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Benchmark</TableHead>
          <TableHead>Category</TableHead>
          <TableHead className="text-right">Best Throughput</TableHead>
          <TableHead className="text-right">Best Bandwidth</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {results.map((r) => {
          const isFastest = r.status === 'ok' && r.stats?.min === fastestBest;
          const note =
            r.status === 'ok' && r.stopReason === 'throttled'
              ? 'Thermally throttled; may understate the device.'
              : r.message;
          return (
            <TableRow key={r.id} className={isFastest ? 'bg-success/5' : undefined}>
              <TableCell className="font-medium">
                <div>{r.label}</div>
                {r.description ? <div className="text-xs text-muted-foreground">{r.description}</div> : null}
                {note ? <div className="text-xs text-muted-foreground italic">{note}</div> : null}
              </TableCell>
              <TableCell>{CATEGORY_LABEL[r.category] ?? r.category}</TableCell>
              <ValueCell r={r} value={r.gflops} unit="FLOP" />
              <ValueCell r={r} value={r.gbps} unit="B" />
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
