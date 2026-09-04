import { humanizeUnit } from 'humanize-units';
import { Info } from 'lucide-react';

import { BENCHMARK_CATALOG, type BenchmarkInfo, type BenchmarkResult } from 'webgpu-bench';

import { Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

const CATALOG_BY_ID = new Map(BENCHMARK_CATALOG.map((b) => [b.id, b]));

function formatThroughput(value: number | undefined, unit: string): string {
  if (value === undefined || !Number.isFinite(value)) return '—';
  return humanizeUnit(value, { postfix: `${unit}/s`, unitSeparator: ' ', significantDigits: 3 });
}

// Kernel labels were written when every compute benchmark was reported as
// "FLOPS", so some (the int8/conversion kernels, now OPS_METRIC) still say
// "FLOPS" in their label even though the metric column correctly shows
// GOP/s for them. Rather than hand-editing every label string, strip the
// stale unit word here and let the metric column's unit speak for itself.
function displayName(label: string): string {
  return label.replace(/\s*\b(?:FLOPS?|ops)\b/gi, '').replace(/\s{2,}/g, ' ').trim();
}

/**
 * What goes in the metric cell. Rows that can't produce a number say why in
 * the cell itself instead of a separate status column; rows still being
 * sampled show their best-so-far (dimmed) since the best only ever improves.
 */
function MetricCell({ r, info }: { r: BenchmarkResult; info: BenchmarkInfo }) {
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
      text = r.metricValue === undefined ? 'measuring…' : formatThroughput(r.metricValue, info.metric.unit);
      className += ' text-muted-foreground';
      break;
    case 'ok':
      text = formatThroughput(r.metricValue, info.metric.unit);
      if (r.stopReason === 'throttled') className += ' text-warning';
      break;
  }
  return (
    <TableCell className={className} title={r.status === 'ok' && r.stopReason === 'throttled' ? r.message : undefined}>
      {text}
    </TableCell>
  );
}

/** (i) button that pops up the benchmark's description and WGSL source in a shadcn dialog. */
function KernelInfoButton({ info }: { info: BenchmarkInfo }) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button
          type="button"
          aria-label={`About ${displayName(info.label)}`}
          className="text-muted-foreground hover:text-foreground"
        >
          <Info className="size-4" />
        </button>
      </DialogTrigger>
      <DialogContent>
        <DialogTitle>{displayName(info.label)}</DialogTitle>
        <DialogDescription className="break-words whitespace-pre-wrap">{info.description}</DialogDescription>
        <pre className="max-h-96 overflow-auto rounded bg-muted p-3 text-xs">
          <code>{info.source.trim()}</code>
        </pre>
      </DialogContent>
    </Dialog>
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
          <TableHead className="text-right">Metrics</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {results.map((r) => {
          const info = CATALOG_BY_ID.get(r.id);
          if (!info) return null;
          const isFastest = r.status === 'ok' && r.stats?.min === fastestBest;
          const note =
            r.status === 'ok' && r.stopReason === 'throttled'
              ? 'Thermally throttled; may understate the device.'
              : r.message;
          return (
            <TableRow key={r.id} className={isFastest ? 'bg-success/5' : undefined}>
              <TableCell className="font-medium">
                <div className="flex items-center gap-1.5">
                  <span>{displayName(info.label)}</span>
                  <KernelInfoButton info={info} />
                </div>
                {note ? <div className="text-xs text-muted-foreground italic">{note}</div> : null}
              </TableCell>
              <MetricCell r={r} info={info} />
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
