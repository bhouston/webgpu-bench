import type { BenchmarkResult, DeviceInfo, SuiteProgressEvent } from 'webgpu-bench';
import { useCallback, useEffect, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { DeviceInfoCard } from '@/components/DeviceInfoCard';
import { ResultsTable } from '@/components/ResultsTable';
import { collectEnvironmentInfo, type EnvironmentInfo } from '@/lib/environment';

type RunState = 'idle' | 'running' | 'done' | 'error';

const WEBGPU_UNAVAILABLE = typeof navigator === 'undefined' || !('gpu' in navigator);

/** Table repaints are debounced to at most one per this many ms while the suite streams updates. */
const UPDATE_INTERVAL_MS = 1000;

function describeProgress(results: BenchmarkResult[], progress: SuiteProgressEvent | null): string {
  const finished = results.filter((r) => r.status !== 'running').length;
  const base = `${finished} of ${results.length} benchmark${results.length === 1 ? '' : 's'} settled`;
  switch (progress?.type) {
    case 'cooldown':
      return `${base} — device is throttling, cooling down for ${(progress.ms / 1000).toFixed(0)}s (${progress.attempt}/${progress.maxAttempts})…`;
    case 'round':
      return `${base} — sampling round ${progress.round}…`;
    default:
      return `${base}…`;
  }
}

export function BenchmarkSuiteApp() {
  const [state, setState] = useState<RunState>('idle');
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [env, setEnv] = useState<EnvironmentInfo | null>(null);
  const [progress, setProgress] = useState<SuiteProgressEvent | null>(null);
  const pendingRows = useRef(new Map<string, BenchmarkResult>());
  const flushTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastFlush = useRef(0);

  // Gather what the browser reveals about this machine as soon as the page
  // mounts (client-side only), so the card is useful before a run starts.
  useEffect(() => {
    let cancelled = false;
    void collectEnvironmentInfo().then((e) => {
      if (!cancelled) setEnv(e);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const run = useCallback(async () => {
    setState('running');
    setResults([]);
    setDeviceInfo(null);
    setErrorMessage(null);
    setProgress(null);
    pendingRows.current.clear();
    lastFlush.current = 0;
    try {
      // Loaded lazily so `navigator.gpu`/WebGPU types are only touched client-side.
      const { runSuite } = await import('webgpu-bench');
      // Rows arrive once per change (setup, every measurement, finish) —
      // several times a second while sampling. Coalesce them and repaint at
      // most once per second so the table updates live without thrashing.
      const flush = () => {
        if (pendingRows.current.size === 0) return;
        const batch = [...pendingRows.current.values()];
        pendingRows.current.clear();
        lastFlush.current = performance.now();
        setResults((prev) => {
          const next = prev.slice();
          for (const row of batch) {
            const i = next.findIndex((r) => r.id === row.id);
            if (i === -1) next.push(row);
            else next[i] = row;
          }
          return next;
        });
      };
      for await (const result of runSuite({ onDeviceInfo: setDeviceInfo, onProgress: setProgress })) {
        pendingRows.current.set(result.id, result);
        const elapsed = performance.now() - lastFlush.current;
        if (elapsed >= UPDATE_INTERVAL_MS) {
          if (flushTimer.current !== null) {
            clearTimeout(flushTimer.current);
            flushTimer.current = null;
          }
          flush();
        } else if (flushTimer.current === null) {
          flushTimer.current = setTimeout(() => {
            flushTimer.current = null;
            flush();
          }, UPDATE_INTERVAL_MS - elapsed);
        }
      }
      if (flushTimer.current !== null) {
        clearTimeout(flushTimer.current);
        flushTimer.current = null;
      }
      flush();
      setState('done');
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
      setState('error');
    }
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 py-10">
      <p className="max-w-3xl text-sm text-muted-foreground">
        Measures this device's raw WebGPU ceilings: memory bandwidth and fp32/fp16/int8 FLOPS. Each kernel's{' '}
        <em>best</em> run is reported; throttled runs are discarded.
      </p>

      <div className="flex items-center gap-3">
        <Button onClick={() => void run()} disabled={state === 'running' || WEBGPU_UNAVAILABLE}>
          {state === 'running' ? 'Running…' : 'Run benchmark suite'}
        </Button>
        {state === 'running' ? (
          <span className="text-sm text-muted-foreground">{describeProgress(results, progress)}</span>
        ) : null}
      </div>

      {WEBGPU_UNAVAILABLE ? (
        <Card>
          <CardContent className="pt-6 text-sm text-destructive">
            WebGPU is not available in this browser. Try a recent Chrome/Edge (desktop) with WebGPU enabled.
          </CardContent>
        </Card>
      ) : null}

      {errorMessage ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base text-destructive">Suite failed to start</CardTitle>
            <CardDescription>{errorMessage}</CardDescription>
          </CardHeader>
        </Card>
      ) : null}

      <DeviceInfoCard env={env} info={deviceInfo} />

      {results.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Results</CardTitle>
            <CardDescription>
              Best run per kernel; numbers update live as sampling continues and only ever improve.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResultsTable results={results} />
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
