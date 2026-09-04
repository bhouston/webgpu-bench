import type { BenchmarkResult, DeviceInfo } from '@webgpu-profiler/performance-suite';
import { useCallback, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { DeviceInfoCard } from '@/components/DeviceInfoCard';
import { ResultsTable } from '@/components/ResultsTable';

type RunState = 'idle' | 'running' | 'done' | 'error';

const WEBGPU_UNAVAILABLE = typeof navigator === 'undefined' || !('gpu' in navigator);

export function BenchmarkSuiteApp() {
  const [state, setState] = useState<RunState>('idle');
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const run = useCallback(async () => {
    setState('running');
    setResults([]);
    setDeviceInfo(null);
    setErrorMessage(null);
    try {
      // Loaded lazily so `navigator.gpu`/WebGPU types are only touched client-side.
      const { runSuite } = await import('@webgpu-profiler/performance-suite');
      for await (const result of runSuite({ onDeviceInfo: setDeviceInfo })) {
        setResults((prev) => {
          const next = prev.filter((r) => r.id !== result.id);
          next.push(result);
          return next;
        });
      }
      setState('done');
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
      setState('error');
    }
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 py-10">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight">WebGPU MatVec Profiler</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Benchmarks compute-shader strategies for the operation that dominates LLM inference: large
          matrix &times; vector multiplication. Every kernel runs against byte-identical, deterministically
          generated data, with a calibrated warmup, 10 timed measurements, and (where the device supports it)
          GPU-side timestamp-query timing rather than CPU wall-clock.
        </p>
      </div>

      <div className="flex items-center gap-3">
        <Button onClick={() => void run()} disabled={state === 'running' || WEBGPU_UNAVAILABLE}>
          {state === 'running' ? 'Running…' : 'Run benchmark suite'}
        </Button>
        {state === 'running' ? (
          <span className="text-sm text-muted-foreground">
            {results.length} benchmark{results.length === 1 ? '' : 's'} complete…
          </span>
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

      {deviceInfo ? <DeviceInfoCard info={deviceInfo} /> : null}

      {results.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Results</CardTitle>
            <CardDescription>
              Lower mean time is better. "Speedup" is relative to the fastest completed benchmark so far.
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
