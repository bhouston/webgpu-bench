import type { BenchmarkResult, DeviceInfo } from '@webgpu-profiler/performance-suite';
import { useCallback, useEffect, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { DeviceInfoCard } from '@/components/DeviceInfoCard';
import { ResultsTable } from '@/components/ResultsTable';
import { collectEnvironmentInfo, type EnvironmentInfo } from '@/lib/environment';

type RunState = 'idle' | 'running' | 'done' | 'error';

const WEBGPU_UNAVAILABLE = typeof navigator === 'undefined' || !('gpu' in navigator);

export function BenchmarkSuiteApp() {
  const [state, setState] = useState<RunState>('idle');
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [env, setEnv] = useState<EnvironmentInfo | null>(null);

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
        <h1 className="text-2xl font-semibold tracking-tight">WebGPU Bandwidth &amp; FLOPS Profiler</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Measures this device's raw WebGPU ceilings: read and write memory bandwidth, and fp32/fp16/int8 FLOPS at
          scalar, vec4, mat4, and register-resident-matvec granularity (plus the packed int8 dot-product extension).
          Every kernel isolates one resource — memory or ALU — with a calibrated warmup, adaptive sampling (3–10 timed
          measurements, stopping once the timings converge), and (where the device supports it) GPU-side timestamp-query
          timing rather than CPU wall-clock.
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

      <DeviceInfoCard env={env} info={deviceInfo} />

      {results.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Results</CardTitle>
            <CardDescription>
              Lower median time is better. Throughput is computed from the median of the timed runs.
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
