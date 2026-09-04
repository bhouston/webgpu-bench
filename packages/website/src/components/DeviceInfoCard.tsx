import type { DeviceInfo } from '@webgpu-profiler/performance-suite';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import type { EnvironmentInfo } from '@/lib/environment';

function Row({ label, value, hint }: { label: string; value: string | undefined; hint?: string }) {
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-x-4 gap-y-0.5">
      <div className="text-muted-foreground">{label}</div>
      <div>
        <span>{value ?? '—'}</span>
        {hint ? <span className="ml-2 text-xs text-muted-foreground">{hint}</span> : null}
      </div>
    </div>
  );
}

function joinDefined(parts: Array<string | undefined>, sep = ' '): string | undefined {
  const present = parts.filter((p): p is string => Boolean(p));
  return present.length > 0 ? present.join(sep) : undefined;
}

/**
 * Shows what the browser reveals about this machine (OS, browser, CPU,
 * GPU via WebGL) alongside the WebGPU adapter the suite runs on. `env` is
 * gathered on page load; `info` arrives once the suite has acquired its
 * device and adds the feature badges.
 */
export function DeviceInfoCard({ env, info }: { env: EnvironmentInfo | null; info: DeviceInfo | null }) {
  const webgpuVendor = info?.vendor ?? env?.webgpuVendor;
  const webgpuArchitecture = info?.architecture ?? env?.webgpuArchitecture;
  const webgpuDescription = info?.description ?? env?.webgpuDescription ?? env?.webgpuDevice;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">This machine</CardTitle>
        <CardDescription>
          Read from standard browser APIs (Client Hints, Navigator, WebGL, WebGPU) — nothing is sent anywhere.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-1 text-sm">
        {env ? (
          <>
            <Row
              label="OS"
              value={joinDefined([env.os, env.osVersion])}
              hint={
                env.osSource === 'user-agent'
                  ? 'from the UA string, which modern browsers freeze/generalize'
                  : undefined
              }
            />
            <Row label="Browser" value={joinDefined([env.browser, env.browserVersion])} />
            <Row
              label="CPU"
              value={joinDefined(
                [env.cpuArchitecture, env.cpuThreads !== undefined ? `${env.cpuThreads} logical threads` : undefined],
                ', ',
              )}
              hint={
                env.memoryGb !== undefined
                  ? `~${env.memoryGb} GB RAM reported (coarse, capped by the browser)`
                  : undefined
              }
            />
            <Row
              label="CPU / GPU (WebGL)"
              value={env.gpuModel ?? env.webglRenderer}
              hint={env.gpuModel && env.webglRenderer !== env.gpuModel ? env.webglRenderer : undefined}
            />
          </>
        ) : (
          <div className="text-muted-foreground">Detecting…</div>
        )}
        <Row
          label="WebGPU adapter"
          value={joinDefined([webgpuVendor, webgpuArchitecture], ' · ')}
          hint={webgpuDescription}
        />
        {info ? (
          <div className="flex flex-wrap gap-2 pt-2">
            <Badge variant={info.supportsF16 ? 'success' : 'outline'}>
              shader-f16 {info.supportsF16 ? 'yes' : 'no'}
            </Badge>
            <Badge variant={info.supportsI8Dot ? 'success' : 'outline'}>
              packed_4x8_integer_dot_product {info.supportsI8Dot ? 'yes' : 'no'}
            </Badge>
            <Badge variant={info.supportsTimestampQuery ? 'success' : 'outline'}>
              timestamp-query {info.supportsTimestampQuery ? 'yes' : 'no'}
            </Badge>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
