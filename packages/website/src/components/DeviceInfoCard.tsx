import type { DeviceInfo } from '@webgpu-profiler/performance-suite';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function DeviceInfoCard({ info }: { info: DeviceInfo }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">GPU device</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div className="text-muted-foreground">
          {info.vendor ?? 'Unknown vendor'} {info.architecture ? `· ${info.architecture}` : ''}
        </div>
        {info.description ? <div className="text-xs text-muted-foreground">{info.description}</div> : null}
        <div className="flex flex-wrap gap-2 pt-1">
          <Badge variant={info.supportsF16 ? 'success' : 'outline'}>shader-f16 {info.supportsF16 ? 'yes' : 'no'}</Badge>
          <Badge variant={info.supportsI8Dot ? 'success' : 'outline'}>
            packed_4x8_integer_dot_product {info.supportsI8Dot ? 'yes' : 'no'}
          </Badge>
          <Badge variant={info.supportsTimestampQuery ? 'success' : 'outline'}>
            timestamp-query {info.supportsTimestampQuery ? 'yes' : 'no'}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}
