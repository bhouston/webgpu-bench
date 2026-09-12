import { createRequire } from 'node:module';
import * as Sentry from '@sentry/node';

const { version } = createRequire(import.meta.url)('../package.json') as { version: string };

// Imported first from cli.ts so the SDK is initialized before anything else loads.
Sentry.init({
  dsn: 'https://1ec32947fec6c989458384059a9133cf@o4508898407481344.ingest.us.sentry.io/4512070547734528',
  release: `webgpu-bench@${version}`,
});
