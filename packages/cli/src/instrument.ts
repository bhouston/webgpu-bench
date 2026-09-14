import { createRequire } from 'node:module';
import * as Sentry from '@sentry/node';

const { version } = createRequire(import.meta.url)('../package.json') as { version: string };

// Local dev runs the same built CLI a published user would, so NODE_ENV can't tell them apart.
// node_modules in the module path means it was installed (npm/npx); a repo checkout means dev.
const isInstalled = import.meta.url.includes('node_modules');

// Imported first from cli.ts so the SDK is initialized before anything else loads.
Sentry.init({
  dsn: 'https://1ec32947fec6c989458384059a9133cf@o4508898407481344.ingest.us.sentry.io/4512070547734528',
  release: `webgpu-bench@${version}`,
  environment: process.env.SENTRY_ENVIRONMENT ?? (isInstalled ? 'production' : 'development'),
});
