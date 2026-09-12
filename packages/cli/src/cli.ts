#!/usr/bin/env node
import './instrument.ts';

import * as Sentry from '@sentry/node';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { fileCommands } from 'yargs-file-commands';

const { version } = createRequire(import.meta.url)('../package.json') as { version: string };
const commandsDir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'commands');

try {
  await yargs(hideBin(process.argv))
    .scriptName('webgpu-bench')
    .version(version)
    .command(await fileCommands({ commandDirs: [commandsDir] }))
    .strict()
    .fail(false)
    .wrap(100)
    .help().argv;
} catch (error) {
  Sentry.captureException(error);
  await Sentry.flush(2000);
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}
