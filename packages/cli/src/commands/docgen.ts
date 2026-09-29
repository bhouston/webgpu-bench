import { createRequire } from 'node:module';
import { infoFromPackageJson } from '@clidoc/core';
import { createDocgenCommand, fromYargs } from '@clidoc/yargs';
import { command as defaultCommand } from './$default.ts';

const pkg = createRequire(import.meta.url)('../../package.json');

export const command: ReturnType<typeof createDocgenCommand> = createDocgenCommand(() =>
  fromYargs([defaultCommand, command], infoFromPackageJson(pkg)),
);
