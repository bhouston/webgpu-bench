import { createRequire } from 'node:module';
import { infoFromPackageJson } from '@clidoc/core';
import { createDocgenCommand, fromYargs } from '@clidoc/yargs';
import type { CommandModule } from 'yargs';
import { defineCommand } from 'yargs-file-commands';
import { command as bench } from './$default.ts';

const pkg = createRequire(import.meta.url)('../../package.json') as {
  name: string;
  version: string;
  description: string;
};
const info = infoFromPackageJson(pkg);

// $default.ts has no `command` field of its own; `$0` is Yargs' token for the default command.
let document: ReturnType<typeof fromYargs>;
export const command: CommandModule = defineCommand(createDocgenCommand(() => document));
document = fromYargs([{ ...bench, command: '$0' }, command], info);
