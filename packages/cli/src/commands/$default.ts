import { randomUUID } from 'node:crypto';
import { createRequire } from 'node:module';
import os from 'node:os';
import * as Sentry from '@sentry/node';
import chalk from 'chalk';
import { create, globals } from 'webgpu';
import { BENCHMARKS, runSuite, SuiteProgress, type BenchmarkResult, type DeviceInfo } from 'webgpu-bench';
import { defineCommand } from 'yargs-file-commands';
import { formatTable, globToRegExp } from '../format.ts';

const { version } = createRequire(import.meta.url)('../../package.json') as { version: string };

// web3dsurvey's expected data version — keep in sync with
// web3dsurvey/packages/shared/src/benchDataVersion.ts (bumped when what's measured changes).
const BENCH_DATA_VERSION = 30;
const SITE = 'https://web3dsurvey.com';
const BAR_WIDTH = 40;

const PLATFORM_NAMES: Record<string, string> = { Darwin: 'Mac OS', Windows_NT: 'Windows', Linux: 'Linux' };
const PLATFORM = PLATFORM_NAMES[os.type()] ?? os.type();
// Parsed server-side by web3dsurvey's bench handler — keep the shape in sync with its CLI_USER_AGENT regex.
// ponytail: os.release() is the kernel version (Darwin 27.0.0), not the marketing one; good enough to group by.
const USER_AGENT = `webgpu-bench-cli/${version} (${PLATFORM} ${os.release()}; ${os.arch()}; Node.js ${process.versions.node})`;

const KEY_LIMITS = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxComputeWorkgroupSizeX',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupStorageSize',
];

function formatDeviceInfo(info: DeviceInfo): string {
  const row = (k: string, v: string) => `${chalk.dim(k.padEnd(14))}${v}`;
  const yesNo = (b: boolean) => (b ? chalk.green('yes') : chalk.yellow('no'));
  return [
    row('Vendor', chalk.bold(info.vendor ?? 'unknown')),
    row('Architecture', chalk.bold(info.architecture ?? 'unknown')),
    row('Description', info.description ?? 'unknown'),
    row('Features', info.features.join(', ') || 'none'),
    row('Limits', KEY_LIMITS.map((k) => `${k}=${info.limits[k] ?? '?'}`).join(', ')),
    row(
      'Supports',
      `f16 ${yesNo(info.supportsF16)}, packed i8 dot ${yesNo(info.supportsI8Dot)}, GPU timestamps ${yesNo(info.supportsTimestampQuery)}`,
    ),
    row('Host', `${PLATFORM} ${os.release()} ${os.arch()}, Node.js ${process.versions.node}, Dawn`),
  ].join('\n');
}

async function report(
  apiHost: string,
  info: DeviceInfo | undefined,
  results: BenchmarkResult[],
): Promise<string | null> {
  const id = randomUUID();
  const body = {
    id,
    version: BENCH_DATA_VERSION,
    adapterInfo: info && { vendor: info.vendor, architecture: info.architecture, description: info.description },
    results: results.filter((r) => r.status === 'ok').map((r) => ({ id: r.id, metricValue: r.metricValue })),
  };
  const response = await fetch(`${apiHost}/api/bench?shareable=true`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'User-Agent': USER_AGENT },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(20_000),
  });
  // The route's only success response is 204; a 200 is the API politely ignoring us as a bot.
  if (response.status !== 204) throw new Error(`Report rejected: HTTP ${response.status} ${await response.text()}`);
  return id;
}

export const command = defineCommand({
  describe: "Benchmark this machine's GPU via WebGPU (Dawn), no browser needed",
  builder: (yargs) =>
    yargs
      .option('filter', {
        type: 'string',
        describe: 'Only run benchmarks whose id matches this glob, e.g. "f16-*" or "read-*"',
      })
      .option('report', {
        type: 'boolean',
        default: true,
        describe: `Submit results to ${SITE} (--no-report to keep them local)`,
      })
      .option('json', { type: 'boolean', default: false, describe: 'Print results as JSON instead of a table' })
      .option('api-host', { type: 'string', default: 'https://api.web3dsurvey.com', hidden: true }),
  handler: async (argv) => {
    const benchmarks = argv.filter ? BENCHMARKS.filter((b) => globToRegExp(argv.filter!).test(b.id)) : BENCHMARKS;
    if (benchmarks.length === 0) {
      throw new Error(`No benchmarks match "${argv.filter}". Ids: ${BENCHMARKS.map((b) => b.id).join(', ')}`);
    }

    // Dawn's node binding: GPUBufferUsage & co. become globals, navigator.gpu is
    // defined onto Node's built-in (getter-only) navigator.
    Object.assign(globalThis, globals);
    Object.defineProperty(navigator, 'gpu', { value: create([]), configurable: true });

    console.error(`${chalk.bold('webgpu-bench-cli')} ${chalk.dim(`v${version}`)}`);

    const progress = new SuiteProgress(benchmarks.length);
    const results = new Map<string, BenchmarkResult>();
    let info: DeviceInfo | undefined;
    let lastPercent = -1;
    const tty = process.stderr.isTTY;
    const bar = (fraction: number) => {
      const filled = Math.round(fraction * BAR_WIDTH);
      return `[${chalk.green('█'.repeat(filled))}${chalk.dim('░'.repeat(BAR_WIDTH - filled))}]`;
    };
    const onDeviceInfo = (i: DeviceInfo) => {
      info = i;
      if (!argv.json) console.log(`${formatDeviceInfo(i)}\n`);
      console.error(`Running ${benchmarks.length} benchmark${benchmarks.length === 1 ? '' : 's'}…`);
    };
    for await (const r of runSuite({ benchmarks, onDeviceInfo, onProgress: progress.onProgress })) {
      results.set(r.id, r);
      progress.onResult(r);
      const percent = Math.floor(progress.fraction * 100);
      // A TTY redraws one line; a log gets a line every 10%.
      if (percent !== lastPercent && (tty || percent % 10 === 0)) {
        lastPercent = percent;
        const eta = progress.remainingSeconds;
        const line = `${bar(progress.fraction)} ${String(percent).padStart(3)}%${eta ? ` (${eta}s remaining)` : ''}`;
        process.stderr.write(tty ? `\r${line}\x1b[K` : `${line}\n`);
      }
    }
    progress.finish();
    process.stderr.write(tty ? `\r${bar(1)} 100%\x1b[K\n\n` : '\n');

    const rows = [...results.values()].toSorted((a, b) => a.id.localeCompare(b.id));
    if (argv.json) {
      console.log(JSON.stringify({ device: info, results: rows }, null, 2));
    } else {
      console.log(formatTable(rows, benchmarks));
    }

    if (argv.report) {
      // A failed submission is the maintainer's problem, not the user's: it goes to Sentry, not the terminal.
      const id = await report(argv.apiHost, info, rows).catch((error: unknown) => {
        Sentry.captureException(error);
        return null;
      });
      if (id) console.log(`\nReported to ${chalk.underline(`${SITE}/benchmark/${id}`)}`);
    }
  },
});
