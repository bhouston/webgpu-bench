import { createServer, type Server } from 'node:http';
import { afterAll, beforeAll, expect, test } from 'vitest';
import { commandLine, extendMatchers } from 'vitest-command-line';

extendMatchers();

// Runs the built CLI end to end (needs `tsc` first, and a GPU for the benchmark tests).
const cli = commandLine({ command: ['node', './dist/cli.js'], env: { FORCE_COLOR: '0' }, timeout: 60_000 });

let server: Server;
let apiHost: string;
const reports: unknown[] = [];
beforeAll(async () => {
  server = createServer((req, res) => {
    let body = '';
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => {
      reports.push({ url: req.url, userAgent: req.headers['user-agent'], body: JSON.parse(body) });
      res.writeHead(204).end();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  apiHost = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
});
afterAll(() => server.close());

test('--help lists the options', async () => {
  const result = await cli.run(['--help']);
  expect(result).toSucceed();
  expect(result).toHaveStdout(/--filter/);
  expect(result).toHaveStdout(/--no-report/);
});

test('--version prints the package version', async () => {
  const result = await cli.run(['--version']);
  expect(result).toSucceed();
  expect(result).toHaveStdout(/^\d+\.\d+\.\d+\n$/);
});

test('unknown flag fails', async () => {
  const result = await cli.run(['--bogus']);
  expect(result).toExitWith(1);
  expect(result).toHaveStderr(/Unknown argument: bogus/);
});

test('filter matching nothing fails and lists ids', async () => {
  const result = await cli.run(['--filter', 'nope', '--no-report']);
  expect(result).toExitWith(1);
  expect(result).toHaveStderr(/No benchmarks match "nope"\. Ids: read-linear/);
});

test('runs a filtered benchmark as a table', async () => {
  const result = await cli.run(['--filter', 'f32-div', '--no-report']);
  expect(result).toSucceed();
  expect(result).toHaveStdout(/^Vendor\s+\S/m);
  expect(result).toHaveStdout(/^f32-div\s+[\d.]+ [kMGT]?FLOP\/s$/m);
  expect(result).toHaveStderr(/Running 1 benchmark…/);
});

test('--json prints device info and results, and --report posts them', async () => {
  const result = await cli.run(['--filter', 'f32-div', '--json', '--api-host', apiHost]);
  expect(result).toSucceed();
  const stdout = result.stdout.replace(/\nReported to .*\n$/, '');
  const { device, results } = JSON.parse(stdout);
  expect(device.limits.maxBufferSize).toBeGreaterThan(0);
  expect(results).toHaveLength(1);
  expect(results[0]).toMatchObject({ id: 'f32-div', status: 'ok' });
  expect(results[0].metricValue).toBeGreaterThan(0);

  expect(result).toHaveStdout(/Reported to https:\/\/web3dsurvey\.com\/benchmark\/[0-9a-f-]{36}\n$/);
  expect(reports).toHaveLength(1);
  expect(reports[0]).toMatchObject({
    url: '/api/bench?shareable=true',
    userAgent: expect.stringMatching(/^webgpu-bench-cli\/\d+\.\d+\.\d+ \(/),
    body: {
      version: expect.any(Number),
      adapterInfo: { vendor: device.vendor },
      results: [{ id: 'f32-div', metricValue: results[0].metricValue }],
    },
  });
});
