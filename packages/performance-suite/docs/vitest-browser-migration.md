# Migrating the WebGPU test harness to Vitest Browser Mode

## Why

Today, verifying a shader actually compiles/runs (as opposed to just
type-checking) means hand-rolling a CDP session against headless Chrome:
bundle an entry point with esbuild, write it to an HTML file, drive Chrome
over the DevTools protocol with a ~40-line Node script, poll
`document.body.innerText` for a `DONE` marker. It works, but every detail is
a trap someone will hit again (`file://` blocks `type="module"` fetches with
a silent `net::ERR_FAILED`, so it must be bundled as IIFE; sandboxed Bash
kills the GPU process; missing `--enable-unsafe-webgpu` /
`--ignore-gpu-blocklist` fails silently). None of it is checked in — it lives
in scratch dirs and gets rebuilt from memory each session.

[Vitest Browser Mode](https://vitest.dev/guide/browser/) with the Playwright
provider replaces all of that with `import { test, expect } from 'vitest'`
running against a real Chromium tab, checked into the repo, runnable with
`pnpm test`, and debuggable in the Vitest UI. It's the standard tool for this
job now, so this is a "delete our infra, adopt the ecosystem one" move, not a
new abstraction.

## Scope

This only touches `packages/performance-suite`. `packages/website` is a
consumer and isn't part of this plan.

## Target end state

```
packages/performance-suite/
  vitest.config.browser.ts      # new — browser project, Playwright provider
  src/
    sampling.test.ts            # unchanged — pure logic, Node environment
    stats.test.ts               # unchanged — pure logic, Node environment
    shaders.browser.test.ts     # new — every kernel compiles & runs, no NaN/0
    suite.browser.test.ts       # new — runSuite() end-to-end smoke test
package.json
  "test": "vitest run"
  "test:browser": "vitest run --project browser"
```

Two Vitest "projects" in one config: the existing Node project for
`sampling.test.ts`/`stats.test.ts` (no GPU needed, keep them fast), and a new
`browser` project for anything that touches `navigator.gpu`. `pnpm test`
runs both; CI can run them as separate jobs if the browser one turns out
slower.

## Steps

### 1. Install

```
pnpm add -D -w vitest-browser
```

Actually two packages, both root devDependencies (Playwright's browser
binaries are shared across the workspace, no need to duplicate per-package):

```
pnpm add -D -w @vitest/browser playwright
pnpm exec playwright install chromium
```

`playwright install chromium` downloads the browser binary — needed once
locally and once per CI cache miss. Chromium only for now; Safari/WebKit
support is a separate concern (see "Safari" below).

### 2. Vitest config

`packages/performance-suite/vitest.config.ts` (or a `vitest.workspace.ts` at
the repo root if other packages grow browser tests later — not needed yet,
YAGNI until a second package needs it):

```ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'node',
          include: ['src/**/*.test.ts'],
          exclude: ['src/**/*.browser.test.ts'],
        },
      },
      {
        test: {
          name: 'browser',
          include: ['src/**/*.browser.test.ts'],
          browser: {
            enabled: true,
            provider: 'playwright',
            headless: true,
            instances: [
              {
                browser: 'chromium',
                launch: {
                  args: [
                    '--enable-unsafe-webgpu',
                    '--ignore-gpu-blocklist',
                  ],
                },
              },
            ],
          },
        },
      },
    ],
  },
});
```

This is the direct replacement for the CDP script's Chrome launch flags —
same two flags, now declared once in config instead of copy-pasted into
every throwaway script. `headless: true` replaces `--headless=new`; Vitest's
Playwright provider handles the rest of the launch (remote debugging port,
`file://` vs served page, waiting for the page) that the old harness had to
do by hand.

### 3. `navigator.gpu` availability in the browser project

The old harness discovered that `about:blank` has no `navigator.gpu` and the
page had to be `file://`. Vitest's browser mode serves test files over
`http://localhost` via its own dev server (Vite under the hood) rather than
`file://` or `about:blank`, which sidesteps both problems the old harness
worked around — no IIFE-vs-ESM `file://` CORS trap either, since it's a real
HTTP origin. Worth confirming `navigator.gpu` is present in that first test
(see smoke test below) before building anything else on top — if Chromium's
software/CPU WebGPU backend kicks in under Playwright instead of the real
GPU, that changes what these tests can assert (see "What these tests can
assert" below).

### 4. Smoke test first

Before porting any real coverage, get one trivial test green end to end —
this is where the environment surprises (if any) will show up, same as the
`net::ERR_FAILED` did last time:

```ts
// src/webgpu.browser.test.ts
import { test, expect } from 'vitest';

test('navigator.gpu is available', async () => {
  expect(navigator.gpu).toBeDefined();
  const adapter = await navigator.gpu.requestAdapter();
  expect(adapter).not.toBeNull();
});
```

Run with `pnpm --filter performance-suite exec vitest run --project
browser`. Once this passes, the rest is porting logic, not fighting the
environment.

### 5. Port shader-compiles-and-runs coverage

This is the direct replacement for the ad hoc `entry.ts` that imported
`runSuite` and printed a results table. Two tiers, mirroring what the old
harness actually checked:

**a. Every kernel compiles and produces a finite, non-zero result** — this
is the check that would have caught the `dot4I8Packed` `enable` vs
`requires` bug immediately instead of showing a silent 0s "ok" row. Iterate
every `prepare*` export (or drive it through `runSuite()` directly, which
exercises the real scheduling path) and assert each kernel's status is `ok`
or a *known* `skipped` reason (missing `shader-f16` etc.), never `error`,
and that `gflops`/`gbps` is finite and > 0 for anything that ran:

```ts
// src/suite.browser.test.ts
import { test, expect } from 'vitest';
import { runSuite } from './suite.ts';

test('every kernel runs clean', async () => {
  const results = [];
  for await (const r of runSuite({ computeThreads: 4096, targetMs: 200 })) {
    if (r.status === 'running') continue;
    results.push(r);
  }
  const byId = new Map(results.map((r) => [r.id, r]));
  for (const [id, r] of byId) {
    expect(r.status, `${id}: ${r.message ?? ''}`).not.toBe('error');
    if (r.status === 'ok') {
      expect(Number.isFinite(r.gflops), `${id} gflops`).toBe(true);
      expect(r.gflops, `${id} gflops`).toBeGreaterThan(0);
    }
  }
});
```

Pass small `computeThreads`/`targetMs` overrides so this runs in ~seconds in
CI, not the multi-minute full-precision run a human would use to actually
read the numbers.

**b. Per-shader unit tests, for fast iteration while writing a new kernel** —
one test per new shader that checks it alone, rather than the whole suite,
so a broken shader fails in under a second instead of after the whole
round-robin scheduler runs:

```ts
// src/shaders.browser.test.ts
import { test, expect } from 'vitest';
import { prepareFlopsF32Rsqrt } from './benchmarks/flopsMath.ts';
import { acquireGpuContext } from './gpu/context.ts';

test('flops-f32-rsqrt compiles and runs', async () => {
  const ctx = await acquireGpuContext();
  const prepared = await prepareFlopsF32Rsqrt(ctx, { iterations: 16, threads: 1024 });
  expect(prepared.kind).not.toBe('skipped'); // or assert the skip reason if expected
  // ... run one sample via KernelSampler, assert a finite duration
  ctx.device.destroy();
});
```

This is the piece to write *alongside* each new kernel from now on — see
"Workflow after migration" below.

### 6. What these tests can assert (and what they can't)

Be honest about this in the test names/comments so nobody reads more into a
green CI run than it means:

- **Can assert:** the WGSL compiles, the pipeline builds, the dispatch runs,
  the reported throughput is a finite positive number. This is exactly what
  catches the class of bug the `dot4I8Packed` incident was (silent 0s "ok").
- **Cannot assert:** that a number is *fast* — Playwright's Chromium may run
  against a software WebGPU fallback (SwiftShader/similar) in CI rather than
  real hardware, especially on a headless Linux CI runner with no GPU. If so,
  numbers will be low and not representative of `M3 GPU FLOPS characteristics`
  — that's still fine for compile/regression checks, just don't assert
  absolute throughput floors in these tests. If CI does need to represent
  real hardware numbers (e.g. a regression-tracking job), that's a separate,
  explicitly-labeled job on self-hosted GPU runners — out of scope here.

### 7. Delete the old harness

Once the browser project is green in CI:

- Remove references to the scratch-dir CDP script from memory/notes (this
  plan supersedes `headless-chrome-webgpu-harness.md` — update or retire
  that memory once this lands).
- Nothing to delete from the repo itself, since the old harness never lived
  there — it only existed in session scratch directories. This migration's
  deliverable *is* deleting the need to rebuild it every session.

### 8. CI wiring

Add a step before `pnpm test` (or a separate job) to install the Playwright
browser binary, since it's not part of `pnpm install`:

```yaml
- run: pnpm install
- run: pnpm exec playwright install --with-deps chromium
- run: pnpm test
```

`--with-deps` also installs the OS-level libraries Chromium needs on a bare
CI image (fonts, GL libs) — skip it only if the runner image already has
them (e.g. a Playwright-provided Docker image).

## Workflow after migration

Adding a new kernel becomes: write the shader + benchmark file (as today),
add one `*.browser.test.ts` case for it (a few lines, from the template in
step 5b), run `pnpm test:browser` locally — green in seconds, not a
multi-step bundle/launch/CDP dance. The full-suite smoke test (5a) catches
anything a per-kernel test missed, and both run in CI on every push.

## Open question for the user

Chromium via Playwright is the only browser this plan wires up. Safari
WebGPU timestamps are known to be unreliable (see `Safari WebGPU
timestamps` memory) and Playwright's WebKit provider is not Safari — it
won't reproduce that quirk. If Safari-specific regression coverage matters,
that needs real Safari (via `xcrun simctl`/`safaridriver`, outside
Playwright's provider model) as a separate follow-up, not part of this
migration.
