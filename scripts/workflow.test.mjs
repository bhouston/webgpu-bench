import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { cpSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { analyzeCommits } from '@semantic-release/commit-analyzer';

test('release analysis maps conventional changes to the requested bump', async () => {
  for (const [message, expected] of [
    ['fix: correct output', 'patch'],
    ['perf: reduce allocations', 'patch'],
    ['feat: add export', 'minor'],
    ['feat!: remove old API', 'major'],
    ['feat!: remove old API\n\nBREAKING CHANGE: use the new API', 'major'],
    ['chore: update tooling', null],
  ]) {
    const type = await analyzeCommits(
      { preset: 'conventionalcommits' },
      { cwd: process.cwd(), commits: [{ hash: 'test', message }], logger: { log() {} } },
    );
    assert.equal(type, expected);
  }
});

test('staging packages synchronizes versions and removes workspace and development metadata', () => {
  const dir = mkdtempSync(join(tmpdir(), 'webgpu-release-'));
  try {
    mkdirSync(join(dir, 'scripts'));
    cpSync('scripts/stage-release.mjs', join(dir, 'scripts/stage-release.mjs'));
    writeFileSync(join(dir, 'LICENSE'), 'MIT');
    for (const name of ['core', 'cli']) {
      const target = join(dir, 'packages', name);
      mkdirSync(join(target, 'dist'), { recursive: true });
      writeFileSync(join(target, 'dist/index.js'), 'export {};');
      cpSync(`packages/${name}/package.json`, join(target, 'package.json'));
      writeFileSync(join(target, 'README.md'), name);
    }
    execFileSync(process.execPath, [join(dir, 'scripts/stage-release.mjs'), '0.10.0']);
    for (const name of ['core', 'cli']) {
      const target = join(dir, 'packages', name);
      const pkg = JSON.parse(readFileSync(join(target, 'publish/package.json')));
      assert.equal(pkg.version, '0.10.0');
      assert.equal(pkg.scripts, undefined);
      assert.equal(pkg.devDependencies, undefined);
      assert.equal(pkg.publishConfig.access, 'public');
      assert.equal(readFileSync(join(target, 'publish/LICENSE'), 'utf8'), 'MIT');
      assert.equal(JSON.parse(readFileSync(join(target, 'package.json'))).version, '0.9.3');
      if (name === 'cli') assert.equal(pkg.dependencies['webgpu-bench-core'], '0.10.0');
    }
    const invalid = spawnSync(process.execPath, [join(dir, 'scripts/stage-release.mjs'), 'not-a-version']);
    assert.notEqual(invalid.status, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('PR policy accepts linked work and same-repository promotions only', () => {
  const dir = mkdtempSync(join(tmpdir(), 'webgpu-pr-'));
  const event = join(dir, 'event.json');
  try {
    for (const [base, head, body, fork, accepted] of [
      ['dev', 'feat/42-export', 'Closes #42', false, true],
      ['dev', 'feat/42-export', 'Closes #43', false, false],
      ['dev', 'unlinked-work', 'Closes #42', false, false],
      ['main', 'dev', '', false, true],
      ['main', 'dev', '', true, false],
      ['main', 'feat/42-export', 'Closes #42', false, false],
    ]) {
      writeFileSync(
        event,
        JSON.stringify({
          pull_request: {
            base: { ref: base, repo: { full_name: 'bhouston/webgpu-bench' } },
            head: { ref: head, repo: { full_name: fork ? 'fork/webgpu-bench' : 'bhouston/webgpu-bench' } },
            body,
          },
        }),
      );
      const result = spawnSync(process.execPath, ['scripts/check-pr.mjs'], {
        env: { ...process.env, GITHUB_EVENT_PATH: event },
      });
      assert.equal(result.status === 0, accepted, `${base} ← ${head}: ${body}`);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
