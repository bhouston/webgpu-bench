import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
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

test('PR policy accepts issue-linked feature branches into main only', () => {
  const dir = mkdtempSync(join(tmpdir(), 'webgpu-pr-'));
  const event = join(dir, 'event.json');
  try {
    for (const [base, head, body, accepted] of [
      ['main', 'feat/42-export', 'Closes #42', true],
      ['main', 'feat/42-export', 'Closes #43', false],
      ['main', 'unlinked-work', 'Closes #42', false],
      ['dev', 'feat/42-export', 'Closes #42', false],
    ]) {
      writeFileSync(
        event,
        JSON.stringify({
          pull_request: {
            base: { ref: base, repo: { full_name: 'bhouston/webgpu-bench' } },
            head: { ref: head, repo: { full_name: 'bhouston/webgpu-bench' } },
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
