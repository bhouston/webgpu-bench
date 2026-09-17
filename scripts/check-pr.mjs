import { readFileSync } from 'node:fs';
const { pull_request: pr } = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8'));
if (pr.base.ref === 'main') {
  if (pr.head.ref !== 'dev' || pr.head.repo.full_name !== pr.base.repo.full_name) {
    throw new Error('Only this repository’s dev branch may target main.');
  }
} else {
  if (pr.base.ref !== 'dev') throw new Error('Contribution PRs must target dev.');
  const match = /^(?:feat|feature|fix|docs|chore|refactor|test|ci|perf|build)\/(\d+)-[a-z0-9-]+$/.exec(pr.head.ref);
  if (!match) throw new Error('Use a branch such as feat/42-batch-export.');
  const closing = new RegExp(`\\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\\s+#${match[1]}\\b`, 'i');
  if (!closing.test(pr.body ?? '')) throw new Error(`Include Closes #${match[1]} in the PR body.`);
}
