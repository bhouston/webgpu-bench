import { readFileSync } from 'node:fs';
const { pull_request: pr } = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8'));
if (pr.base.ref !== 'main') throw new Error('Contribution PRs must target main.');
const closing = /\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#\d+\b/i;
if (!closing.test(pr.body ?? '')) throw new Error('Include Closes #<issue-number> in the PR body.');
