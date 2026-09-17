import { readFileSync } from 'node:fs';

const report = JSON.parse(readFileSync('coverage/coverage-summary.json', 'utf8'));
for (const metric of ['statements', 'branches', 'functions', 'lines']) {
  if (!(report.total?.[metric]?.total > 0)) throw new Error(`Empty ${metric} coverage report`);
}
for (const name of ['core', 'cli']) {
  if (!Object.keys(report).some((file) => file.includes(`/packages/${name}/src/`))) {
    throw new Error(`Coverage is missing package ${name}`);
  }
}
