export default {
  branches: ['main'],
  repositoryUrl: 'https://github.com/bhouston/webgpu-bench.git',
  tagFormat: 'v${version}',
  plugins: [
    ['@semantic-release/commit-analyzer', { preset: 'conventionalcommits' }],
    ['@semantic-release/release-notes-generator', { preset: 'conventionalcommits' }],
    ['@anolilab/semantic-release-pnpm', { pkgRoot: 'packages/core' }],
    ['@anolilab/semantic-release-pnpm', { pkgRoot: 'packages/cli' }],
    ['@semantic-release/github', { successComment: false, failComment: false, releasedLabels: false }],
  ],
};
