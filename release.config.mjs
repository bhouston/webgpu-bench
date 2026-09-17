export default {
  branches: ['main'],
  repositoryUrl: 'https://github.com/bhouston/webgpu-bench.git',
  tagFormat: 'v${version}',
  plugins: [
    '@semantic-release/commit-analyzer',
    '@semantic-release/release-notes-generator',
    ['@semantic-release/exec', { prepareCmd: 'node scripts/stage-release.mjs ${nextRelease.version}' }],
    ['@semantic-release/npm', { pkgRoot: 'packages/core/publish' }],
    ['@semantic-release/npm', { pkgRoot: 'packages/cli/publish' }],
    ['@semantic-release/github', { successComment: false, failComment: false, releasedLabels: false }],
  ],
};
