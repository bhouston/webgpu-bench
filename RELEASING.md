# Releasing

Releases are manual and maintainer-only. Merging to `main` never publishes.

## Running a Release

```sh
gh workflow run release.yml --ref main                  # release
gh workflow run release.yml --ref main -f dry_run=true  # preview only, publishes nothing
```

When there are no release-worthy commits since the last `v*` tag, the run is a no-op.

## What the Workflow Does

1. Refuses any ref other than `main`.
2. Runs the full CI workflow on the dispatched commit.
3. Stops if `main` has moved since the dispatch.
4. Installs, builds, and runs `pnpm release`, which is semantic-release.
5. semantic-release computes the next version, creates the `v<version>` tag and GitHub Release, and publishes.

Releases run one at a time and never cancel a publish in progress. npm packages publish through npm trusted publishing (GitHub OIDC), so no `NPM_TOKEN` is stored.

## Versioning

The version comes from the Conventional Commits since the last `v*` tag. The highest change wins:

- `feat` produces a minor release.
- `fix` and `perf` produce a patch release.
- `!` or a `BREAKING CHANGE:` footer produces a major release.
- Other types do not trigger a release.

Everything this repository publishes shares one version. `package.json` versions in source are development snapshots; the Git tag and the registry hold the released version. [GitHub Releases](https://github.com/bhouston/webgpu-bench/releases) is the changelog of record.

## One-Time Setup

1. On npmjs.com, open each package's Trusted Publisher settings, choose GitHub Actions, and enter:

   | Field                | Value         |
   | -------------------- | ------------- |
   | Organization or user | `bhouston`    |
   | Repository           | `webgpu-bench`    |
   | Workflow filename    | `release.yml` |
   | Environment name     | `npm`         |

2. Protect `main`: require pull requests and the CI checks, allow merge commits, and block force pushes and deletion.
3. If the repository uploads coverage to Codecov, add the `CODECOV_TOKEN` repository secret.

## Recovery

Publishing is not atomic. If a run fails part way, compare the registry versions, the `v<version>` tag, and the workflow log, then finish what is missing from the same release commit. If publishing succeeded but the GitHub Release failed, create the release from the existing tag. Never republish a version, delete a published version, or move a release tag. Resolve the failure before dispatching another release.

## Repository Specifics

### Published Artifacts

- npm package `webgpu-bench-core` (`packages/core`).
- npm package `webgpu-bench` (`packages/cli`), the command-line tool.

### Package Publishing

`@anolilab/semantic-release-pnpm` publishes `packages/core` first, then `packages/cli`, each with `pnpm publish`. It writes the released version into each manifest in memory only, including the exact `webgpu-bench-core` version the CLI depends on, so committed manifests are never changed. Both packages share one version. If core publishes and the CLI does not, finish the CLI from the same release commit; do not republish core.

### Thresholds

CI enforces Brotli size budgets (`pnpm size`): 100 kB for core compiled JavaScript and 20 kB for CLI compiled JavaScript with dependencies excluded. Explain any change in the PR.

### Local Validation

```sh
pnpm build
pnpm --filter webgpu-bench-core pack --dry-run
pnpm --filter webgpu-bench pack --dry-run
```

npm OIDC authentication can only be verified inside GitHub Actions, so do not run `pnpm release` locally to test it.
