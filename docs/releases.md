# Release setup

## npm trusted publishing (maintainer action)

Configure a GitHub Actions trusted publisher in the npm settings of **each** package:

- https://www.npmjs.com/package/webgpu-bench-core/access
- https://www.npmjs.com/package/webgpu-bench/access

Use these exact values for both:

| Field                | Value                                       |
| -------------------- | ------------------------------------------- |
| Organization or user | `bhouston`                                  |
| Repository           | `webgpu-bench`                              |
| Workflow filename    | `release.yml`                               |
| Environment          | Leave blank (no GitHub Environment is used) |

The workflow is `.github/workflows/release.yml`. Do not enter a path in npm's workflow filename field. No `NPM_TOKEN` or `NODE_AUTH_TOKEN` is needed. GitHub-hosted Ubuntu, Node 24, npm 11.5.1 or newer, and `id-token: write` provide OIDC authentication and provenance. The npm plugin also ships a compatible npm CLI. See [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/).

## GitHub settings

`main` is the default and sole active integration branch. It is protected with required PRs, up-to-date `ci` and `policy` checks, resolved conversations, no force pushes, and no deletion (including administrator enforcement). Squash merges are enabled for contribution PRs; rebase merging is disabled. For future setup or verification:

1. Enable branch protection for `main`: require PRs and passing `ci` and `policy` checks, block force pushes and deletion, and require branches up to date. Do not require the advisory `webkit` check.
2. Set `main` as the default branch so ordinary PRs target it and linked issues close on merge.
3. Activate the repository in Codecov for the coverage badge.
4. Configure both npm trusted publishers before the first release.

Releases never run automatically. A maintainer dispatches them manually with `gh workflow run release.yml --ref main` (or via the Actions tab), after the reusable CI workflow passes on the dispatched commit. Dispatching from any ref other than `main` is rejected before any release step runs. Pass a dry run to verify staging, changelog rendering, and tag ancestry without publishing. The release job has package OIDC and repository-content write permissions; PR jobs cannot publish. Releases are serialized and never cancelled midway. GitHub release comments and issue labeling are disabled, so no issue-write token permission is needed. A dispatch with no release-worthy commits since the last tag succeeds as a no-op.

## Version baseline and release behavior

The existing published version is `0.9.3` for both packages. The adoption setup tags historical commit `6f065d25426d440225ca1f1787885ce5ab43c851`, which introduced those versions, as `v0.9.3`. Without this baseline Semantic Release would assume a first release of `1.0.0`.

`feat` bumps minor, `fix`/`perf` bump patch, and breaking changes bump major, including while on 0.x. A maintenance-only merge may correctly produce no release. Release notes are generated as the GitHub Release changelog. Source manifests keep their development versions: the staging script copies built files, license, and README, writes the calculated version to both staged manifests, and replaces the workspace dependency with the exact released core version. The official npm plugins publish core before CLI.

Validate staging locally without publishing:

```sh
pnpm build
pnpm release:stage
npm pack ./packages/core/publish --dry-run
npm pack ./packages/cli/publish --dry-run
```

Actual OIDC authentication can only be verified inside GitHub Actions after npm setup. Do not run the release command locally to test authentication. Publication of two npm packages is not atomic. If a publish fails after a tag or one package is published, inspect npm and the GitHub release before retrying; Semantic Release does not automatically repair partially published releases. Never delete a published npm version to retry. Correct the cause and recover the missing package from the same tag, or make a new fix commit and release a new synchronized version.
