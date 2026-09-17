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

Before merging a release PR:

1. Enable branch protection for `dev` and `main`, require PRs and passing `ci` and `policy` checks, block force pushes and deletion, and require branches up to date. Do not require the advisory `webkit` check. Allow merge commits on main (linear history would conflict with the release model).
2. Allow squash merges for contributions; merge `dev` → `main` with a merge commit to retain version-bearing commits. Use a conventional title such as `chore(release): promote dev to main`.
3. Optionally set `dev` as the default branch so ordinary PRs target it and linked issues close on integration.
4. Activate the repository in Codecov for the coverage badge.
5. Configure both npm trusted publishers before the first release merge.

Only pushes to `main` trigger Semantic Release, after the reusable CI workflow passes. The release job has package OIDC and repository-content write permissions; PR jobs cannot publish. Releases are serialized and never cancelled midway. GitHub release comments and issue labeling are disabled, so no issue-write token permission is needed.

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
