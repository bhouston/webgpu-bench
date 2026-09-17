# Contributing

These rules apply to humans, Claude, Codex, and other agents. This file is the single workflow standard.

## Issue → branch → PR

1. Before starting a feature or improvement, create a GitHub issue (or reuse the existing requested issue). Describe what changes, why, acceptance criteria, and constraints, following the feature issue template. Agents should use `gh issue create --body-file`.
2. Branch from updated `dev`, using `<type>/<issue>-<short-description>`, for example `feat/42-batch-export` or `fix/43-empty-input`. Never commit directly to `main` or `dev`.
3. Implement the issue and run the checks below. Use Conventional Commits for every commit. Reference the issue in the body when useful.
4. Push the branch and open a PR **against `dev`**, with a Conventional Commit title, a description of behavior and validation, and `Closes #42` matching the branch's issue number. Use `gh pr create --base dev --body-file`. Do not merge unless requested.
5. Squash merge contribution PRs with their Conventional Commit title and preserve any breaking-change footer. Release PRs go from **`dev` to `main`** and must use a **merge commit**, preserving the original commits for release analysis. Never squash or rebase a release PR.
6. Only merging `dev` into `main` triggers publication. After a release, merge `main` back into `dev` to keep ancestry aligned. Agents must not publish locally or manually bump versions.

GitHub automatically closes linked issues when changes reach the default branch. If the default branch remains `main`, closure happens at release time; selecting `dev` as the default branch closes them at integration time.

## Commit format

Use `type(optional-scope): description`. Allowed types are `feat`, `fix`, `perf`, `docs`, `chore`, `refactor`, `test`, `style`, `build`, `ci`, and `revert`.

- `feat` produces a minor release; `fix` and `perf` produce patch releases.
- `feat!:` or a `BREAKING CHANGE: ...` footer produces a major release.
- Other types do not ordinarily produce a release.

Husky checks messages locally; CI checks PR titles and new commits even when local hooks are bypassed. Existing pre-adoption history is not linted.

## Development and checks

Use Node 22 or newer (CI publishing uses Node 24) and the pinned pnpm version in `package.json`.

```sh
pnpm install --frozen-lockfile
pnpm exec playwright install chromium webkit
pnpm build
pnpm tsc
pnpm lint
pnpm test
pnpm test:workflow
pnpm audit --audit-level high
pnpm size
```

`pnpm test` includes core unit tests, Chromium WebGPU tests, CLI integration tests, and a coverage gate. Linux CI installs lavapipe for software WebGPU. WebKit runs separately on macOS as advisory because hosted runners lack Metal GPU access. Use `pnpm test:all` for all projects locally. Build before running the root tests.

Coverage includes all production TypeScript in both packages. CLI subprocess code is not measured by Vitest's in-process coverage; integration tests still exercise it. Thresholds are 75% for aggregate statements, functions, and lines, and 70% for branches; raise them as coverage improves. CI uploads HTML and LCOV reports and sends push coverage to Codecov using OIDC. Activate this public repository in Codecov to enable the percentage badge; no token is required.

Size Limit measures compressed compiled JavaScript, with budgets of 100 kB for core and 20 kB for CLI. This excludes external dependencies and native Dawn binaries; it is not the CLI's full install size. Dependency auditing fails on high or critical advisories, including development dependencies.

## Releases and initial setup

See [the release setup guide](docs/releases.md) for npm trusted publishing and GitHub settings. Both packages receive the same version and the CLI requires that exact core version. Semantic Release derives versions from `v*` tags and Conventional Commits. The generated changelog lives in [GitHub Releases](https://github.com/bhouston/webgpu-bench/releases); source package manifests are development versions and are not rewritten by the release bot.

After this pilot's first successful release, copy the shared policy, templates, hooks, and CI into a dedicated GitHub template repository. Adapt package paths, repository names, release baseline, coverage and size budgets, and npm publisher settings per repository. Do not copy this repository's release baseline or assume its two-package configuration fits another project.
