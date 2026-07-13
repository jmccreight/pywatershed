# Release guide

This document describes how to release `pywatershed`. It is written as a
sequence of concrete actions following a running, hypothetical example
which is outlined at the start of
[the step-by-step section](#releasing-pywatershed-step-by-step).

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Conventions](#conventions)
- [What is automated](#what-is-automated)
- [Releasing pywatershed, step by step](#releasing-pywatershed-step-by-step)
  - [1. Start from a ready develop (or main for a patch)](#1-start-from-a-ready-develop-or-main-for-a-patch)
  - [2. Create the release branch](#2-create-the-release-branch)
  - [3. Set the version](#3-set-the-version)
  - [4. Update documentation and metadata](#4-update-documentation-and-metadata)
  - [5. Open a pull request to main](#5-open-a-pull-request-to-main)
  - [6. Merge the pull request to main (merge, never squash)](#6-merge-the-pull-request-to-main-merge-never-squash)
  - [7. Publish the draft GitHub release](#7-publish-the-draft-github-release)
  - [8. Bring the release back into develop](#8-bring-the-release-back-into-develop)
  - [9. Publish extended release notes on gh-pages (major releases)](#9-publish-extended-release-notes-on-gh-pages-major-releases)
- [If something goes wrong](#if-something-goes-wrong)
- [Utility scripts](#utility-scripts)
  - [update_version.py](#update_versionpy)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Conventions

- Release numbers follow [semantic versioning](https://semver.org/):
  `MAJOR.MINOR.PATCH`.
- Branching follows [git flow](https://nvie.com/posts/a-successful-git-branching-model/):
  - `main` always holds the latest release.
  - `develop` holds everything intended for the next release.
  - Minor and major releases branch from `develop`. Patch releases branch
    from `main`.
  - Either way, the release lands on `main` first and is then merged from
    `main` back into `develop`, so fixes reach both branches without
    cherry-picking.
- The release branch is named with a leading `v`: `v3.0.0`. The GitHub
  release, and the tag it creates, have no leading `v`: `3.0.0`.
- On `develop`, the version carries a `.dev0` suffix for the anticipated
  next release (e.g. `3.1.0.dev0` after `3.0.0` is released).
- Pull requests to `main`, and the post-release pull request from `main`
  to `develop`, must be **merged with a merge commit, never squashed**.
  Squashing rewrites the commits and makes `main` and `develop` diverge
  permanently; merging preserves the shared history.

## What is automated

The GitHub organization does not permit Actions to push commits, open
pull requests, or merge (this forbids the "automerge" style release
automation used in the past). What remains automated lives in
`.github/workflows/release.yaml`:

- **When a release PR (branch `vX.Y.Z` into `main`) is opened or
  updated** (release.yaml: `check`): runs
  `.github/scripts/release_preflight.sh` against the version in the
  branch name — the same script you can run locally, checking
  `version.txt`, `pywatershed/version.py`, `CITATION.cff`, and the top
  section of `doc/whats-new.rst` — and builds the package, uploading it
  as the `dist` artifact on the workflow run so it can be inspected
  before merging. Regular CI (`ci.yaml`) also runs on the PR as it would
  for any PR, serving as the final full check.
- **When the release PR is merged to `main`** (release.yaml: `prep`):
  creates a *draft* GitHub release named from `version.txt` (e.g.
  `3.0.0`). Nothing is public yet.
- **When the draft release is published (a human decision in the GitHub
  UI or CLI)**: GitHub creates the `3.0.0` tag on the tip of `main`, and
  two jobs run against the tag:
  - release.yaml: `publish` verifies the package version matches the tag
    and publishes the package to
    [PyPI](https://pypi.org/p/pywatershed).
  - release.yaml: `freeze_envs` attaches a frozen conda environment file
    for each platform (Linux, macOS, Windows) to the release. The frozen
    environments record exactly the dependency versions that solved at
    release time (which can differ from what CI tested only by whatever
    conda-forge published between the release PR and publication).

Everything else below is done by a human.

## Releasing pywatershed, step by step

The steps below follow a hypothetical example. The state at the outset:

- `main` holds the latest release, `2.0.4`, tagged `2.0.4`.
- `develop` holds everything intended for the next release, a major
  release, `3.0.0`.
- We will release `3.0.0` by creating branch `v3.0.0` from `develop`,
  merging it to `main` via pull request, and publishing the draft GitHub
  release that the merge creates. Afterwards, `main` comes back into
  `develop` and `develop` moves to `3.1.0.dev0`.
- The same steps release a patch, `2.0.5`, except the release branch
  (`v2.0.5`) starts from `main` instead of `develop`. Patch differences
  are called out inline.

### 1. Start from a ready develop (or main for a patch)

Everything intended for `3.0.0` is merged to `develop` and CI is passing
there. Documentation requirements are met, including release notes in
`doc/whats-new.rst` for all changes.

If the release will have extended release notes on the `gh-pages`
branch (typically major releases), draft and review them any time
before or during the release, but publish them after the release
(step 9) — their links only resolve once the tag, the merged `main`,
and the rebuilt documentation exist.

For a patch release (`2.0.5`), the fix branches from `main` instead:
either merge the fix PR to a patch branch off `main`, or make the patch
branch itself the fix. The remaining steps are the same, with `main` as
the starting point.

### 2. Create the release branch

With your local `develop` up to date with upstream:

```shell
git switch develop
git switch -c v3.0.0
```

This branch should contain no code changes, only the release
preparations below (version files, documentation, and metadata).

### 3. Set the version

```shell
python .github/scripts/update_version.py -v 3.0.0
```

This updates `version.txt` and `pywatershed/version.py`. The release PR
check (release.yaml: `check`) verifies both against the branch name, so
a typo here (or a misnamed branch) fails before anything is released.

### 4. Update documentation and metadata

All on the `v3.0.0` branch:

- `doc/whats-new.rst`: change the top heading from
  `v3.0.0 (Unreleased)` to the release date, e.g.
  `v3.0.0 (10 July 2026)`. For a patch, its section sits below the
  pending minor/major section (which stays "Unreleased").
- `CITATION.cff`: update the `version:` and `date-released:` fields (the
  `version:` field is verified by release.yaml: `check`). For a major
  release, obtain the provisional new DOI from USGS — ideally ahead of
  time, it may require the network/VPN — and update it in all three
  places it appears: `CITATION.cff` `identifiers:`, the `README.md` DOI
  badge, and the `README.md` "How to Cite" line. Nothing automated
  checks the DOI, so verify it by hand.
- `README.md`: if the release is USGS-approved, put the approved-release
  disclaimer at the top level; otherwise keep the provisional disclaimer.
- `code.json`: update `version:`, `downloadURL:` (the release archive,
  e.g. `.../archive/refs/tags/3.0.0.zip`), and `metadataLastUpdated:`;
  check `status:` still reflects the release's USGS approval status.
  Nothing automated verifies this file, so it goes stale silently if
  skipped.
- `doc/index.rst` (major releases): add a "Version 3.0.0 (date)" section
  to the documentation landing page, following the pattern of the prior
  majors, linking the release notes and the extended release notes. The
  legacy `check_version.yaml` workflow (push to `v*` branches) fails
  without it.

When done, check your work from the repository root (the version is
taken from the branch name):

```shell
.github/scripts/release_preflight.sh
```

This runs all the mechanical checks in seconds: version files,
`CITATION.cff`, the `doc/whats-new.rst` top heading, and leftover
placeholder pull-request numbers in the new section. The release PR
runs the identical script (release.yaml: `check`), so passing locally
means those CI checks will pass.

Commit these changes to `v3.0.0` and push the branch to upstream (not
to a fork: the release branch is shared state, step 7 deletes it on
upstream, and a same-repo PR runs the release workflows with full
repository permissions):

```shell
git push upstream v3.0.0
```

### 5. Open a pull request to main

Open a PR from `v3.0.0` into `main`. Two sets of checks run:

- Regular CI (`ci.yaml`), the same as for any PR.
- The release checks (release.yaml: `check`): the same
  `release_preflight.sh` run in step 4, plus a package build. Download
  the `dist` artifact from the workflow run and inspect the package if
  desired.

### 6. Merge the pull request to main (merge, never squash)

When everything passes and the package looks good, merge the PR **with a
merge commit** (see [Conventions](#conventions) for why squashing is not
allowed here).

The merge triggers creation of a draft GitHub release named `3.0.0`
(release.yaml: `prep`).

### 7. Publish the draft GitHub release

Find the draft at
[github.com/DOI-USGS/pywatershed/releases](https://github.com/DOI-USGS/pywatershed/releases).
Edit the release notes as needed (they can also be edited after
publishing). When it looks good, publish it via the GitHub UI or
`gh release edit 3.0.0 --draft=false`.

Publishing:

- creates the `3.0.0` tag on the tip of `main` (tags are created at
  publish time, not draft time),
- publishes the package to PyPI (release.yaml: `publish`), after
  verifying the built package's version matches the tag — a mismatch
  (e.g. version files on `main` changed after the release PR merged)
  fails the publication before anything reaches PyPI, which matters
  because a version number can never be re-uploaded to PyPI, and
- attaches `environment_frozen_{Linux,macOS,Windows}.yml` files to the
  release (release.yaml: `freeze_envs`).

Confirm the PyPI version and the release assets, then delete the
`v3.0.0` branch on upstream.

### 8. Bring the release back into develop

The release commits on `main` (version files, dates, citation) must flow
back into `develop`, along with setting the next development version:

```shell
git switch main && git pull upstream main
git switch -c post_v3.0.0
python .github/scripts/update_version.py -v 3.1.0.dev0
```

Then, on this branch:

- Add a new `v3.1.0 (Unreleased)` section to the top of
  `doc/whats-new.rst`.
- If `main` got the approved-release disclaimer, revert `README.md` to
  the provisional disclaimer.

Open a PR from `post_v3.0.0` into `develop` and merge it **with a merge
commit** (again, never squash). This single merge is also how patch
releases reach `develop`: after releasing `2.0.5` from `main`, the same
main-into-develop PR carries the fix, so nothing is cherry-picked.

### 9. Publish extended release notes on gh-pages (major releases)

Extended release notes live on the `gh-pages` branch as a Jekyll post
(`_posts/YYYY-MM-DD-vX-Y-Z-overview.md`) plus a link entry at the top
of that branch's `README.md`. Before pushing:

- The release date appears in four coupled places that must agree: the
  post filename, its front-matter `date:`, the permalinks in
  `README.md` (permalinks derive from the post date), and the dated
  whats-new anchor in the post
  (e.g. `whats-new.html#v3-0-0-11-july-2026`).
- Check the links that only resolve post-release: the GitHub release
  tag URL, the whats-new anchor, and readthedocs pages for new classes
  (these need the documentation build from `main` to have completed).

Push `gh-pages` and confirm the rendering at
<https://doi-usgs.github.io/pywatershed/>. The notes can be edited and
re-pushed at any time after release. Note: `gh-pages` has no
`.pre-commit-config.yaml`, so commit there with
`PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit ...`.

## If something goes wrong

- **A check fails on the release PR**: fix it on the `v3.0.0` branch and
  push; the checks rerun.
- **Something was missed, discovered after merging to `main` but before
  publishing**: the draft release is cheap and nothing is public or
  tagged yet. Merge another PR with the fix into `main` (for release-prep
  files this can be a plain branch; drafts are only created by `v*`-named
  PRs and this one already exists). Since the tag is created on the tip
  of `main` at publish time, the published release includes the fix. The
  fix must leave the version files at `3.0.0` — if it changes them, the
  publication fails on purpose (release.yaml: `publish` requires the
  package version to match the tag). In that case, or whenever the fix
  deserves its own version number, delete the draft and redo from step 2
  with the new version.
- **Something was missed, discovered after publishing**: the release is
  public and tagged; do not rewrite it. Release a patch (e.g. `3.0.1`)
  from `main` following the patch steps above.

## Utility scripts

Release-related scripts live in `.github/scripts`.

### update_version.py

Updates the version numbers embedded in the repository (`version.txt`
and `pywatershed/version.py`). A file lock ensures only one process
edits version files at a time.

Set the version for a release:

```shell
python .github/scripts/update_version.py -v 3.0.0
```

Set the development version on the branch returning `main` to `develop`
after a release:

```shell
python .github/scripts/update_version.py -v 3.1.0.dev0
```

Print the current version without changing anything:

```shell
python .github/scripts/update_version.py -g
```
