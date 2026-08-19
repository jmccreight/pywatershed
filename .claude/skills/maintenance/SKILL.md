---
name: maintenance
description: Walk the repo's MAINTENANCE.md ledger of cross-repo and externally-blocked todos, check each item's blocked-on condition live via public APIs, and report what is actionable now. Use when the user invokes /maintenance or asks about maintenance status, loose ends, or whether something is unblocked.
---

# Working the maintenance ledger

`MAINTENANCE.md` at the repo root is the source of truth: a list of
maintenance items, each with a **Blocked on** condition (stated so it
can be checked mechanically) and an **Action** for when it unblocks.
Read it fully first.

## Ground rules

- You run read-only checks (curl against public APIs, local greps) and
  edit `MAINTENANCE.md`; the human runs all git commands and clicks
  (re-runs, merges, PR creation). Draft paste-ready commands.
- Read-only git (`status`, `log`, `branch`, `remote`, `diff`) in this
  repo and the related clones listed below is fine. Anything that
  creates, mutates, or destroys state — commits, pushes, PRs,
  comments, or running project code — requires asking the human AND
  receiving explicit permission first; drafting a command is not
  permission to run it.
- Unauthenticated GitHub API: 60 requests/hour per IP. Batch, cache to
  /tmp, and prefer one query that answers several items.
- The ledger is repo-tracked: after updating it (item status, moves to
  Done with a date), hand the human a commit. New items discovered in
  conversation get added with a checkable blocked-on condition.

## Procedure

1. Read `MAINTENANCE.md`.
2. For each Open item, evaluate its blocked-on condition live (recipes
   below). Classify: **actionable now**, **still blocked** (say what
   you observed), or **already done** (evidence found — move to Done).
3. Report a short table: item / verdict / evidence. Lead with the
   actionable items and draft their commands.
4. Update `MAINTENANCE.md` to match reality; hand over the commit.

## Check recipes

- PyPI latest + release dates:
  `curl -s https://pypi.org/pypi/<pkg>/json` →
  `info.version`, `releases[<v>][0].upload_time`.
- conda-forge package versions:
  `curl -s https://api.anaconda.org/package/conda-forge/<pkg>` →
  `versions`.
- GitHub PR state:
  `curl -s https://api.github.com/repos/<owner>/<repo>/pulls/<n>` →
  `state`, `merged_at`; open PRs:
  `.../pulls?state=open`.
- GitHub releases/tags:
  `curl -s https://api.github.com/repos/<owner>/<repo>/releases/latest`.
- Whether a commit/PR is in a release: compare the release tag date to
  the PR `merged_at`, or
  `.../compare/<tag>...<sha>` → `status` (`behind` means included).
- A deployed GitHub Pages site: `curl -sI https://<site>/<path>` (200
  vs 404) — sufficient to check publication without a browser.
- CI state on a sha:
  `.../commits/<sha>/check-runs?per_page=100` (see the `/ci-triage`
  skill for interpreting results).

## Related repos this ledger spans

- `DOI-USGS/pywatershed` (this repo) and James's fork `jmccreight/...`
- `DOI-USGS/pyPRMS` (upstream dependency; local clone `~/usgs/pyPRMS`)
- `conda-forge/pywatershed-feedstock` and `conda-forge/pyprms-feedstock`
  (local clones `~/usgs/pywatershed-feedstock`, `~/usgs/pyPRMS-feedstock`)
- `modflowpy/flopy` (dependency whose release gates the 3.0.1 hotfix)
