#!/usr/bin/env bash

# Preflight checks for a pywatershed release.
#
# Usage:
#   .github/scripts/release_preflight.sh [version]
#
# The version may be given with or without the leading "v" (e.g. v3.0.0
# or 3.0.0). With no argument, the version is taken from the name of the
# current git branch, which must be named vMAJOR.MINOR.PATCH.
#
# Run from the repository root. All checks are run and reported before
# exiting; exits non-zero if any check fails.
#
# This same script is run on release pull requests by the check job in
# .github/workflows/release.yaml, so passing locally means the CI
# version checks will pass. See .github/RELEASE.md.

fail=0
err() {
    echo "FAIL: $1"
    fail=1
}
ok() {
    echo "  ok: $1"
}
warn() {
    echo "WARN: $1"
}

# --- resolve the version
if [ -n "$1" ]; then
    version="${1#v}"
    echo "Checking release version: $version (from argument)"
else
    branch=$(git rev-parse --abbrev-ref HEAD)
    version="${branch#v}"
    echo "Checking release version: $version (from branch name '$branch')"
fi

if ! echo "$version" | grep -qE "^[0-9]+\.[0-9]+\.[0-9]+$"; then
    err "version '$version' is not MAJOR.MINOR.PATCH"
    echo "Exiting: remaining checks are meaningless without a version."
    exit 1
fi

# --- version files
if grep -qE "^${version}[[:space:]]*$" version.txt; then
    ok "version.txt"
else
    err "version.txt does not match: $(cat version.txt)"
fi

if grep -qE "^__version__ = \"${version}\"[[:space:]]*$" pywatershed/version.py; then
    ok "pywatershed/version.py"
else
    err "pywatershed/version.py does not match: $(grep '^__version__' pywatershed/version.py)"
fi

if grep -qE "^version: ${version}[[:space:]]*$" CITATION.cff; then
    ok "CITATION.cff version"
else
    err "CITATION.cff does not match: $(grep '^version:' CITATION.cff)"
fi

# --- CITATION.cff release date (warning only: may be a planned date)
cff_date=$(grep "^date-released:" CITATION.cff | sed -E "s/^date-released:[[:space:]]*'?([0-9-]+)'?.*/\1/")
today=$(date "+%Y-%m-%d")
if [ "$cff_date" == "$today" ]; then
    ok "CITATION.cff date-released is today"
else
    warn "CITATION.cff date-released is '$cff_date' (today is $today)"
fi

# --- whats-new.rst: top section is this version and dated
top_heading=$(grep -m 1 -E "^v[0-9]" doc/whats-new.rst)
if ! echo "$top_heading" | grep -qE "^v${version} \("; then
    err "top heading of doc/whats-new.rst is '$top_heading', expected v${version} (...)"
elif echo "$top_heading" | grep -qi "unreleased"; then
    err "top heading of doc/whats-new.rst still says Unreleased: '$top_heading'"
else
    ok "doc/whats-new.rst top heading: '$top_heading'"
fi

# --- whats-new.rst: no placeholder pull numbers in the top (this
# release's) section, i.e. before the second version heading
top_section=$(awk "/^v[0-9]/ {count++} count == 1" doc/whats-new.rst)
if echo "$top_section" | grep -q "XXX"; then
    err "placeholder pull request numbers (XXX) in the top section of doc/whats-new.rst:"
    echo "$top_section" | grep -n "XXX" | sed "s/^/        /"
else
    ok "doc/whats-new.rst top section has no placeholder pull numbers"
fi

# --- verdict
echo
if [ "$fail" -ne 0 ]; then
    echo "Release preflight FAILED for version $version."
    exit 1
fi
echo "Release preflight passed for version $version."
