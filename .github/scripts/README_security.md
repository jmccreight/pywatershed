<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Security and PII Checking](#security-and-pii-checking)
  - [Overview](#overview)
  - [Usage](#usage)
    - [Pre-commit Hook (Automatic)](#pre-commit-hook-automatic)
    - [Manual Checking](#manual-checking)
      - [Check staged files (same as pre-commit)](#check-staged-files-same-as-pre-commit)
      - [Check all tracked files (full audit)](#check-all-tracked-files-full-audit)
      - [Check specific issue types only](#check-specific-issue-types-only)
      - [Verbose output (show line-by-line details)](#verbose-output-show-line-by-line-details)
      - [Save results to file](#save-results-to-file)
  - [Configuration](#configuration)
    - [Excluded Files](#excluded-files)
    - [Allowed Paths](#allowed-paths)
    - [Allowed Domains](#allowed-domains)
    - [Filtering](#filtering)
  - [False Positives](#false-positives)
    - [If you get false positives for absolute paths:](#if-you-get-false-positives-for-absolute-paths)
    - [If you get false positives for credentials:](#if-you-get-false-positives-for-credentials)
    - [If you get false positives for IP addresses:](#if-you-get-false-positives-for-ip-addresses)
  - [Administrative Review](#administrative-review)
  - [Exit Codes](#exit-codes)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Security and PII Checking

Centralize and build comprehensive security checks which are applied at commit-time or as requested (e.g. for reviews).

## Overview

The `check_security.py` script checks for:

1. **Absolute file system paths** - Paths like `/Users/username/...`, `/home/username/...`, `C:\Users\...` that might reveal developer machine information
2. **IP addresses** - IP addresses that might point to internal servers
3. **Internal server hostnames** - Domain names that might reveal internal infrastructure
4. **Credentials** - Passwords, API keys, tokens, and other secrets

## Usage

### Pre-commit Hook (Automatic)

The security checker runs automatically on staged files when you commit. If issues are found, the commit will be blocked.

To bypass the check temporarily (not recommended):

```bash
git commit --no-verify
```

### Manual Checking

#### Check staged files (same as pre-commit)

```bash
python .github/scripts/check_security.py
```

#### Check all tracked files (full audit)

```bash
python .github/scripts/check_security.py --audit
```

#### Check specific issue types only

```bash
# Only check for absolute paths
python .github/scripts/check_security.py --check-paths

# Only check for credentials
python .github/scripts/check_security.py --check-credentials

# Check multiple specific types
python .github/scripts/check_security.py --check-ips --check-hostnames
```

#### Verbose output (show line-by-line details)

```bash
python .github/scripts/check_security.py --audit --verbose
```

#### Save results to file

```bash
python .github/scripts/check_security.py --audit --output security_report.txt
```

## Configuration

### Excluded Files

The following file types are automatically excluded:

- Binary files (images, PDFs, compiled files, etc.)
- NetCDF files (`.nc`)
- Python files (`.py`, `.pyc`) when checking hostnames
- Configuration files (`.xml`, `.json`, `.yaml`, `.yml`, `.test`)
- Lock files

### Allowed Paths

Some absolute paths are allowed because they're standard system paths:

- `/usr/bin`, `/usr/local`, `/bin`, `/etc`, `/opt`, `/var`, `/tmp`
- `/dev/null`

### Allowed Domains

Public domains are allowlisted and won't be flagged:

- `github.com`
- `usgs.gov` (public USGS sites)
- `doi.gov`
- `doi.org` (Digital Object Identifier system)
- `doi-usgs.github.io` (pywatershed GitHub Pages documentation)
- `sciencebase.gov`
- `waterdata.usgs.gov`
- `waterservices.usgs.gov`
- `pypi.org`, `conda.io`, `anaconda.org`
- `readthedocs.io`
- `docs.xarray.dev` (xarray documentation)

### Filtering

The checker includes smart filtering to reduce false positives:

- **Version numbers**: Strings like `5.2.1.1` or `1.0.dev` are not flagged as IP addresses or hostnames when in version context
- **Python code**: Module/attribute references like `pytest.main`, `dict.items`, `vv.stdev`, or `np.testing.assert_equal` are not flagged as hostnames
- **File paths in URLs**: If a line contains an allowed domain (like `github.com`), other parts of that URL won't be flagged
- **Quoted strings**: Version numbers in quotes (like `"5.2.1.1"`) are recognized and not flagged as IPs
- **Internal hostnames**: Hostnames containing "usgs" or "doi" are flagged **unless** they match the specific public domains in the allowlist (e.g., `usgs.gov`, `doi.org` are allowed, but `internal.usgs.gov` would be flagged)

## False Positives

### If you get false positives for absolute paths:

1. **Add to `.gitignore`** - If it's a generated file, make sure it's not tracked
2. **Add to exclusion patterns** - Edit `check_security.py` and add the pattern to `exclude_patterns`
3. **Add to allowed paths** - If it's a legitimate system path, add it to `allowed_paths` in the script

### If you get false positives for credentials:

The script tries to skip obvious placeholders like:

- Lines containing "example", "placeholder", "your\_"
- Lines with `xxx`, `***`, or `<password>`

If you have documentation with example credentials, use these patterns.

### If you get false positives for IP addresses:

Version numbers like `5.2.1.1` should be automatically filtered if they appear in version-related context. If you still get false positives, the line should contain keywords like "version", "prms", or have the number in quotes.

## Administrative Review

For code reviews, run a full audit before submission:

```bash
python .github/scripts/check_security.py --audit --verbose --output security_check.txt
```

This generates a report that can be included with your review documentation.

## Exit Codes

- `0`: No issues found
- `1`: Issues found (will block pre-commit)
