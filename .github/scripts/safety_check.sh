#!/bin/bash

# Safety check script for conda environments
# This script exports the current conda environment, separates conda and pip
# packages, and runs Safety scans on each group separately plus pyproject.toml.

# Note: not using set -e so all three scans run even if one finds vulnerabilities

# Function to check Safety output for unignored vulnerabilities
# Returns 0 if no unignored vulnerabilities, 1 if any found
# Parses the summary line: "N vulnerability found, M ignored due to policy."
check_vulnerabilities() {
    local output="$1"
    local summary=$(echo "$output" | grep -E '[0-9]+ vulnerabilit.* found, [0-9]+ ignored')
    if [ -z "$summary" ]; then
        # No summary line found means no vulnerabilities
        return 0
    fi
    local found=$(echo "$summary" | grep -oE '^[0-9]+')
    local ignored=$(echo "$summary" | grep -oE '[0-9]+ ignored' | grep -oE '[0-9]+')
    found=${found:-0}
    ignored=${ignored:-0}
    local unignored=$((found - ignored))
    if [ "$unignored" -gt 0 ]; then
        return 1
    fi
    return 0
}

# Temporary directories for isolated scans
CONDA_SCAN_DIR=$(mktemp -d)
PIP_SCAN_DIR=$(mktemp -d)

# Function to clean up temporary files and directories
cleanup() {
    echo ""
    echo "Cleaning up temporary scan directories..."
    rm -rf "$CONDA_SCAN_DIR" "$PIP_SCAN_DIR"
}

# Trap to ensure cleanup happens even if the script is interrupted
trap cleanup EXIT

# Check if we're in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: No conda environment is active."
    echo "Please activate your conda environment first."
    exit 1
fi

echo "Running safety check on conda environment: $CONDA_DEFAULT_ENV"
echo ""
echo "This script performs three security scans:"
echo "  1. Conda-installed packages from the active environment"
echo "  2. Pip-installed packages from the active environment"
echo "  3. Dependencies declared in pyproject.toml"
echo ""
echo "Freezing current environment..."
echo "  - Exporting all packages from conda environment"
echo "  - Separating into conda-installed and pip-installed packages"
echo "  - Writing temporary requirements files for scanning"
echo ""

# Generate requirements file from current conda environment
FULL_EXPORT=$(mktemp)
conda list -e >"$FULL_EXPORT"

# Separate conda and pip packages into their respective scan directories
CONDA_REQ="$CONDA_SCAN_DIR/requirements.txt"
PIP_REQ="$PIP_SCAN_DIR/requirements.txt"

# Copy policy file into each scan directory if it exists
POLICY_FILE=".safety-policy.yml"
if [ -f "$POLICY_FILE" ]; then
    cp "$POLICY_FILE" "$CONDA_SCAN_DIR/"
    cp "$POLICY_FILE" "$PIP_SCAN_DIR/"
fi
touch "$CONDA_REQ"
touch "$PIP_REQ"

while IFS= read -r line; do
    if [[ $line =~ ^#.*$ ]] || [[ -z $line ]]; then
        # Skip comments and empty lines
        continue
    elif [[ $line == *"=pypi_0"* ]]; then
        # Pip package - convert to pip requirements format
        package_info=$(echo "$line" | sed 's/=/==/;s/=pypi_0//')
        echo "$package_info" >>"$PIP_REQ"
    else
        # Conda package - convert to pip requirements format
        package_info=$(echo "$line" | awk -F= '{print $1"=="$2}')
        echo "$package_info" >>"$CONDA_REQ"
    fi
done <"$FULL_EXPORT"
rm -f "$FULL_EXPORT"

# Show what we found
CONDA_COUNT=$(wc -l <"$CONDA_REQ" | tr -d ' ')
PIP_COUNT=$(wc -l <"$PIP_REQ" | tr -d ' ')
echo "Package separation results:"
echo "  Conda packages: $CONDA_COUNT"
echo "  Pip packages: $PIP_COUNT"
echo ""

OVERALL_EXIT_CODE=0

# --- Scan 1: Conda packages ---
if [ -s "$CONDA_REQ" ]; then
    echo "=== Scan 1: Conda-installed packages ($CONDA_COUNT) ==="
    echo ""
    CONDA_OUTPUT=$(safety scan --target "$CONDA_SCAN_DIR" 2>&1)
    ret=$?
    echo "$CONDA_OUTPUT"
    echo "potentially buggy return value: $ret"

    if check_vulnerabilities "$CONDA_OUTPUT"; then
        echo "✓ No unignored security vulnerabilities found in conda packages"
    else
        echo "⚠ Unignored security vulnerabilities detected in conda packages!"
        OVERALL_EXIT_CODE=1
    fi
else
    echo "=== Scan 1: No conda packages found to scan ==="
fi
echo ""

# --- Scan 2: Pip packages ---
if [ -s "$PIP_REQ" ]; then
    echo "=== Scan 2: Pip-installed packages ($PIP_COUNT) ==="
    echo ""
    PIP_OUTPUT=$(safety scan --target "$PIP_SCAN_DIR" 2>&1)
    ret=$?
    echo "$PIP_OUTPUT"
    echo "potentially buggy return value: $ret"

    if check_vulnerabilities "$PIP_OUTPUT"; then
        echo "✓ No unignored security vulnerabilities found in pip packages"
    else
        echo "⚠ Unignored security vulnerabilities detected in pip packages!"
        OVERALL_EXIT_CODE=1
    fi
else
    echo "=== Scan 2: No pip packages found to scan ==="
fi
echo ""

# --- Scan 3: pyproject.toml ---
if [ -f "pyproject.toml" ]; then
    echo "=== Scan 3: pyproject.toml dependencies ==="
    echo ""
    PYPROJECT_OUTPUT=$(safety scan --target . 2>&1)
    ret=$?
    echo "$PYPROJECT_OUTPUT"
    echo "potentially buggy return value: $ret"

    if check_vulnerabilities "$PYPROJECT_OUTPUT"; then
        echo "✓ No unignored security vulnerabilities found in pyproject.toml"
    else
        echo "⚠ Unignored security vulnerabilities detected in pyproject.toml!"
        OVERALL_EXIT_CODE=1
    fi
else
    echo "=== Scan 3: No pyproject.toml found to scan ==="
fi

# Exit with error if any scan found vulnerabilities
if [ $OVERALL_EXIT_CODE -ne 0 ]; then
    exit $OVERALL_EXIT_CODE
fi

# Cleanup is handled by the trap
exit 0
