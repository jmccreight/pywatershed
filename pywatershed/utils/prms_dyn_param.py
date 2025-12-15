"""PRMS Dynamic Parameter File Reader/Writer.

This module provides functionality to read, subset, and write PRMS dynamic
parameter files. These files contain time-varying parameter values that
change during a PRMS simulation.

Dynamic parameter file format:
- Header section containing metadata, ending with a line starting with "####"
- Data section with lines formatted as: year month day value1 value2 ... valueN
  where N is the number of HRUs (nhru)

Supported dynamic parameter types:
- ag_frac_dynamic: Dynamic agriculture fraction (float values)
- springfrost_dynamic: Dynamic spring frost dates (integer values)
- fallfrost_dynamic: Dynamic fall frost dates (integer values)
"""

import pathlib as pl
from typing import Union

import numpy as np


class PrmsDynamicParameter:
    """Class for reading, subsetting, and writing PRMS dynamic parameter files.

    Attributes:
        header_lines: List of header lines from the file
        dates: numpy array of shape (n_times, 3) containing [year, month, day]
        data: numpy array of shape (n_times, nhru) containing parameter values
        dtype: Data type of the values ('float' or 'int')
        nhru: Number of HRUs in the data
        date_separator: Separator string between date and first value (e.g., " " or "  ")
        line_ending: Line ending style ("\n" for Unix, "\r\n" for Windows)
    """

    def __init__(
        self,
        header_lines: list[str],
        dates: np.ndarray,
        data: np.ndarray,
        dtype: str = "float",
        date_separator: str = " ",
        line_ending: str = "\n",
    ) -> None:
        """Initialize a PrmsDynamicParameter object.

        Args:
            header_lines: List of header lines from the file
            dates: numpy array of shape (n_times, 3) with [year, month, day]
            data: numpy array of shape (n_times, nhru) with parameter values
            dtype: Data type of values, either 'float' or 'int'
            date_separator: Separator between date and first value
            line_ending: Line ending style ("\n" for Unix, "\r\n" for Windows)
        """
        self.header_lines = header_lines
        self.dates = dates
        self.data = data
        self.dtype = dtype
        self.nhru = data.shape[1] if len(data.shape) > 1 else 0
        self.date_separator = date_separator
        self.line_ending = line_ending

    @staticmethod
    def load(
        file_path: Union[str, pl.Path],
        dtype: str = "float",
    ) -> "PrmsDynamicParameter":
        """Load a PRMS dynamic parameter file.

        Args:
            file_path: Path to the dynamic parameter file
            dtype: Data type of the values ('float' or 'int')

        Returns:
            PrmsDynamicParameter object containing the file data
        """
        file_path = pl.Path(file_path)

        # Detect line ending style by reading raw bytes
        with open(file_path, "rb") as f:
            raw_content = f.read(4096)  # Read first 4KB to detect line endings
        if b"\r\n" in raw_content:
            line_ending = "\r\n"
        elif b"\r" in raw_content:
            line_ending = "\r"
        else:
            line_ending = "\n"

        header_lines = []
        data_lines = []

        with open(file_path, "r") as f:
            in_header = True
            for line in f:
                line = line.rstrip("\n\r")
                if in_header:
                    # Check if this line starts with a year (4-digit number)
                    # which indicates data, not header
                    parts = line.split()
                    if (
                        len(parts) >= 4
                        and parts[0].isdigit()
                        and len(parts[0]) == 4
                    ):
                        # This is a data line, header is done
                        in_header = False
                        data_lines.append(line)
                    elif line.startswith("####"):
                        # Traditional header end marker
                        header_lines.append(line)
                        in_header = False
                    else:
                        header_lines.append(line)
                else:
                    # Skip empty lines in data section
                    if line.strip():
                        data_lines.append(line)

        # Detect the separator between date and first value from first data line
        date_separator = " "
        if data_lines:
            first_line = data_lines[0]
            # Find position after the day (third whitespace-separated token)
            parts = first_line.split()
            if len(parts) >= 4:
                # Find where the day ends and count spaces until first value
                day_str = parts[2]
                # Find the day in the original line
                pos = 0
                tokens_found = 0
                while tokens_found < 3 and pos < len(first_line):
                    # Skip whitespace
                    while pos < len(first_line) and first_line[pos] == " ":
                        pos += 1
                    # Skip token
                    while pos < len(first_line) and first_line[pos] != " ":
                        pos += 1
                    tokens_found += 1
                # Now pos is at the end of day, count spaces
                space_count = 0
                while pos < len(first_line) and first_line[pos] == " ":
                    space_count += 1
                    pos += 1
                if space_count > 0:
                    date_separator = " " * space_count

        # Parse data lines
        dates = []
        data = []

        for line in data_lines:
            parts = line.split()
            if len(parts) < 4:
                continue

            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            dates.append([year, month, day])

            if dtype == "int":
                values = [int(x) for x in parts[3:]]
            else:
                values = [float(x) for x in parts[3:]]
            data.append(values)

        dates_array = np.array(dates, dtype=np.int32)
        if dtype == "int":
            data_array = np.array(data, dtype=np.int32)
        else:
            data_array = np.array(data, dtype=np.float64)

        return PrmsDynamicParameter(
            header_lines=header_lines,
            dates=dates_array,
            data=data_array,
            dtype=dtype,
            date_separator=date_separator,
            line_ending=line_ending,
        )

    def subset(
        self,
        hru_indices: np.ndarray,
    ) -> "PrmsDynamicParameter":
        """Subset the dynamic parameter data to specific HRU indices.

        Args:
            hru_indices: Array of HRU indices (0-based) or boolean mask to keep

        Returns:
            New PrmsDynamicParameter object with subsetted data
        """
        # Convert to numpy array if needed (e.g., from xarray DataArray)
        if hasattr(hru_indices, "values"):
            hru_indices = hru_indices.values

        # Subset the data array on the HRU dimension
        subset_data = self.data[:, hru_indices]

        # Handle boolean mask vs integer indices for nhru calculation
        if hru_indices.dtype == bool:
            new_nhru = int(hru_indices.sum())
            # Convert boolean mask to integer indices for header subsetting
            int_indices = np.where(hru_indices)[0]
        else:
            new_nhru = len(hru_indices)
            int_indices = hru_indices

        # Update the header to reflect new nhru and subset column headers
        new_header = []

        for line in self.header_lines:
            parts = line.split()
            # Check for dimension line (format: "nhru <n>" or "param_name <n>")
            if len(parts) == 2 and parts[1].isdigit():
                # Update the count
                new_header.append(f"{parts[0]} {new_nhru}")
            # Check for column header line (format: "year month day HRU 1 2 3 ...")
            elif (
                len(parts) > 4
                and parts[0].lower() == "year"
                and parts[1].lower() == "month"
                and parts[2].lower() == "day"
            ):
                # Keep the first 4 parts (year month day HRU) and subset the rest
                prefix = parts[:4]
                hru_ids = parts[4:]
                # Subset the HRU IDs based on indices
                subset_hru_ids = [
                    hru_ids[i] for i in int_indices if i < len(hru_ids)
                ]
                new_header.append(" ".join(prefix + subset_hru_ids))
            else:
                new_header.append(line)

        return PrmsDynamicParameter(
            header_lines=new_header,
            dates=self.dates.copy(),
            data=subset_data,
            dtype=self.dtype,
            date_separator=self.date_separator,
            line_ending=self.line_ending,
        )

    @staticmethod
    def _format_float(v: float) -> str:
        """Format a float value, ensuring decimal point for whole numbers."""
        s = f"{v:g}"
        if "." not in s and "e" not in s:
            s += ".0"
        return s

    def write(
        self,
        file_path: Union[str, pl.Path],
    ) -> None:
        """Write the dynamic parameter data to a file.

        Args:
            file_path: Path to write the file to
        """
        file_path = pl.Path(file_path)

        # Use binary mode to control line endings exactly
        newline = self.line_ending

        with open(file_path, "w", newline="") as f:
            # Write header
            for line in self.header_lines:
                f.write(line + newline)

            # Write data
            for i in range(len(self.dates)):
                year, month, day = self.dates[i]
                if self.dtype == "int":
                    values_str = " ".join(str(int(v)) for v in self.data[i])
                else:
                    # Use :g format for floats, but ensure whole numbers
                    # have a decimal point (e.g., "1.0" not "1", "0.0" not "0")
                    values_str = " ".join(
                        self._format_float(v) for v in self.data[i]
                    )

                f.write(
                    f"{year} {month} {day}{self.date_separator}{values_str}{newline}"
                )


# Mapping of control variable names to their associated flag names and dtypes
DYNAMIC_PARAM_CONFIG = {
    "ag_frac_dynamic": {
        "flag_name": "dyn_ag_frac_flag",
        "dtype": "float",
        "description": "Dynamic agriculture fraction",
    },
    "springfrost_dynamic": {
        "flag_name": "dyn_springfrost_flag",
        "dtype": "int",
        "description": "Dynamic spring frost dates",
    },
    "fallfrost_dynamic": {
        "flag_name": "dyn_fallfrost_flag",
        "dtype": "int",
        "description": "Dynamic fall frost dates",
    },
}


def get_dynamic_param_files_from_control(
    control_file: Union[str, pl.Path],
    control_dir: Union[str, pl.Path, None] = None,
) -> dict[str, dict]:
    """Get active dynamic parameter files from a PRMS control file.

    This function reads a PRMS control file and identifies which dynamic
    parameter files are enabled (flag = 1) and returns their paths and
    configuration.

    Args:
        control_file: Path to the PRMS control file
        control_dir: Directory containing the control file. If None, uses
            the parent directory of control_file. Relative paths in the
            control file are resolved relative to this directory.

    Returns:
        Dictionary mapping control variable names to dicts containing:
            - 'path': pathlib.Path to the dynamic parameter file
            - 'dtype': Data type ('float' or 'int')
            - 'flag_name': Name of the associated flag variable
            - 'description': Description of the parameter
    """
    control_file = pl.Path(control_file)
    if control_dir is None:
        control_dir = control_file.parent
    else:
        control_dir = pl.Path(control_dir)

    # Read control file and parse variables
    control_vars = _parse_control_file(control_file)

    active_files = {}

    for param_name, config in DYNAMIC_PARAM_CONFIG.items():
        flag_name = config["flag_name"]

        # Check if the flag is set and enabled (value = 1)
        if flag_name in control_vars:
            flag_value = control_vars[flag_name]
            if isinstance(flag_value, list):
                flag_value = flag_value[0]
            flag_value = int(flag_value)
        else:
            flag_value = 0

        if flag_value != 1:
            continue

        # Check if the file path is specified
        if param_name not in control_vars:
            continue

        file_path = control_vars[param_name]
        if isinstance(file_path, list):
            file_path = file_path[0]

        # Resolve the path relative to control directory
        file_path = pl.Path(file_path)
        if not file_path.is_absolute():
            file_path = control_dir / file_path

        if file_path.exists():
            active_files[param_name] = {
                "path": file_path,
                "dtype": config["dtype"],
                "flag_name": flag_name,
                "description": config["description"],
            }

    return active_files


def _parse_control_file(control_file: Union[str, pl.Path]) -> dict:
    """Parse a PRMS control file and return a dictionary of variables.

    Args:
        control_file: Path to the control file

    Returns:
        Dictionary mapping variable names to their values
    """
    control_vars = {}
    control_file = pl.Path(control_file)

    with open(control_file, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for variable block start (####)
        if line.startswith("####"):
            i += 1
            if i >= len(lines):
                break

            # Variable name
            var_name = lines[i].strip()
            i += 1
            if i >= len(lines):
                break

            # Number of values
            try:
                n_values = int(lines[i].strip())
            except ValueError:
                i += 1
                continue
            i += 1
            if i >= len(lines):
                break

            # Data type (1=int, 2=float, 4=string)
            try:
                dtype_code = int(lines[i].strip())
            except ValueError:
                i += 1
                continue
            i += 1

            # Read values
            values = []
            for _ in range(n_values):
                if i >= len(lines):
                    break
                val_line = lines[i].strip()
                i += 1

                if dtype_code == 1:  # integer
                    try:
                        values.append(int(val_line))
                    except ValueError:
                        values.append(val_line)
                elif dtype_code == 2:  # float
                    try:
                        values.append(float(val_line))
                    except ValueError:
                        values.append(val_line)
                else:  # string (4) or other
                    values.append(val_line)

            if len(values) == 1:
                control_vars[var_name] = values[0]
            else:
                control_vars[var_name] = values
        else:
            i += 1

    return control_vars


def subset_dynamic_param_file(
    input_file: Union[str, pl.Path],
    output_file: Union[str, pl.Path],
    hru_indices: np.ndarray,
    dtype: str = "float",
) -> None:
    """Subset a dynamic parameter file and write to a new file.

    This is a convenience function that loads, subsets, and writes a
    dynamic parameter file in one call.

    Args:
        input_file: Path to the input dynamic parameter file
        output_file: Path to write the subsetted file
        hru_indices: Array of HRU indices (0-based) to keep
        dtype: Data type of the values ('float' or 'int')
    """
    dyn_param = PrmsDynamicParameter.load(input_file, dtype=dtype)
    subset_param = dyn_param.subset(hru_indices)
    subset_param.write(output_file)


def compare_dynamic_param_files_text(
    file1: Union[str, pl.Path],
    file2: Union[str, pl.Path],
    max_diffs_to_report: int = 20,
    context_chars: int = 50,
    verbose: bool = True,
) -> dict:
    """Compare two dynamic parameter files as raw text to find formatting diffs.

    This function compares the actual text content of the files, not the
    parsed values. Useful for identifying formatting differences like
    "0" vs "0.0" or different floating point representations.

    Args:
        file1: Path to the first dynamic parameter file
        file2: Path to the second dynamic parameter file
        max_diffs_to_report: Maximum number of differences to report
        context_chars: Number of characters of context around differences
        verbose: If True, print differences to stdout

    Returns:
        Dictionary containing comparison results:
            - 'identical': bool, True if files are identical
            - 'n_line_diffs': number of lines that differ
            - 'line_diffs': list of line difference details
    """
    file1 = pl.Path(file1)
    file2 = pl.Path(file2)

    with open(file1, "r") as f:
        lines1 = f.readlines()
    with open(file2, "r") as f:
        lines2 = f.readlines()

    result = {
        "identical": True,
        "n_line_diffs": 0,
        "line_diffs": [],
        "file1_n_lines": len(lines1),
        "file2_n_lines": len(lines2),
    }

    if len(lines1) != len(lines2):
        result["identical"] = False
        if verbose:
            print(
                f"Line count differs: file1={len(lines1)}, file2={len(lines2)}"
            )

    max_lines = max(len(lines1), len(lines2))
    n_diffs_found = 0

    for i in range(max_lines):
        line1 = lines1[i].rstrip("\n\r") if i < len(lines1) else None
        line2 = lines2[i].rstrip("\n\r") if i < len(lines2) else None

        if line1 != line2:
            n_diffs_found += 1
            result["identical"] = False

            if n_diffs_found <= max_diffs_to_report:
                diff_info = {
                    "line_num": i + 1,
                    "file1": line1,
                    "file2": line2,
                }

                # Find first character position that differs
                if line1 is not None and line2 is not None:
                    min_len = min(len(line1), len(line2))
                    first_diff_pos = None
                    for j in range(min_len):
                        if line1[j] != line2[j]:
                            first_diff_pos = j
                            break
                    if first_diff_pos is None and len(line1) != len(line2):
                        first_diff_pos = min_len

                    if first_diff_pos is not None:
                        diff_info["first_diff_char"] = first_diff_pos
                        # Extract context around the difference
                        start = max(0, first_diff_pos - context_chars)
                        end = min(
                            max(len(line1), len(line2)),
                            first_diff_pos + context_chars,
                        )
                        diff_info["file1_context"] = line1[start:end]
                        diff_info["file2_context"] = line2[start:end]
                        diff_info["context_start"] = start

                result["line_diffs"].append(diff_info)

                if verbose:
                    print(f"\n--- Line {i + 1} differs ---")
                    if line1 is None:
                        print("  file1: <missing line>")
                    elif len(line1) > 200:
                        print(f"  file1: ({len(line1)} chars)")
                    else:
                        print(f"  file1: {line1}")

                    if line2 is None:
                        print("  file2: <missing line>")
                    elif len(line2) > 200:
                        print(f"  file2: ({len(line2)} chars)")
                    else:
                        print(f"  file2: {line2}")

                    if "first_diff_char" in diff_info:
                        pos = diff_info["first_diff_char"]
                        print(f"  First difference at character {pos}:")
                        ctx_start = diff_info.get("context_start", 0)
                        print(
                            f"    file1[{ctx_start}:{ctx_start + len(diff_info['file1_context'])}]: "
                            f"'{diff_info['file1_context']}'"
                        )
                        print(
                            f"    file2[{ctx_start}:{ctx_start + len(diff_info['file2_context'])}]: "
                            f"'{diff_info['file2_context']}'"
                        )

    result["n_line_diffs"] = n_diffs_found

    if verbose:
        if n_diffs_found > max_diffs_to_report:
            print(
                f"\n... and {n_diffs_found - max_diffs_to_report} more "
                f"line differences (total: {n_diffs_found})"
            )
        if result["identical"]:
            print("Files are identical.")
        else:
            print(f"\nSummary: {n_diffs_found} lines differ.")

    return result


def compare_dynamic_param_files(
    file1: Union[str, pl.Path],
    file2: Union[str, pl.Path],
    dtype: str = "float",
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_diffs_to_report: int = 20,
    verbose: bool = True,
) -> dict:
    """Compare two dynamic parameter files and report differences.

    Args:
        file1: Path to the first dynamic parameter file
        file2: Path to the second dynamic parameter file
        dtype: Data type of the values ('float' or 'int')
        rtol: Relative tolerance for float comparison
        atol: Absolute tolerance for float comparison
        max_diffs_to_report: Maximum number of value differences to report
        verbose: If True, print differences to stdout

    Returns:
        Dictionary containing comparison results:
            - 'identical': bool, True if files are identical
            - 'header_match': bool, True if headers match
            - 'dates_match': bool, True if all dates match
            - 'data_match': bool, True if all data values match
            - 'nhru_match': bool, True if nhru dimensions match
            - 'n_times_match': bool, True if number of time steps match
            - 'header_diffs': list of header line differences
            - 'date_diffs': list of date differences
            - 'value_diffs': list of value differences (up to max_diffs_to_report)
            - 'n_value_diffs': total number of value differences
    """
    file1 = pl.Path(file1)
    file2 = pl.Path(file2)

    dyn1 = PrmsDynamicParameter.load(file1, dtype=dtype)
    dyn2 = PrmsDynamicParameter.load(file2, dtype=dtype)

    result = {
        "identical": True,
        "header_match": True,
        "dates_match": True,
        "data_match": True,
        "nhru_match": True,
        "n_times_match": True,
        "header_diffs": [],
        "date_diffs": [],
        "value_diffs": [],
        "n_value_diffs": 0,
    }

    # Compare nhru
    if dyn1.nhru != dyn2.nhru:
        result["nhru_match"] = False
        result["identical"] = False
        if verbose:
            print(f"nhru mismatch: file1={dyn1.nhru}, file2={dyn2.nhru}")

    # Compare number of time steps
    if len(dyn1.dates) != len(dyn2.dates):
        result["n_times_match"] = False
        result["identical"] = False
        if verbose:
            print(
                f"n_times mismatch: file1={len(dyn1.dates)}, "
                f"file2={len(dyn2.dates)}"
            )

    # Compare headers
    if dyn1.header_lines != dyn2.header_lines:
        result["header_match"] = False
        result["identical"] = False
        max_len = max(len(dyn1.header_lines), len(dyn2.header_lines))
        for i in range(max_len):
            h1 = dyn1.header_lines[i] if i < len(dyn1.header_lines) else None
            h2 = dyn2.header_lines[i] if i < len(dyn2.header_lines) else None
            if h1 != h2:
                result["header_diffs"].append(
                    {"line": i, "file1": h1, "file2": h2}
                )
                if verbose:
                    print(f"Header line {i} differs:")
                    print(f"  file1: {h1}")
                    print(f"  file2: {h2}")

    # Compare dates
    if result["n_times_match"]:
        for t in range(len(dyn1.dates)):
            if not np.array_equal(dyn1.dates[t], dyn2.dates[t]):
                result["dates_match"] = False
                result["identical"] = False
                result["date_diffs"].append(
                    {
                        "time_idx": t,
                        "file1": dyn1.dates[t].tolist(),
                        "file2": dyn2.dates[t].tolist(),
                    }
                )
                if verbose:
                    print(
                        f"Date mismatch at time {t}: "
                        f"file1={dyn1.dates[t]}, file2={dyn2.dates[t]}"
                    )

    # Compare data values
    if result["nhru_match"] and result["n_times_match"]:
        n_diffs_found = 0
        for t in range(len(dyn1.dates)):
            for h in range(dyn1.nhru):
                v1 = dyn1.data[t, h]
                v2 = dyn2.data[t, h]

                if dtype == "int":
                    values_differ = v1 != v2
                else:
                    values_differ = not np.isclose(
                        v1, v2, rtol=rtol, atol=atol
                    )

                if values_differ:
                    n_diffs_found += 1
                    result["data_match"] = False
                    result["identical"] = False

                    if n_diffs_found <= max_diffs_to_report:
                        date = dyn1.dates[t]
                        diff_info = {
                            "time_idx": t,
                            "date": f"{date[0]}-{date[1]:02d}-{date[2]:02d}",
                            "hru_idx": h,
                            "file1_value": float(v1)
                            if dtype == "float"
                            else int(v1),
                            "file2_value": float(v2)
                            if dtype == "float"
                            else int(v2),
                        }
                        if dtype == "float":
                            if v1 != 0:
                                diff_info["rel_diff"] = abs(v2 - v1) / abs(v1)
                            diff_info["abs_diff"] = abs(v2 - v1)
                        result["value_diffs"].append(diff_info)

                        if verbose:
                            print(
                                f"Value diff at time={t} "
                                f"({date[0]}-{date[1]:02d}-{date[2]:02d}), "
                                f"hru={h}: file1={v1}, file2={v2}"
                            )

        result["n_value_diffs"] = n_diffs_found
        if verbose and n_diffs_found > max_diffs_to_report:
            print(
                f"... and {n_diffs_found - max_diffs_to_report} more "
                f"value differences (total: {n_diffs_found})"
            )

    if verbose:
        if result["identical"]:
            print("Files are identical.")
        else:
            print("\nSummary: Files differ.")
            print(f"  Header match: {result['header_match']}")
            print(f"  Dates match: {result['dates_match']}")
            print(f"  Data match: {result['data_match']}")
            print(f"  Total value differences: {result['n_value_diffs']}")

    return result
