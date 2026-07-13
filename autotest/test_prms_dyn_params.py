"""Tests for PRMS dynamic parameter file reading and writing.

This module tests the PrmsDynamicParameter class which handles reading,
subsetting, and writing PRMS dynamic parameter files.
"""

import pathlib as pl
import tempfile

import numpy as np
import pytest

from pywatershed.utils import (
    PrmsDynamicParameter,
    compare_dynamic_param_files_text,
)


class TestPrmsDynamicParameter:
    """Tests for PrmsDynamicParameter class."""

    def create_sample_dyn_param_file(
        self, file_path: pl.Path, dtype: str = "float"
    ) -> dict:
        """Create a sample dynamic parameter file for testing.

        Args:
            file_path: Path to write the sample file
            dtype: Data type, either 'float' or 'int'

        Returns:
            dict with expected values for verification
        """
        nhru = 5
        n_times = 3

        header_lines = [
            "Dynamic parameter test file",
            "Created for testing purposes",
            f"nhru {nhru}",
            "####",
        ]

        # Create sample dates and data
        dates = np.array(
            [
                [2020, 1, 15],
                [2020, 6, 1],
                [2021, 1, 1],
            ],
            dtype=np.int32,
        )

        if dtype == "int":
            data = np.array(
                [
                    [100, 110, 120, 130, 140],
                    [200, 210, 220, 230, 240],
                    [300, 310, 320, 330, 340],
                ],
                dtype=np.int32,
            )
        else:
            data = np.array(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.15, 0.25, 0.35, 0.45, 0.55],
                    [0.2, 0.3, 0.4, 0.5, 0.6],
                ],
                dtype=np.float64,
            )

        # Write the file
        with open(file_path, "w") as f:
            for line in header_lines:
                f.write(line + "\n")

            for i in range(n_times):
                year, month, day = dates[i]
                if dtype == "int":
                    values_str = " ".join(str(int(v)) for v in data[i])
                else:
                    values_str = " ".join(f"{v:.6g}" for v in data[i])
                f.write(f"{year} {month} {day} {values_str}\n")

        return {
            "header_lines": header_lines,
            "dates": dates,
            "data": data,
            "nhru": nhru,
            "n_times": n_times,
            "dtype": dtype,
        }

    def test_load_float(self):
        """Test loading a dynamic parameter file with float values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pl.Path(tmpdir)
            file_path = tmpdir / "test_float.dyn"

            expected = self.create_sample_dyn_param_file(
                file_path, dtype="float"
            )

            dyn_param = PrmsDynamicParameter.load(file_path, dtype="float")

            assert dyn_param.nhru == expected["nhru"]
            assert dyn_param.dtype == "float"
            assert len(dyn_param.dates) == expected["n_times"]
            assert dyn_param.data.shape == (
                expected["n_times"],
                expected["nhru"],
            )

            # Check dates
            np.testing.assert_array_equal(dyn_param.dates, expected["dates"])

            # Check data values (with tolerance for float)
            np.testing.assert_allclose(
                dyn_param.data, expected["data"], rtol=1e-5
            )

    def test_load_int(self):
        """Test loading a dynamic parameter file with integer values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pl.Path(tmpdir)
            file_path = tmpdir / "test_int.dyn"

            expected = self.create_sample_dyn_param_file(
                file_path, dtype="int"
            )

            dyn_param = PrmsDynamicParameter.load(file_path, dtype="int")

            assert dyn_param.nhru == expected["nhru"]
            assert dyn_param.dtype == "int"
            assert len(dyn_param.dates) == expected["n_times"]

            # Check dates
            np.testing.assert_array_equal(dyn_param.dates, expected["dates"])

            # Check data values (exact match for integers)
            np.testing.assert_array_equal(dyn_param.data, expected["data"])

    def test_daily_data_array_forward_fill_mid_window(self):
        """Forward-fill when daily_start_date falls between file dates.

        The values in effect at daily_start_date come from the most recent
        file date at or before it, as PRMS applies dynamic updates.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pl.Path(tmpdir)
            file_path = tmpdir / "test_int.dyn"
            expected = self.create_sample_dyn_param_file(
                file_path, dtype="int"
            )

            # start between the first (2020-01-15) and second (2020-06-01)
            # file dates
            dyn_param = PrmsDynamicParameter.load(file_path, dtype="int")
            dyn_param.daily_start_date = "2020-03-01"
            dyn_param.daily_end_date = "2020-06-02"
            daily = dyn_param.daily_data_array

            assert daily.shape == (94, expected["nhru"])
            # 2020-03-01 through 2020-05-31 (92 days) forward-fill the
            # 2020-01-15 values from before the window
            np.testing.assert_array_equal(
                daily.values[:92, :],
                np.tile(expected["data"][0, :], (92, 1)),
            )
            # 2020-06-01 and after take the second entry
            np.testing.assert_array_equal(
                daily.values[92:, :],
                np.tile(expected["data"][1, :], (2, 1)),
            )

    def test_daily_data_array_fill_before_first_date(self):
        """Days before the first date in the file remain fill values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pl.Path(tmpdir)
            file_path = tmpdir / "test_int.dyn"
            expected = self.create_sample_dyn_param_file(
                file_path, dtype="int"
            )

            # start two days before the first file date (2020-01-15)
            dyn_param = PrmsDynamicParameter.load(file_path, dtype="int")
            dyn_param.daily_start_date = "2020-01-13"
            dyn_param.daily_end_date = "2020-01-16"
            daily = dyn_param.daily_data_array

            assert daily.shape == (4, expected["nhru"])
            # 2020-01-13 and 2020-01-14 precede all file dates: fill values
            np.testing.assert_array_equal(
                daily.values[:2, :],
                np.full((2, expected["nhru"]), -999, dtype=np.int32),
            )
            # 2020-01-15 and 2020-01-16 take the first entry
            np.testing.assert_array_equal(
                daily.values[2:, :],
                np.tile(expected["data"][0, :], (2, 1)),
            )

    def test_round_trip_float(self):
        """Test reading and writing a float file produces identical results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pl.Path(tmpdir)
            original_file = tmpdir / "original_float.dyn"
            output_file = tmpdir / "output_float.dyn"

            expected = self.create_sample_dyn_param_file(
                original_file, dtype="float"
            )

            # Load original
            dyn_param = PrmsDynamicParameter.load(original_file, dtype="float")

            # Write to new file
            dyn_param.write(output_file)

            # Load the written file
            reloaded = PrmsDynamicParameter.load(output_file, dtype="float")

            # Compare
            assert reloaded.nhru == dyn_param.nhru
            assert reloaded.dtype == dyn_param.dtype
            np.testing.assert_array_equal(reloaded.dates, dyn_param.dates)
            np.testing.assert_allclose(
                reloaded.data, dyn_param.data, rtol=1e-5
            )

            # Also compare with original expected values
            np.testing.assert_array_equal(reloaded.dates, expected["dates"])
            np.testing.assert_allclose(
                reloaded.data, expected["data"], rtol=1e-5
            )

    def test_round_trip_int(self):
        """Test reading and writing an int file produces identical results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pl.Path(tmpdir)
            original_file = tmpdir / "original_int.dyn"
            output_file = tmpdir / "output_int.dyn"

            expected = self.create_sample_dyn_param_file(
                original_file, dtype="int"
            )

            # Load original
            dyn_param = PrmsDynamicParameter.load(original_file, dtype="int")

            # Write to new file
            dyn_param.write(output_file)

            # Load the written file
            reloaded = PrmsDynamicParameter.load(output_file, dtype="int")

            # Compare
            assert reloaded.nhru == dyn_param.nhru
            assert reloaded.dtype == dyn_param.dtype
            np.testing.assert_array_equal(reloaded.dates, dyn_param.dates)
            np.testing.assert_array_equal(reloaded.data, dyn_param.data)

            # Also compare with original expected values
            np.testing.assert_array_equal(reloaded.dates, expected["dates"])
            np.testing.assert_array_equal(reloaded.data, expected["data"])

    def test_subset(self):
        """Test subsetting a dynamic parameter to specific HRU indices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pl.Path(tmpdir)
            file_path = tmpdir / "test_subset.dyn"

            expected = self.create_sample_dyn_param_file(
                file_path, dtype="float"
            )

            dyn_param = PrmsDynamicParameter.load(file_path, dtype="float")

            # Subset to HRUs 1 and 3 (0-based indices)
            hru_indices = np.array([1, 3])
            subset = dyn_param.subset(hru_indices)

            assert subset.nhru == 2
            assert subset.data.shape == (expected["n_times"], 2)

            # Check that the correct columns were selected
            expected_subset_data = expected["data"][:, hru_indices]
            np.testing.assert_allclose(
                subset.data, expected_subset_data, rtol=1e-5
            )

            # Dates should be unchanged
            np.testing.assert_array_equal(subset.dates, dyn_param.dates)

    def test_subset_round_trip(self):
        """Test that subsetting and round-tripping works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pl.Path(tmpdir)
            original_file = tmpdir / "original.dyn"
            subset_file = tmpdir / "subset.dyn"

            expected = self.create_sample_dyn_param_file(
                original_file, dtype="float"
            )

            # Load, subset, write, reload
            dyn_param = PrmsDynamicParameter.load(original_file, dtype="float")
            hru_indices = np.array([0, 2, 4])
            subset = dyn_param.subset(hru_indices)
            subset.write(subset_file)
            reloaded = PrmsDynamicParameter.load(subset_file, dtype="float")

            # Verify
            assert reloaded.nhru == 3
            expected_data = expected["data"][:, hru_indices]
            np.testing.assert_allclose(reloaded.data, expected_data, rtol=1e-5)
            np.testing.assert_array_equal(reloaded.dates, expected["dates"])

    def test_header_preservation(self):
        """Test that header lines are preserved through round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pl.Path(tmpdir)
            original_file = tmpdir / "original.dyn"
            output_file = tmpdir / "output.dyn"

            self.create_sample_dyn_param_file(original_file, dtype="float")

            dyn_param = PrmsDynamicParameter.load(original_file, dtype="float")
            original_header = dyn_param.header_lines.copy()

            dyn_param.write(output_file)
            reloaded = PrmsDynamicParameter.load(output_file, dtype="float")

            assert reloaded.header_lines == original_header

    def test_empty_lines_ignored(self):
        """Test that empty lines in the data section are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pl.Path(tmpdir)
            file_path = tmpdir / "test_empty_lines.dyn"

            # Create file with empty lines in data section
            with open(file_path, "w") as f:
                f.write("Test file\n")
                f.write("####\n")
                f.write("2020 1 1 0.1 0.2 0.3\n")
                f.write("\n")  # Empty line
                f.write("2020 6 1 0.4 0.5 0.6\n")
                f.write("   \n")  # Whitespace-only line
                f.write("2021 1 1 0.7 0.8 0.9\n")

            dyn_param = PrmsDynamicParameter.load(file_path, dtype="float")

            assert len(dyn_param.dates) == 3
            assert dyn_param.nhru == 3

    def test_file_comparison_utility(self):
        """Test comparing two dynamic parameter files line by line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pl.Path(tmpdir)
            file1 = tmpdir / "file1.dyn"
            file2 = tmpdir / "file2.dyn"

            # Create identical files
            self.create_sample_dyn_param_file(file1, dtype="float")

            # Load and write to create file2
            dyn_param = PrmsDynamicParameter.load(file1, dtype="float")
            dyn_param.write(file2)

            # Read both files and compare content
            with open(file1, "r") as f1, open(file2, "r") as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()

            # Header should match exactly
            # Find header end in both files
            header_end_1 = next(
                i for i, line in enumerate(lines1) if line.startswith("####")
            )
            header_end_2 = next(
                i for i, line in enumerate(lines2) if line.startswith("####")
            )

            assert lines1[: header_end_1 + 1] == lines2[: header_end_2 + 1]

            # Data section should have same number of lines
            data_lines_1 = [
                line.strip()
                for line in lines1[header_end_1 + 1 :]
                if line.strip()
            ]
            data_lines_2 = [
                line.strip()
                for line in lines2[header_end_2 + 1 :]
                if line.strip()
            ]

            assert len(data_lines_1) == len(data_lines_2)

            # Parse and compare data values
            for line1, line2 in zip(data_lines_1, data_lines_2):
                parts1 = line1.split()
                parts2 = line2.split()

                # Dates should match exactly
                assert parts1[:3] == parts2[:3]

                # Values should be close
                vals1 = [float(x) for x in parts1[3:]]
                vals2 = [float(x) for x in parts2[3:]]
                np.testing.assert_allclose(vals1, vals2, rtol=1e-5)


class TestRealFiles:
    """Tests using real dynamic parameter files.

    These tests are skipped if REAL_FLOAT_FILE and REAL_INT_FILE are None.
    Set the module-level variables to actual file paths to run these tests.

    Set PERSIST_OUTPUT_DIR to a directory path to keep output files after
    tests complete. Otherwise, temporary directories are used and cleaned up.
    """

    # List of domains that have the required dynamic parameter files
    SUPPORTED_DOMAINS = ["fgr_ag_2yr"]

    @staticmethod
    def _get_output_dir(subdir: str = None):
        """Get output directory using temporary directory.

        Args:
            subdir: Optional subdirectory name within the output dir

        Returns:
            tuple: (output_dir_path, context_manager)
            Context manager should be used with 'with'.
        """
        return None, tempfile.TemporaryDirectory()

    def test_real_float_file_round_trip(self, simulation):
        """Test round-trip with a real float dynamic parameter file."""
        domain_name = simulation["name"].split(":")[0]
        if domain_name not in self.SUPPORTED_DOMAINS:
            pytest.skip(
                f"test_prms_dyn_params only runs on domains: "
                f"{self.SUPPORTED_DOMAINS}"
            )

        file_path = simulation["dir"] / "dyn_ag_frac.param"
        if not file_path.exists():
            pytest.skip(f"Real file not found: {file_path}")

        output_dir, temp_ctx = self._get_output_dir("float_round_trip")

        try:
            if temp_ctx is not None:
                temp_ctx.__enter__()
                output_dir = pl.Path(temp_ctx.name)

            output_file = output_dir / f"output_{file_path.name}"

            # Load original
            print(f"Loading real file: {file_path}")
            dyn_param = PrmsDynamicParameter.load(file_path, dtype="float")
            print(f"  nhru: {dyn_param.nhru}")
            print(f"  n_times: {len(dyn_param.dates)}")
            print(f"  dtype: {dyn_param.dtype}")

            # Write to new file
            dyn_param.write(output_file)
            print(f"Wrote to: {output_file}")

            # Load the written file
            reloaded = PrmsDynamicParameter.load(output_file, dtype="float")

            # Compare data values
            assert reloaded.nhru == dyn_param.nhru
            assert reloaded.dtype == dyn_param.dtype
            assert len(reloaded.dates) == len(dyn_param.dates)
            np.testing.assert_array_equal(reloaded.dates, dyn_param.dates)
            np.testing.assert_allclose(
                reloaded.data, dyn_param.data, rtol=1e-5
            )
            print("Data round-trip successful!")

            # Compare text formatting
            text_result = compare_dynamic_param_files_text(
                file_path, output_file, verbose=False
            )
            assert text_result["identical"], (
                f"Text files differ: {text_result['n_line_diffs']} "
                "lines differ"
            )
            print("Text round-trip successful!")

        finally:
            if temp_ctx is not None:
                temp_ctx.__exit__(None, None, None)

    def test_real_int_file_round_trip(self, simulation):
        """Test round-trip with a real integer dynamic parameter file."""
        domain_name = simulation["name"].split(":")[0]
        if domain_name not in self.SUPPORTED_DOMAINS:
            pytest.skip(
                f"test_prms_dyn_params only runs on domains: "
                f"{self.SUPPORTED_DOMAINS}"
            )

        file_path = simulation["dir"] / "spring_frost.dyn"
        if not file_path.exists():
            pytest.skip(f"Real file not found: {file_path}")

        output_dir, temp_ctx = self._get_output_dir("int_round_trip")

        try:
            if temp_ctx is not None:
                temp_ctx.__enter__()
                output_dir = pl.Path(temp_ctx.name)

            output_file = output_dir / f"output_{file_path.name}"

            # Load original
            print(f"Loading real file: {file_path}")
            dyn_param = PrmsDynamicParameter.load(file_path, dtype="int")
            print(f"  nhru: {dyn_param.nhru}")
            print(f"  n_times: {len(dyn_param.dates)}")
            print(f"  dtype: {dyn_param.dtype}")

            # Write to new file
            dyn_param.write(output_file)
            print(f"Wrote to: {output_file}")

            # Load the written file
            reloaded = PrmsDynamicParameter.load(output_file, dtype="int")

            # Compare data values
            assert reloaded.nhru == dyn_param.nhru
            assert reloaded.dtype == dyn_param.dtype
            assert len(reloaded.dates) == len(dyn_param.dates)
            np.testing.assert_array_equal(reloaded.dates, dyn_param.dates)
            np.testing.assert_array_equal(reloaded.data, dyn_param.data)
            print("Data round-trip successful!")

            # Compare text formatting
            text_result = compare_dynamic_param_files_text(
                file_path, output_file, verbose=False
            )
            assert text_result["identical"], (
                f"Text files differ: {text_result['n_line_diffs']} "
                "lines differ"
            )
            print("Text round-trip successful!")

        finally:
            if temp_ctx is not None:
                temp_ctx.__exit__(None, None, None)

    def test_real_file_subset_round_trip(self, simulation):
        """Test subset and round-trip with a real file."""
        domain_name = simulation["name"].split(":")[0]
        if domain_name not in self.SUPPORTED_DOMAINS:
            pytest.skip(
                f"test_prms_dyn_params only runs on domains: "
                f"{self.SUPPORTED_DOMAINS}"
            )

        # Try float file first
        file_path = simulation["dir"] / "dyn_ag_frac.param"
        if file_path.exists():
            dtype = "float"
        else:
            # Try int file
            file_path = simulation["dir"] / "spring_frost.dyn"
            if file_path.exists():
                dtype = "int"
            else:
                pytest.skip(
                    "No dynamic parameter files found in simulation directory"
                )

        output_dir, temp_ctx = self._get_output_dir("subset_round_trip")

        try:
            if temp_ctx is not None:
                temp_ctx.__enter__()
                output_dir = pl.Path(temp_ctx.name)

            output_file = output_dir / f"subset_{file_path.name}"

            # Load original
            print(f"Loading real file: {file_path}")
            dyn_param = PrmsDynamicParameter.load(file_path, dtype=dtype)
            print(f"  nhru: {dyn_param.nhru}")

            # Subset to first 10 HRUs (or all if fewer)
            n_subset = min(10, dyn_param.nhru)
            hru_indices = np.arange(n_subset)
            print(f"Subsetting to {n_subset} HRUs")

            subset = dyn_param.subset(hru_indices)
            assert subset.nhru == n_subset

            # Write and reload
            subset.write(output_file)
            reloaded = PrmsDynamicParameter.load(output_file, dtype=dtype)

            # Compare
            assert reloaded.nhru == n_subset
            np.testing.assert_array_equal(reloaded.dates, subset.dates)

            if dtype == "int":
                np.testing.assert_array_equal(reloaded.data, subset.data)
            else:
                np.testing.assert_allclose(
                    reloaded.data, subset.data, rtol=1e-5
                )

            print("Subset round-trip successful!")

        finally:
            if temp_ctx is not None:
                temp_ctx.__exit__(None, None, None)
