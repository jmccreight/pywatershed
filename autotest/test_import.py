import pytest


@pytest.mark.domainless
class TestImport:
    """Test that pywatershed can be imported without circular import errors."""

    def test_import_pywatershed(self):
        """Test basic import of pywatershed package."""
        import pywatershed as pws

        assert pws is not None
        assert hasattr(pws, "__version__")

    def test_import_base_modules(self):
        """Test importing base modules."""
        from pywatershed.base import Control, Model, Parameters

        assert Control is not None
        assert Parameters is not None
        assert Model is not None

    def test_import_utils_modules(self):
        """Test importing utils modules without circular imports."""
        from pywatershed.utils import ControlVariables, NetCdfRead

        assert ControlVariables is not None
        assert NetCdfRead is not None

    def test_import_analysis_modules(self):
        """Test importing analysis modules."""
        from pywatershed.analysis import ModelGraph

        assert ModelGraph is not None

    def test_import_hydrology_modules(self):
        """Test importing hydrology modules."""
        from pywatershed.hydrology import (
            PRMSCanopy,
            PRMSRunoff,
            PRMSSnow,
        )

        assert PRMSCanopy is not None
        assert PRMSRunoff is not None
        assert PRMSSnow is not None

    def test_no_circular_import_control_utils(self):
        """
        Regression test for circular import between base.control and utils.

        This specifically tests the fix for the circular import:
        base.control -> utils.ControlVariables -> utils.__init__ ->
        utils.cbh_utils -> base.meta -> base.__init__ -> base.adapter ->
        base.control (circular!)

        The fix uses lazy import of ControlVariables in base.control.
        """
        # Import in the order that previously caused circular import
        from pywatershed.base import Control
        from pywatershed.utils import ControlVariables

        assert Control is not None
        assert ControlVariables is not None

    def test_optional_dependency_import(self):
        """Test that import_optional_dependency can be imported directly."""
        from pywatershed.utils.optional_import import (
            import_optional_dependency,
        )

        assert import_optional_dependency is not None
        assert callable(import_optional_dependency)

    def test_import_without_optional_deps(self, monkeypatch):
        """
        Test that pywatershed can be imported even when optional
        dependencies are missing (simulating CI environment).
        """
        import builtins
        import sys

        # Mock out the optional dependencies by making them unimportable
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            # Block common optional dependencies
            blocked_modules = [
                "panel",
                "geoviews",
                "holoviews",
                "geopandas",
                "cartopy",
                "bokeh",
                "hvplot",
                "folium",
                "pydot",
                "shapely",
            ]
            if any(name.startswith(mod) for mod in blocked_modules):
                raise ModuleNotFoundError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        # Apply the mock
        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Clear any already-imported pywatershed modules
        modules_to_clear = [
            key for key in sys.modules.keys() if key.startswith("pywatershed")
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # This should succeed even without optional dependencies
        import pywatershed as pws

        assert pws is not None
        assert hasattr(pws, "__version__")

        # Verify that trying to use a class that needs optional deps fails
        # gracefully (should fail when instantiating, not when importing)
        from pywatershed.analysis import HRUComparisonPanel

        # The class should be importable
        assert HRUComparisonPanel is not None
