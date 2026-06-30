import os
from types import SimpleNamespace

import pandas as pd
import pytest
from shapely.geometry import Point

import dms_datastore_ui.datastore_uimgr as datastore_uimgr
from dms_datastore_ui.datastore_uimgr import (
    DatastoreCatalogBuilder,
    DatastoreDataReference,
    DatastoreFilepathReader,
    DatastoreUIMgr,
)
from dms_datastore_ui.map_inventory_explorer import StationDatastore
from dvue.catalog import DataCatalog, DataReference, InMemoryDataReferenceReader


@pytest.fixture()
def timeseries_df():
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    return pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=idx)


@pytest.fixture()
def inventory_row_dict():
    return {
        "filename": "usgs_anh_11303500_flow_2024.csv",
        "station_id": "anh",
        "subloc": "",
        "name": "Andrus",
        "param": "flow",
        "unit": "cfs",
        "min_year": 2024,
        "max_year": 2024,
        "agency": "usgs",
        "agency_id_dbase": "11303500",
        "x": 625000.0,
        "y": 4200000.0,
    }


class TestDatastoreFilepathReader:
    def test_load_reads_filepath(self, monkeypatch, timeseries_df):
        called = {}

        def fake_read_ts(path):
            called["path"] = path
            return timeseries_df

        monkeypatch.setattr(datastore_uimgr, "read_ts", fake_read_ts)
        reader = DatastoreFilepathReader()

        result = reader.load(filepath=r"C:\repo\screened\example.csv")

        assert called["path"] == r"C:\repo\screened\example.csv"
        pd.testing.assert_frame_equal(result, timeseries_df)

    def test_load_requires_filepath(self):
        reader = DatastoreFilepathReader()

        with pytest.raises(KeyError, match="filepath"):
            reader.load(repo_level="screened", filename="example.csv")


class TestDatastoreDataReference:
    def test_requires_filepath(self):
        with pytest.raises(ValueError, match="filepath"):
            DatastoreDataReference(name="missing-filepath")

    def test_from_inventory_row_normalizes_metadata(self, inventory_row_dict):
        row = pd.Series({**inventory_row_dict, "subloc": float("nan")})
        ref = DatastoreDataReference.from_inventory_row(
            row=row,
            repo_dir=r"C:\repo",
            repo_level="screened",
        )

        assert isinstance(ref, DatastoreDataReference)
        assert os.path.normpath(ref.filepath) == os.path.normpath(
            r"C:\repo\screened\usgs_anh_11303500_flow_2024.csv"
        )
        assert ref.station_id == "anh"
        assert ref.subloc == ""
        assert ref.parameter == "flow"
        assert ref.unit == "cfs"
        assert isinstance(ref.geometry, Point)

    def test_get_data_uses_default_filepath_reader(
        self, monkeypatch, timeseries_df, inventory_row_dict
    ):
        def fake_read_ts(path):
            assert path.endswith(inventory_row_dict["filename"])
            return timeseries_df

        monkeypatch.setattr(datastore_uimgr, "read_ts", fake_read_ts)

        ref = DatastoreDataReference.from_inventory_row(
            row=pd.Series(inventory_row_dict),
            repo_dir=r"C:\repo",
            repo_level="screened",
        )

        result = ref.getData()
        pd.testing.assert_frame_equal(result, timeseries_df)


class TestDatastoreCatalogBuilder:
    def test_can_handle(self):
        builder = DatastoreCatalogBuilder()
        station_datastore = StationDatastore.__new__(StationDatastore)

        assert builder.can_handle(station_datastore)
        assert not builder.can_handle(object())

    def test_build_returns_datastore_data_references(self, inventory_row_dict):
        builder = DatastoreCatalogBuilder()

        common_cols = {
            k: inventory_row_dict[k]
            for k in [
                "station_id",
                "subloc",
                "name",
                "unit",
                "param",
                "min_year",
                "max_year",
                "agency",
                "agency_id_dbase",
                "x",
                "y",
            ]
        }
        station_df = pd.DataFrame([common_cols])
        dataset_df = pd.DataFrame([inventory_row_dict])

        source = SimpleNamespace(
            dir=r"C:\repo",
            repo_level=["screened"],
            df_station_inventory=station_df,
            df_dataset_inventory=dataset_df,
            caching_read_ts=datastore_uimgr.read_ts,
        )

        refs = builder.build(source)

        assert len(refs) == 1
        ref = refs[0]
        assert isinstance(ref, DatastoreDataReference)
        assert ref.get_attribute("filename") == inventory_row_dict["filename"]
        assert ref.get_attribute("repo_level") == "screened"
        assert os.path.normpath(ref.filepath) == os.path.normpath(
            r"C:\repo\screened\usgs_anh_11303500_flow_2024.csv"
        )

    def test_build_shares_reader_instance(self, inventory_row_dict):
        builder = DatastoreCatalogBuilder()

        common_cols = {
            k: inventory_row_dict[k]
            for k in [
                "station_id",
                "subloc",
                "name",
                "unit",
                "param",
                "min_year",
                "max_year",
                "agency",
                "agency_id_dbase",
                "x",
                "y",
            ]
        }
        station_df = pd.DataFrame([common_cols, {**common_cols, "station_id": "bks"}])
        dataset_df = pd.DataFrame(
            [
                inventory_row_dict,
                {
                    **inventory_row_dict,
                    "filename": "usgs_bks_11300000_flow_2024.csv",
                    "station_id": "bks",
                },
            ]
        )
        source = SimpleNamespace(
            dir=r"C:\repo",
            repo_level=["screened"],
            df_station_inventory=station_df,
            df_dataset_inventory=dataset_df,
            caching_read_ts=datastore_uimgr.read_ts,
        )

        refs = builder.build(source)
        readers = {id(ref._reader_instance) for ref in refs}

        assert len(refs) == 2
        assert len(readers) == 1


class TestMixedCatalog:
    def test_mixed_references_work_in_single_catalog(self, monkeypatch, timeseries_df):
        other_df = timeseries_df * 10.0

        monkeypatch.setattr(datastore_uimgr, "read_ts", lambda path: timeseries_df)

        ds_ref = DatastoreDataReference(
            name="ds_flow",
            cache=False,
            filepath=r"C:\repo\screened\flow.csv",
            repo_level="screened",
            filename="flow.csv",
            station_id="anh",
            subloc="",
            station_name="Andrus",
            param="flow",
            unit="cfs",
            min_year=2024,
            max_year=2024,
            agency="usgs",
            agency_id_dbase="11303500",
            x=625000.0,
            y=4200000.0,
            geometry=Point(625000.0, 4200000.0),
        )
        mem_ref = DataReference(
            reader=InMemoryDataReferenceReader(other_df),
            name="mem_flow",
            station_id="anh",
            param="flow",
        )

        catalog = DataCatalog(primary_key=["station_id", "subloc", "param", "repo_level"]).add(ds_ref).add(mem_ref)

        assert set(catalog.list_names()) == {"ds_flow", "mem_flow"}
        assert len(catalog.search(param="flow")) == 2
        catalog_df = catalog.to_dataframe()
        assert "param" in catalog_df.columns

        sum_ref = ds_ref + mem_ref
        result = sum_ref.getData()
        expected = pd.DataFrame({"result": (timeseries_df + other_df)["value"]})
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Helpers shared by TestGetDataReference
# ---------------------------------------------------------------------------

def _make_ds_ref(name, station_id, subloc, param, filepath=r"C:\repo\screened\flow.csv"):
    return DatastoreDataReference(
        name=name,
        cache=False,
        filepath=filepath,
        repo_level="screened",
        filename=os.path.basename(filepath),
        station_id=station_id,
        subloc=subloc,
        station_name="Test Station",
        param=param,
        unit="cfs",
        min_year=2024,
        max_year=2024,
        agency="usgs",
        agency_id_dbase="11303500",
        x=625000.0,
        y=4200000.0,
        geometry=Point(625000.0, 4200000.0),
    )


def _make_mock_mgr(refs, unit_conversion=False):
    """Return a minimal object that satisfies DatastoreUIMgr.get_data_reference."""
    catalog = DataCatalog(primary_key=["station_id", "subloc", "param"])
    for ref in refs:
        catalog.add(ref)
    return SimpleNamespace(_catalog=catalog, unit_conversion=unit_conversion)


class TestGetDataReference:
    """Regression tests for DatastoreUIMgr.get_data_reference.

    Uses SimpleNamespace to supply the two attributes the method needs
    (_catalog, unit_conversion) without constructing a full DatastoreUIMgr
    (which requires a real StationDatastore/filesystem).
    """

    def test_lookup_by_name_column(self):
        """Primary path: row contains a 'name' column."""
        ref = _make_ds_ref("anh_flow", "anh", "", "flow")
        mgr = _make_mock_mgr([ref])

        row = pd.Series({"name": "anh_flow", "station_id": "anh", "subloc": "", "param": "flow", "unit": "cfs"})
        result = DatastoreUIMgr.get_data_reference(mgr, row)

        assert result is ref

    def test_lookup_without_name_column_uses_pk_fallback(self):
        """Fallback path: row lacks 'name' — should resolve via pk lookup, not _ref_name."""
        ref = _make_ds_ref("anh_flow", "anh", "", "flow")
        mgr = _make_mock_mgr([ref])

        # Simulate selected_dataframe from the download action when 'name' was
        # not in the display table columns.
        row = pd.Series({"station_id": "anh", "subloc": "", "param": "flow", "unit": "cfs"})
        result = DatastoreUIMgr.get_data_reference(mgr, row)

        assert result is ref

    def test_missing_name_does_not_raise_attribute_error(self):
        """Regression: 'DatastoreUIMgr' object has no attribute '_ref_name' must not occur."""
        ref = _make_ds_ref("anh_flow", "anh", "", "flow")
        mgr = _make_mock_mgr([ref])

        row = pd.Series({"station_id": "anh", "subloc": "", "param": "flow", "unit": "cfs"})

        # AttributeError was raised before the fix; KeyError/other exceptions are bugs too.
        result = DatastoreUIMgr.get_data_reference(mgr, row)
        assert result is not None

    def test_get_table_schema_includes_name_as_hidden_column(self, monkeypatch):
        """'name' must be in required_columns so selected_dataframe always carries it."""
        # We only need the method return value; bypass get_data_catalog() entirely.
        mgr = SimpleNamespace()
        schema = DatastoreUIMgr.get_table_schema(mgr, df=pd.DataFrame())

        assert "name" in schema["required_columns"], (
            "'name' must be a required column so selected_dataframe has the catalog key"
        )
        assert "name" in schema["hidden_by_default"], (
            "'name' must be hidden by default to keep the table uncluttered"
        )


class TestGetDataColumnNaming:
    """Downloaded series should have descriptive column names, not the generic 'value'."""

    def test_series_label_no_subloc(self, timeseries_df):
        """station_id/param (unit) when there is no subloc."""
        row = pd.Series({"station_id": "anh", "subloc": "", "param": "flow", "unit": "cfs"})
        label = DatastoreUIMgr._series_label(row, timeseries_df)
        assert label == "anh/flow (cfs)"

    def test_series_label_with_subloc(self, timeseries_df):
        """station_id@subloc/param (unit) when subloc is set."""
        row = pd.Series({"station_id": "msd", "subloc": "bottom", "param": "ec", "unit": "uS/cm"})
        label = DatastoreUIMgr._series_label(row, timeseries_df)
        assert label == "msd@bottom/ec (uS/cm)"

    def test_series_label_uses_converted_unit_from_attrs(self, timeseries_df):
        """When unit_conversion is active, the converted unit in data.attrs is used."""
        row = pd.Series({"station_id": "anh", "subloc": "", "param": "flow", "unit": "cfs"})
        data_with_attr = timeseries_df.copy()
        data_with_attr.attrs["unit"] = "cms"  # converted unit
        label = DatastoreUIMgr._series_label(row, data_with_attr)
        assert label == "anh/flow (cms)"

    def test_value_column_is_renamed_in_get_data(self, monkeypatch, timeseries_df):
        """get_data renames the 'value' column using _series_label."""
        ref = _make_ds_ref("anh_flow", "anh", "", "flow")
        catalog = DataCatalog(primary_key=["station_id", "subloc", "param"])
        catalog.add(ref)
        monkeypatch.setattr(ref, "getData", lambda time_range=None: timeseries_df.copy())

        # SimpleNamespace with get_data_reference callable — SimpleNamespace does not
        # bind functions as methods so the function receives only the explicit arg.
        mgr = SimpleNamespace(
            _catalog=catalog,
            unit_conversion=False,
            time_range=None,
            data_catalog=catalog,
            _dataui=None,
            get_data_reference=lambda row: ref,
        )
        selected_df = pd.DataFrame([{
            "name": "anh_flow", "station_id": "anh",
            "subloc": "", "param": "flow", "unit": "cfs",
        }])

        results = list(DatastoreUIMgr.get_data(mgr, selected_df))

        assert len(results) == 1
        col = results[0].columns[0]
        assert col == "anh/flow (cfs)", f"Expected 'anh/flow (cfs)', got {col!r}"
        assert "value" not in results[0].columns
