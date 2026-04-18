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
        )

        refs = builder.build(source)
        readers = {id(ref._reader) for ref in refs}

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

        catalog = DataCatalog().add(ds_ref).add(mem_ref)

        assert set(catalog.list_names()) == {"ds_flow", "mem_flow"}
        assert len(catalog.search(param="flow")) == 2
        catalog_df = catalog.to_dataframe()
        assert "param" in catalog_df.columns

        sum_ref = ds_ref + mem_ref
        result = sum_ref.getData()
        expected = pd.DataFrame({"result": (timeseries_df + other_df)["value"]})
        pd.testing.assert_frame_equal(result, expected)
