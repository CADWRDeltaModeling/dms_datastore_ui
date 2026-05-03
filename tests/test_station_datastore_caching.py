"""Unit tests for StationDatastore caching.

Creates a minimal in-memory fake repo (two stations, one file each) and verifies
that:
  1. Cold reads populate the cache.
  2. Warm reads return the same data from the cache without re-reading the file.
  3. cache_repo_level() pre-populates the cache for all inventory entries.

No real repository is required; runs with plain `pytest`.
"""
import os
import textwrap

import pandas as pd
import pytest

from dms_datastore_ui.map_inventory_explorer import StationDatastore

# ---------------------------------------------------------------------------
# Minimal DWR-DMS 1.0 CSV content for two fake stations
# ---------------------------------------------------------------------------

def _make_dms_csv(agency, agency_id, lat, lon, values):
    """Return a DWR-DMS 1.0 CSV string with hourly rows starting 2024-01-01."""
    header = textwrap.dedent(f"""\
        # format: dwr-dms-1.0
        # agency: {agency}
        # agency_id: {agency_id}
        # latitude: {lat}
        # longitude: {lon}
        # date_formatted: 2024-01-01 00:00:00
        """)
    rows = ["datetime,value,user_flag"]
    for i, v in enumerate(values):
        ts = f"2024-01-01T{i:02d}:00:00"
        rows.append(f"{ts},{v},")
    return header + "\n".join(rows) + "\n"


_STATION_A_VALUES = list(range(1, 25))   # 24 hourly points, values 1–24
_STATION_B_VALUES = list(range(10, 250, 10))  # 24 hourly points, values 10–240

_STATION_A_CSV = _make_dms_csv("usgs", "11303500", 38.0, -121.5, _STATION_A_VALUES)
_STATION_B_CSV = _make_dms_csv("cdec", "SAC", 38.5, -121.8, _STATION_B_VALUES)


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_repo(tmp_path):
    """Build a minimal repository directory tree accepted by StationDatastore."""
    screened = tmp_path / "screened"
    screened.mkdir()

    # Write two station CSV files
    file_a = "usgs_anh_11303500_flow_2024.csv"
    file_b = "cdec_sac_11447650_ec_2024.csv"
    (screened / file_a).write_text(_STATION_A_CSV)
    (screened / file_b).write_text(_STATION_B_CSV)

    # Write a minimal inventory file (name matches the glob used in __init__)
    inventory = pd.DataFrame(
        [
            {
                "filename": file_a,
                "station_id": "anh",
                "subloc": "",
                "name": "Andrus Island",
                "param": "flow",
                "unit": "cfs",
                "min_year": 2024,
                "max_year": 2024,
                "agency": "usgs",
                "agency_id_dbase": "11303500",
                "x": 625000.0,
                "y": 4200000.0,
            },
            {
                "filename": file_b,
                "station_id": "sac",
                "subloc": "",
                "name": "Sacramento River",
                "param": "ec",
                "unit": "microS/cm",
                "min_year": 2024,
                "max_year": 2024,
                "agency": "cdec",
                "agency_id_dbase": "11447650",
                "x": 620000.0,
                "y": 4210000.0,
            },
        ]
    )
    inventory.to_csv(tmp_path / "inventory_datasets_screened.csv", index=False)

    return tmp_path


@pytest.fixture()
def datastore(fake_repo, monkeypatch):
    """StationDatastore pointing at fake_repo with cache inside tmp dir."""
    # Keep the diskcache inside tmp_path so it is isolated per test and
    # is cleaned up automatically.
    monkeypatch.chdir(fake_repo)
    ds = StationDatastore(str(fake_repo))
    yield ds
    ds.cache.clear()
    ds.cache.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _filepath(ds, filename):
    return ds.get_data_filepath(ds.repo_level[0], filename)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStationDatastoreCaching:
    def test_cold_read_returns_dataframe(self, datastore):
        """get_data() returns a non-empty DataFrame on a cold (uncached) read."""
        inv = datastore.df_dataset_inventory
        filename = inv.iloc[0]["filename"]

        df = datastore.get_data(datastore.repo_level[0], filename)

        assert isinstance(df, pd.DataFrame), "get_data() should return a DataFrame"
        assert len(df) > 0, "DataFrame should not be empty"

    def test_cold_read_populates_cache(self, datastore):
        """After a cold read the cache should contain at least one entry."""
        datastore.cache.clear()
        inv = datastore.df_dataset_inventory
        filename = inv.iloc[0]["filename"]
        filepath = _filepath(datastore, filename)

        assert len(datastore.cache) == 0, "Cache should be empty before first read"

        datastore.caching_read_ts(filepath)

        assert len(datastore.cache) >= 1, (
            "Cache should have at least one entry after reading a file; "
            f"cache has {len(datastore.cache)} entries"
        )

    def test_warm_read_returns_same_data(self, datastore):
        """A second read for the same file returns identical data (from cache)."""
        datastore.cache.clear()
        inv = datastore.df_dataset_inventory
        filename = inv.iloc[0]["filename"]
        filepath = _filepath(datastore, filename)

        df_cold = datastore.caching_read_ts(filepath)
        df_warm = datastore.caching_read_ts(filepath)

        pd.testing.assert_frame_equal(
            df_cold, df_warm,
            check_like=True,
            obj="warm-read DataFrame should equal cold-read DataFrame",
        )

    def test_two_stations_cached_independently(self, datastore):
        """Reading two distinct station files results in two cache entries."""
        datastore.cache.clear()
        inv = datastore.df_dataset_inventory

        fp_a = _filepath(datastore, inv.iloc[0]["filename"])
        fp_b = _filepath(datastore, inv.iloc[1]["filename"])

        datastore.caching_read_ts(fp_a)
        datastore.caching_read_ts(fp_b)

        assert len(datastore.cache) >= 2, (
            "Both station files should produce separate cache entries; "
            f"cache has {len(datastore.cache)} entries"
        )

    def test_cache_repo_level_pre_populates_all_entries(self, datastore):
        """cache_repo_level() should populate the cache for every inventory file."""
        datastore.cache.clear()
        assert len(datastore.cache) == 0

        datastore.cache_repo_level(datastore.repo_level[0])

        n_files = len(datastore.df_dataset_inventory["filename"].unique())
        assert len(datastore.cache) >= n_files, (
            f"Expected at least {n_files} cache entries after cache_repo_level(); "
            f"found {len(datastore.cache)}"
        )

    def test_get_data_returns_correct_values_for_each_station(self, datastore):
        """Each station's data has the expected row count and value range."""
        datastore.cache.clear()
        inv = datastore.df_dataset_inventory

        # Station A: flow, 24 hourly rows, values 1–24
        df_a = datastore.get_data(
            datastore.repo_level[0], inv.iloc[0]["filename"]
        )
        assert len(df_a) == 24
        assert list(df_a["value"]) == _STATION_A_VALUES

        # Station B: ec, 24 hourly rows, values 10,20,...,240
        df_b = datastore.get_data(
            datastore.repo_level[0], inv.iloc[1]["filename"]
        )
        assert len(df_b) == 24
        assert list(df_b["value"]) == _STATION_B_VALUES

    def test_cached_data_survives_reinit(self, fake_repo, monkeypatch):
        """A new StationDatastore instance with the same dir reads from the
        existing on-disk cache (i.e. the cache is persistent across instances)."""
        monkeypatch.chdir(fake_repo)

        ds1 = StationDatastore(str(fake_repo))
        ds1.cache.clear()
        inv = ds1.df_dataset_inventory
        filename = inv.iloc[0]["filename"]
        filepath = ds1.get_data_filepath(ds1.repo_level[0], filename)

        ds1.caching_read_ts(filepath)
        n_after_write = len(ds1.cache)
        ds1.cache.close()

        # A second instance pointing at the same dir should pick up the cached data
        ds2 = StationDatastore(str(fake_repo))
        n_on_open = len(ds2.cache)
        ds2.cache.close()

        assert n_after_write >= 1, "First instance should have written to cache"
        assert n_on_open >= n_after_write, (
            "Second instance should see at least as many cache entries as first instance wrote"
        )
