"""Test that the caching chain works correctly end-to-end.

Run from the dms_datastore_ui root directory:
    conda activate dms_datastore_ui
    python -m pytest tests/test_caching.py -v -s --repo=continuous

Or run directly:
    conda activate dms_datastore_ui
    python tests/test_caching.py continuous
"""
import os
import sys
import time
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# pytest fixture / conftest hook to accept --repo on the command line
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    try:
        parser.addoption("--repo", action="store", default=None, help="Path to repo dir")
    except ValueError:
        pass  # already registered by conftest.py


@pytest.fixture(scope="module")
def repo_dir(request):
    d = request.config.getoption("--repo", default=None)
    if d is None:
        pytest.skip("Pass --repo=<path> to run caching tests")
    return d


@pytest.fixture(scope="module")
def datastore(repo_dir):
    from dms_datastore_ui.map_inventory_explorer import StationDatastore
    ds = StationDatastore(repo_dir)
    return ds


@pytest.fixture(scope="module")
def sample_filepath(datastore):
    """Pick the first file from the inventory."""
    row = datastore.df_dataset_inventory.iloc[0]
    repo_level = datastore.repo_level[0]
    return datastore.get_data_filepath(repo_level, row["filename"])


# ---------------------------------------------------------------------------
# Layer 1: caching_read_ts (diskcache memoize) called directly
# ---------------------------------------------------------------------------

def test_caching_read_ts_second_call_is_faster(datastore, sample_filepath):
    """diskcache memoize should serve the second call from cache (much faster)."""
    print(f"\nFile: {sample_filepath}")

    t0 = time.perf_counter()
    df1 = datastore.caching_read_ts(sample_filepath)
    elapsed1 = time.perf_counter() - t0
    print(f"  1st call: {elapsed1:.3f}s  ({len(df1)} rows)")

    t0 = time.perf_counter()
    df2 = datastore.caching_read_ts(sample_filepath)
    elapsed2 = time.perf_counter() - t0
    print(f"  2nd call: {elapsed2:.3f}s  ({len(df2)} rows)")

    assert len(df1) == len(df2), "Row counts differ between calls"
    # Cache hit should be at least 10x faster
    assert elapsed2 < elapsed1 / 10, (
        f"2nd call ({elapsed2:.3f}s) not significantly faster than 1st ({elapsed1:.3f}s) — "
        "diskcache memoize is not caching"
    )


# ---------------------------------------------------------------------------
# Layer 2: DatastoreFilepathReader.load() with caching_read_ts
# ---------------------------------------------------------------------------

def test_filepath_reader_second_call_is_faster(datastore, sample_filepath):
    """DatastoreFilepathReader should pass the filepath through to caching_read_ts."""
    from dms_datastore_ui.datastore_uimgr import DatastoreFilepathReader

    reader = DatastoreFilepathReader(read_fn=datastore.caching_read_ts)

    # Clear the diskcache so we get a true cold read on the first call
    datastore.cache.clear()

    t0 = time.perf_counter()
    df1 = reader.load(filepath=sample_filepath)
    elapsed1 = time.perf_counter() - t0
    print(f"\n  Reader 1st call: {elapsed1:.3f}s  ({len(df1)} rows)")

    t0 = time.perf_counter()
    df2 = reader.load(filepath=sample_filepath)
    elapsed2 = time.perf_counter() - t0
    print(f"  Reader 2nd call: {elapsed2:.3f}s  ({len(df2)} rows)")

    assert elapsed2 < elapsed1 / 10, (
        f"Reader 2nd call ({elapsed2:.3f}s) not significantly faster — caching broken at reader layer"
    )


# ---------------------------------------------------------------------------
# Layer 3: DatastoreDataReference.getData() end-to-end
# ---------------------------------------------------------------------------

def test_data_reference_second_call_is_faster(datastore, sample_filepath):
    """getData() on a DatastoreDataReference should hit the cache on the 2nd call."""
    from dms_datastore_ui.datastore_uimgr import DatastoreFilepathReader, DatastoreDataReference

    reader = DatastoreFilepathReader(read_fn=datastore.caching_read_ts)
    datastore.cache.clear()

    row = datastore.df_dataset_inventory.iloc[0]
    repo_level = datastore.repo_level[0]
    ref = DatastoreDataReference.from_inventory_row(
        row=row,
        repo_dir=datastore.dir,
        repo_level=repo_level,
        reader=reader,
    )

    t0 = time.perf_counter()
    df1 = ref.getData()
    elapsed1 = time.perf_counter() - t0
    print(f"\n  DataReference 1st getData: {elapsed1:.3f}s  ({len(df1)} rows)")

    t0 = time.perf_counter()
    df2 = ref.getData()
    elapsed2 = time.perf_counter() - t0
    print(f"  DataReference 2nd getData: {elapsed2:.3f}s  ({len(df2)} rows)")

    assert elapsed2 < elapsed1 / 10, (
        f"DataReference 2nd getData ({elapsed2:.3f}s) not faster — caching broken at DataReference layer"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/test_caching.py <repo_dir>")
        sys.exit(1)

    repo_dir = sys.argv[1]

    from dms_datastore_ui.map_inventory_explorer import StationDatastore
    from dms_datastore_ui.datastore_uimgr import DatastoreFilepathReader, DatastoreDataReference

    print(f"=== Testing caching chain for repo: {repo_dir} ===\n")
    ds = StationDatastore(repo_dir)
    row = ds.df_dataset_inventory.iloc[0]
    filepath = ds.get_data_filepath(ds.repo_level[0], row["filename"])
    print(f"Sample file: {filepath}\n")

    # Layer 1 — diskcache memoize directly
    print("--- Layer 1: caching_read_ts directly ---")
    ds.cache.clear()
    print(f"  Cache size before read : {len(ds.cache)} entries")
    t0 = time.perf_counter(); df = ds.caching_read_ts(filepath); e1 = time.perf_counter() - t0
    print(f"  Cold read : {e1:.3f}s  ({len(df)} rows)")
    print(f"  Cache size after read  : {len(ds.cache)} entries")
    print(f"  Cache keys             : {list(ds.cache.iterkeys())}")
    t0 = time.perf_counter(); df = ds.caching_read_ts(filepath); e2 = time.perf_counter() - t0
    print(f"  Warm read : {e2:.3f}s  (speedup: {e1/max(e2,0.001):.0f}x)")

    # Layer 2 — through DatastoreFilepathReader
    print("\n--- Layer 2: DatastoreFilepathReader.load() ---")
    reader = DatastoreFilepathReader(read_fn=ds.caching_read_ts)
    ds.cache.clear()
    t0 = time.perf_counter(); df = reader.load(filepath=filepath); e1 = time.perf_counter() - t0
    print(f"  Cold read : {e1:.3f}s  ({len(df)} rows)")
    t0 = time.perf_counter(); df = reader.load(filepath=filepath); e2 = time.perf_counter() - t0
    print(f"  Warm read : {e2:.3f}s  (speedup: {e1/max(e2,0.001):.0f}x)")

    # Layer 3 — through DatastoreDataReference.getData()
    print("\n--- Layer 3: DatastoreDataReference.getData() ---")
    ref = DatastoreDataReference.from_inventory_row(
        row=row, repo_dir=ds.dir, repo_level=ds.repo_level[0], reader=reader
    )
    ds.cache.clear()
    t0 = time.perf_counter(); df = ref.getData(); e1 = time.perf_counter() - t0
    print(f"  Cold read : {e1:.3f}s  ({len(df)} rows)")
    t0 = time.perf_counter(); df = ref.getData(); e2 = time.perf_counter() - t0
    print(f"  Warm read : {e2:.3f}s  (speedup: {e1/max(e2,0.001):.0f}x)")
