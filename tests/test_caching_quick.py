"""Quick cache diagnostic - uses a small file to finish fast.
Run: conda activate dms_datastore_ui && python tests/test_caching_quick.py continuous
"""
import sys, time, os

repo_dir = sys.argv[1] if len(sys.argv) > 1 else "continuous"

from dms_datastore_ui.map_inventory_explorer import StationDatastore

ds = StationDatastore(repo_dir)
repo_level = ds.repo_level[0]

# Pick the smallest file by size
inv = ds.df_dataset_inventory
filepaths = [ds.get_data_filepath(repo_level, fn) for fn in inv["filename"]]
existing = [(fp, os.path.getsize(fp.replace("*", "").replace("_.csv", "_2020_9999.csv")) if os.path.exists(fp) else 999e9) for fp in filepaths[:50]]

# Just pick the first file
filepath = ds.get_data_filepath(repo_level, inv["filename"].iloc[0])
print(f"\nTest file: {filepath}")

# --- Direct diskcache inspection ---
print("\n=== Layer 1: diskcache memoize ===")
ds.cache.clear()
print(f"  Cache entries before call : {len(ds.cache)}")

t0 = time.perf_counter()
df = ds.caching_read_ts(filepath)
e1 = time.perf_counter() - t0
print(f"  Cold read : {e1:.3f}s  ({len(df)} rows)")
print(f"  Cache entries after call  : {len(ds.cache)}")
print(f"  Cache keys                : {list(ds.cache.iterkeys())}")

t0 = time.perf_counter()
df2 = ds.caching_read_ts(filepath)
e2 = time.perf_counter() - t0
print(f"  Warm read : {e2:.3f}s  (speedup: {e1/max(e2,0.001):.0f}x)")

if len(ds.cache) == 0:
    print("\n  *** DISKCACHE IS EMPTY AFTER WRITE — testing direct set() ***")
    # Try storing a simple value manually
    ds.cache.set("test_key", "test_value")
    print(f"  Manual set string: cache now has {len(ds.cache)} entries")
    # Try storing the actual DataFrame directly
    try:
        ds.cache.set("df_test_key", df)
        print(f"  Manual set DataFrame: cache now has {len(ds.cache)} entries")
    except Exception as exc:
        print(f"  Manual set DataFrame FAILED: {type(exc).__name__}: {exc}")
    # Check cache directory
    print(f"  Cache directory: {ds.cache.directory}")
    import diskcache
    print(f"  diskcache version: {diskcache.__version__}")
    print(f"  Cache size_limit: {ds.cache.size_limit}")
    print(f"  Cache volume(): {ds.cache.volume()}")
