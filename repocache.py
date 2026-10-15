"""Prepare the repository series cache for the current data.

Reuses the local cache when it matches the data, otherwise restores a published
snapshot, otherwise warms the cache and publishes a snapshot for later starts.
The cache is never cleared unless ``--rebuild`` is given.
"""
import argparse
import datetime as dt
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_dir", help="repository directory, e.g. continuous")
    parser.add_argument("--cache-root", help="local cache root (default: $DMS_DATASTORE_UI_CACHE_ROOT or cwd)")
    parser.add_argument("--snapshot-dir", help="durable snapshot directory (default: $DMS_DATASTORE_UI_CACHE_SNAPSHOT_DIR)")
    parser.add_argument("--restore-only", action="store_true", help="use or restore a cache but never warm")
    parser.add_argument("--rebuild", action="store_true", help="discard cached entries and warm from scratch")
    parser.add_argument("--no-publish", action="store_true", help="do not publish a snapshot after warming")
    args = parser.parse_args(argv)

    from dms_datastore_ui.cache_snapshot import prepare_cache

    print(f"Cache task started at: {dt.datetime.now()}", flush=True)
    status = prepare_cache(
        args.repo_dir,
        cache_root=args.cache_root,
        snapshot_dir=args.snapshot_dir,
        restore_only=args.restore_only,
        rebuild=args.rebuild,
        publish=not args.no_publish,
        log=lambda msg: print(msg, flush=True),
    )
    print(f"Cache status: {status} at {dt.datetime.now()}", flush=True)
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
