"""Versioned, restorable snapshots of the repository series cache.

The serving cache is a local ``diskcache`` directory (fast, per container).
Its location is derived from a *data version* fingerprint of the repository, so
a changed repository automatically selects a new, empty cache generation::

    <cache_root>/cache_<repo>/<data_version>/        local, used by the UI
    <snapshot_dir>/<data_version>/{cache.tar,manifest.json}   durable, shared

A container start therefore does one of three things, in order:

1. local generation already ready      -> use it
2. matching snapshot is published      -> restore it (no CSV parsing)
3. otherwise                           -> warm the cache, then publish a snapshot

Readiness is an explicit marker (``ready.json``) written only after a warm-up
finishes; a log file's existence says nothing about the cache's contents.
"""

import hashlib
import json
import os
import shutil
import socket
import sqlite3
import tarfile
import tempfile
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

# Bump when cached DataFrames change shape/meaning (e.g. reader or unit changes)
# so older caches and snapshots are never reused.
FORMAT_VERSION = 1

ENV_CACHE_ROOT = "DMS_DATASTORE_UI_CACHE_ROOT"
ENV_SNAPSHOT_DIR = "DMS_DATASTORE_UI_CACHE_SNAPSHOT_DIR"

READY_NAME = "ready.json"
MANIFEST_NAME = "manifest.json"
ARCHIVE_NAME = "cache.tar"
KEEP_SNAPSHOTS = 2
# A warm-up with more failed series than this is not trusted or published.
MAX_FAILED_FRACTION = 0.01

_VERSION_TTL_SECONDS = 300
_version_memo = {}


# ---------------------------------------------------------------------------
# Data version
# ---------------------------------------------------------------------------


def _inventory_files(repo_dir, repo_level):
    return sorted(Path(repo_dir).glob(f"inventory_datasets_{repo_level}*.csv"))


def discover_repo_levels(repo_dir):
    """Repository levels are sub-directories that have an inventory file."""
    levels = []
    for entry in sorted(os.scandir(repo_dir), key=lambda e: e.name):
        if entry.is_dir() and not entry.name.startswith(".") and _inventory_files(repo_dir, entry.name):
            levels.append(entry.name)
    return levels


def _scan_files(directory):
    """Yield ``(relative_path, size, mtime_ns)`` for every file below *directory*."""
    stack = [(directory, "")]
    while stack:
        path, prefix = stack.pop()
        with os.scandir(path) as it:
            for entry in it:
                rel = f"{prefix}{entry.name}"
                if entry.is_dir(follow_symlinks=False):
                    stack.append((entry.path, rel + "/"))
                else:
                    st = entry.stat()
                    yield rel, st.st_size, st.st_mtime_ns


def compute_data_version(repo_dir, use_memo=True):
    """Fingerprint the repository's data files.

    The fingerprint covers the name, size and modification time of every file
    in each repository level plus its inventory files, so added, removed and
    rewritten files all change it. A recent result is reused unless the
    inventory files changed, which keeps per-session construction cheap.
    """
    repo_dir = os.path.abspath(repo_dir)
    levels = discover_repo_levels(repo_dir)
    inventories = tuple(
        (f.name, f.stat().st_size, f.stat().st_mtime_ns)
        for level in levels
        for f in _inventory_files(repo_dir, level)
    )
    memo_key = (repo_dir, inventories)
    hit = _version_memo.get(memo_key)
    if use_memo and hit and time.monotonic() - hit[0] < _VERSION_TTL_SECONDS:
        return hit[1]

    digest = hashlib.sha256(f"format={FORMAT_VERSION}\n".encode())
    for item in inventories:
        digest.update(f"inventory|{item}\n".encode())
    for level in levels:
        for item in sorted(_scan_files(os.path.join(repo_dir, level))):
            digest.update(f"{level}|{item}\n".encode())
    version = digest.hexdigest()[:16]
    _version_memo[memo_key] = (time.monotonic(), version)
    return version


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def cache_family_dir(repo_dir, cache_root=None):
    root = os.path.abspath(cache_root or os.environ.get(ENV_CACHE_ROOT) or os.getcwd())
    return os.path.join(root, "cache_" + os.path.basename(os.path.normpath(repo_dir)))


def generation_dir(repo_dir, version, cache_root=None):
    return os.path.join(cache_family_dir(repo_dir, cache_root), version)


def snapshot_root(repo_dir, snapshot_dir=None):
    if snapshot_dir or os.environ.get(ENV_SNAPSHOT_DIR):
        return os.path.abspath(snapshot_dir or os.environ[ENV_SNAPSHOT_DIR])
    repo_dir = os.path.abspath(repo_dir)
    return os.path.join(os.path.dirname(repo_dir), "cache_snapshots", os.path.basename(repo_dir))


# ---------------------------------------------------------------------------
# Manifests and readiness
# ---------------------------------------------------------------------------


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _manifest_matches(manifest, version):
    return bool(manifest) and manifest.get("data_version") == version and manifest.get("format_version") == FORMAT_VERSION


def is_ready(gen_dir, version):
    return _manifest_matches(_read_json(os.path.join(gen_dir, READY_NAME)), version)


def write_ready(gen_dir, manifest):
    _write_json(os.path.join(gen_dir, READY_NAME), manifest)


def _count_entries(cache_dir):
    import diskcache

    with diskcache.Cache(cache_dir) as cache:
        return len(cache)


# ---------------------------------------------------------------------------
# Snapshot files
# ---------------------------------------------------------------------------

_LIVE_DB_FILES = {"cache.db", "cache.db-wal", "cache.db-shm"}


def copy_cache_consistent(src, dst):
    """Copy a diskcache directory that may be in use by other processes.

    Value files are copied first and the SQLite index last, through SQLite's
    online backup, so every index row in the copy refers to a file that was
    already copied (files are written before their row is committed).
    """
    os.makedirs(dst, exist_ok=True)
    for root, _, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel_root), exist_ok=True)
        for name in files:
            if rel_root == "." and (name in _LIVE_DB_FILES or name.startswith(READY_NAME)):
                continue
            try:
                shutil.copy2(os.path.join(root, name), os.path.join(dst, rel_root, name))
            except FileNotFoundError:
                continue
    src_db = os.path.join(src, "cache.db")
    if not os.path.isfile(src_db):
        raise FileNotFoundError(f"No cache.db in {src}")
    with closing(sqlite3.connect(src_db)) as source, closing(sqlite3.connect(os.path.join(dst, "cache.db"))) as target:
        source.backup(target)


def _make_archive(src_dir, archive_path):
    with tarfile.open(archive_path, "w") as tar:
        for name in sorted(os.listdir(src_dir)):
            tar.add(os.path.join(src_dir, name), arcname=name)


def _safe_extract(archive_path, dest):
    base = os.path.realpath(dest)
    with tarfile.open(archive_path) as tar:
        members = tar.getmembers()
        for member in members:
            target = os.path.realpath(os.path.join(base, member.name))
            if not (member.isfile() or member.isdir()) or os.path.commonpath([base, target]) != base:
                raise ValueError(f"Unsafe member in cache archive: {member.name!r}")
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(base, members=members, **kwargs)


def find_snapshot(repo_dir, version, snapshot_dir=None):
    """Return ``(archive_path, manifest)`` for a complete matching snapshot, else ``None``."""
    directory = os.path.join(snapshot_root(repo_dir, snapshot_dir), version)
    manifest = _read_json(os.path.join(directory, MANIFEST_NAME))
    archive = os.path.join(directory, ARCHIVE_NAME)
    if (
        _manifest_matches(manifest, version)
        and os.path.isfile(archive)
        and os.path.getsize(archive) == manifest.get("archive_size")
    ):
        return archive, manifest
    return None


def restore_snapshot(repo_dir, version, cache_root=None, snapshot_dir=None, log=print):
    """Install the published snapshot for *version* as the local generation.

    Must run before anything opens the local generation. Returns ``True`` when
    a snapshot was restored; any failure is logged and returns ``False`` so the
    caller falls back to warming.
    """
    found = find_snapshot(repo_dir, version, snapshot_dir)
    if found is None:
        return False
    archive, manifest = found
    family = cache_family_dir(repo_dir, cache_root)
    os.makedirs(family, exist_ok=True)
    stage = tempfile.mkdtemp(prefix=".restore-", dir=family)
    try:
        started = time.monotonic()
        _safe_extract(archive, stage)
        entries = _count_entries(stage)
        if entries < manifest.get("entries", 0):
            raise ValueError(f"snapshot has {entries} entries, expected {manifest['entries']}")
        write_ready(stage, manifest)
        target = generation_dir(repo_dir, version, cache_root)
        shutil.rmtree(target, ignore_errors=True)
        os.rename(stage, target)
        log(f"Restored cache snapshot {version} ({entries} entries) in {time.monotonic() - started:.1f}s")
        return True
    except Exception as exc:  # a bad snapshot must never block startup
        log(f"Could not restore cache snapshot {version}: {exc}")
        shutil.rmtree(stage, ignore_errors=True)
        return False


def publish_snapshot(repo_dir, version, manifest, cache_root=None, snapshot_dir=None, log=print):
    """Publish the local generation as a durable snapshot, atomically.

    The archive and manifest are written to a temporary directory that is
    renamed into place, so readers only ever see a complete snapshot.
    """
    if find_snapshot(repo_dir, version, snapshot_dir):
        return True
    root = snapshot_root(repo_dir, snapshot_dir)
    os.makedirs(root, exist_ok=True)
    family = cache_family_dir(repo_dir, cache_root)
    stage = tempfile.mkdtemp(prefix=".snapshot-", dir=family)
    tmp = tempfile.mkdtemp(prefix=".tmp-", dir=root)
    try:
        staged_cache = os.path.join(stage, "cache")
        copy_cache_consistent(generation_dir(repo_dir, version, cache_root), staged_cache)
        entries = _count_entries(staged_cache)
        archive = os.path.join(tmp, ARCHIVE_NAME)
        _make_archive(staged_cache, archive)
        published = {
            **manifest,
            "format_version": FORMAT_VERSION,
            "data_version": version,
            "entries": entries,
            "archive_size": os.path.getsize(archive),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
        }
        _write_json(os.path.join(tmp, MANIFEST_NAME), published)
        final = os.path.join(root, version)
        shutil.rmtree(final, ignore_errors=True)
        try:
            os.rename(tmp, final)
        except OSError:
            if not find_snapshot(repo_dir, version, snapshot_dir):  # lost a race to another publisher
                raise
        log(f"Published cache snapshot {version} ({entries} entries, {published['archive_size']} bytes)")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
    prune_snapshots(root, keep_version=version)
    return True


def prune_snapshots(root, keep_version, keep=KEEP_SNAPSHOTS):
    """Keep the newest *keep* published snapshots (always including *keep_version*)."""
    published = []
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            if time.time() - entry.stat().st_mtime > 86400:
                shutil.rmtree(entry.path, ignore_errors=True)
            continue
        manifest = _read_json(os.path.join(entry.path, MANIFEST_NAME)) or {}
        published.append((manifest.get("created_at", ""), entry.name, entry.path))
    for _, name, path in sorted(published, reverse=True)[keep:]:
        if name != keep_version:
            shutil.rmtree(path, ignore_errors=True)


def prune_local(repo_dir, keep_version, cache_root=None):
    """Delete local cache generations (and leftovers) other than *keep_version*."""
    family = cache_family_dir(repo_dir, cache_root)
    if not os.path.isdir(family):
        return
    for entry in os.scandir(family):
        if entry.name == keep_version:
            continue
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path, ignore_errors=True)
        else:
            try:
                os.unlink(entry.path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def warm_cache(repo_dir, cache_root=None, snapshot_dir=None, rebuild=False, publish=True, log=print):
    """Warm the local generation for the current data, then mark and publish it."""
    import diskcache
    from dms_datastore_ui.map_inventory_explorer import StationDatastore

    datastore = StationDatastore(repo_dir, cache_root=cache_root)
    try:
        if not isinstance(datastore.cache, diskcache.Cache):
            raise RuntimeError("A persistent cache directory is unavailable; refusing to warm a non-persistent cache")
        version, generation = datastore.data_version, datastore.cache_dir
        if rebuild:
            datastore.clear_cache()
            try:
                os.unlink(os.path.join(generation, READY_NAME))
            except OSError:
                pass
        expected, failed = 0, []
        for repo_level in datastore.param.repo_level.objects:
            log(f"Cache level: {repo_level}")
            level_total, level_failed = datastore.cache_repo_level(repo_level)
            expected += level_total
            failed += level_failed
        if expected and len(failed) / expected > MAX_FAILED_FRACTION:
            log(f"Warm-up failed: {len(failed)} of {expected} series could not be cached")
            return "failed"
        manifest = {
            "format_version": FORMAT_VERSION,
            "data_version": version,
            "expected": expected,
            "failed": len(failed),
            "failed_series": failed[:50],
            "entries": len(datastore.cache),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_ready(generation, manifest)
        if publish:
            publish_snapshot(repo_dir, version, manifest, cache_root, snapshot_dir, log)
        return "built"
    finally:
        datastore.cache.close()


def prepare_cache(
    repo_dir,
    cache_root=None,
    snapshot_dir=None,
    restore_only=False,
    rebuild=False,
    publish=True,
    log=print,
):
    """Make the local cache ready for the repository's current data.

    Returns ``"ready"``, ``"restored"``, ``"built"``, ``"missing"`` (restore-only
    found nothing to use) or ``"failed"``.
    """
    repo_dir = os.path.abspath(repo_dir)
    version = compute_data_version(repo_dir, use_memo=False)
    generation = generation_dir(repo_dir, version, cache_root)
    log(f"Repository data version: {version}")
    if not rebuild:
        if is_ready(generation, version):
            prune_local(repo_dir, version, cache_root)
            log("Local cache is ready")
            return "ready"
        if restore_snapshot(repo_dir, version, cache_root, snapshot_dir, log):
            prune_local(repo_dir, version, cache_root)
            return "restored"
    if restore_only:
        log("No matching cache snapshot published")
        return "missing"
    prune_local(repo_dir, version, cache_root)
    return warm_cache(repo_dir, cache_root, snapshot_dir, rebuild, publish, log)
