"""Tests for versioned cache generations and restorable snapshots."""
import os
import time

import pytest

import dms_datastore
from dms_datastore_ui import cache_snapshot as cs
from dms_datastore_ui.map_inventory_explorer import StationDatastore

from tests.test_station_datastore_caching import fake_repo  # noqa: F401


@pytest.fixture()
def dirs(tmp_path):
    """A 'container-local' cache root and a durable snapshot directory."""
    return {"cache_root": str(tmp_path / "local_a"), "snapshot_dir": str(tmp_path / "snapshots")}


@pytest.fixture()
def read_calls(monkeypatch):
    calls = []
    real = dms_datastore.read_ts_repo

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(dms_datastore, "read_ts_repo", counting)
    return calls


def _prepare(repo, dirs, **kwargs):
    return cs.prepare_cache(repo, log=lambda msg: None, **dirs, **kwargs)


def _read_anh(repo, cache_root):
    ds = StationDatastore(str(repo), cache_root=cache_root)
    try:
        return ds.caching_read_ts_repo(
            "anh", "flow", repo="screened", data_path=os.path.join(ds.dir, "screened")
        )
    finally:
        ds.cache.close()


class TestDataVersion:
    def test_stable_when_nothing_changes(self, fake_repo):
        assert cs.compute_data_version(fake_repo, use_memo=False) == cs.compute_data_version(
            fake_repo, use_memo=False
        )

    def test_changes_when_a_data_file_is_rewritten(self, fake_repo):
        before = cs.compute_data_version(fake_repo, use_memo=False)
        data_file = fake_repo / "screened" / "usgs_anh_11303500_flow_2024.csv"
        data_file.write_text(data_file.read_text() + "2024-01-02T00:00:00,99,\n")
        assert cs.compute_data_version(fake_repo, use_memo=False) != before

    def test_changes_when_a_data_file_is_added_or_removed(self, fake_repo):
        before = cs.compute_data_version(fake_repo, use_memo=False)
        extra = fake_repo / "screened" / "usgs_new_1_flow_2024.csv"
        extra.write_text("x")
        added = cs.compute_data_version(fake_repo, use_memo=False)
        extra.unlink()
        assert added != before
        assert cs.compute_data_version(fake_repo, use_memo=False) == before

    def test_ignores_unrelated_directories(self, fake_repo):
        before = cs.compute_data_version(fake_repo, use_memo=False)
        (fake_repo / "cache_elsewhere").mkdir()
        (fake_repo / "cache_elsewhere" / "junk.bin").write_bytes(b"junk")
        assert cs.compute_data_version(fake_repo, use_memo=False) == before


class TestSnapshotLifecycle:
    def test_first_start_warms_marks_ready_and_publishes(self, fake_repo, dirs):
        assert _prepare(fake_repo, dirs) == "built"

        version = cs.compute_data_version(fake_repo, use_memo=False)
        generation = cs.generation_dir(str(fake_repo), version, dirs["cache_root"])
        assert cs.is_ready(generation, version)
        assert cs.find_snapshot(str(fake_repo), version, dirs["snapshot_dir"]) is not None

    def test_second_start_in_same_container_is_a_noop(self, fake_repo, dirs, read_calls):
        _prepare(fake_repo, dirs)
        n_reads = len(read_calls)

        assert _prepare(fake_repo, dirs) == "ready"
        assert len(read_calls) == n_reads

    def test_new_container_restores_instead_of_rebuilding(self, fake_repo, dirs, tmp_path, read_calls):
        _prepare(fake_repo, dirs)
        n_reads = len(read_calls)

        fresh = {**dirs, "cache_root": str(tmp_path / "local_b")}  # container replaced: no local cache
        assert _prepare(fake_repo, fresh, restore_only=True) == "restored"
        df = _read_anh(fake_repo, fresh["cache_root"])

        assert len(read_calls) == n_reads, "restored cache should serve reads without touching the repository"
        assert len(df) == 24

    def test_changed_data_does_not_reuse_old_snapshot(self, fake_repo, dirs, tmp_path):
        _prepare(fake_repo, dirs)
        data_file = fake_repo / "screened" / "usgs_anh_11303500_flow_2024.csv"
        data_file.write_text(data_file.read_text() + "2024-01-02T00:00:00,99,\n")

        fresh = {**dirs, "cache_root": str(tmp_path / "local_b")}
        assert _prepare(fake_repo, fresh, restore_only=True) == "missing"

    def test_truncated_snapshot_is_rejected(self, fake_repo, dirs, tmp_path):
        _prepare(fake_repo, dirs)
        version = cs.compute_data_version(fake_repo, use_memo=False)
        archive, _ = cs.find_snapshot(str(fake_repo), version, dirs["snapshot_dir"])
        with open(archive, "r+b") as f:
            f.truncate(os.path.getsize(archive) // 2)

        fresh = {**dirs, "cache_root": str(tmp_path / "local_b")}
        assert _prepare(fake_repo, fresh, restore_only=True) == "missing"

    def test_old_local_generations_are_pruned(self, fake_repo, dirs):
        _prepare(fake_repo, dirs)
        family = cs.cache_family_dir(str(fake_repo), dirs["cache_root"])
        stale = os.path.join(family, "0000stale0000")
        os.makedirs(stale)

        assert _prepare(fake_repo, dirs) == "ready"
        assert not os.path.exists(stale)

    def test_only_recent_snapshots_are_kept(self, fake_repo, dirs):
        for step in range(cs.KEEP_SNAPSHOTS + 1):
            data_file = fake_repo / "screened" / "usgs_anh_11303500_flow_2024.csv"
            data_file.write_text(data_file.read_text() + f"2024-02-0{step + 1}T00:00:00,1,\n")
            _prepare(fake_repo, dirs)
            time.sleep(0.01)

        published = [d for d in os.listdir(dirs["snapshot_dir"]) if not d.startswith(".")]
        assert len(published) == cs.KEEP_SNAPSHOTS

    def test_rebuild_discards_entries_and_rewarms(self, fake_repo, dirs, read_calls):
        _prepare(fake_repo, dirs)
        n_reads = len(read_calls)

        assert _prepare(fake_repo, dirs, rebuild=True) == "built"
        assert len(read_calls) > n_reads
