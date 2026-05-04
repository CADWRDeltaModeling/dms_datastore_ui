# dms_datastore_ui — Agent Instructions

## Scope

Use this file when working in the `dms_datastore_ui/` workspace root.

## Purpose

`dms_datastore_ui` is a Panel-based dashboard/CLI layer over `dms_datastore`, built with reusable `dvue` managers and actions.

## Terminal And Environment (Required)

- Always run commands from a Command Prompt shell (`cmd`), not PowerShell.
- If currently in PowerShell, enter `cmd` first.
- Activate the project environment before running any install/test/run command:
  - `conda activate dms_datastore_ui`
- Verify the active environment when needed:
  - `conda info --envs` and confirm `dms_datastore_ui` is marked with `*`.
- Prefer `python -m pytest ...` over `pytest ...` to avoid PATH-related issues in this environment.

## Fast Start For Agents

1. Install editable package:
   - `cmd`
   - `conda activate dms_datastore_ui`
   - `pip install --no-deps -e .`
2. Run tests:
   - `python -m pytest tests/`
3. Run CLI help:
   - `dms_datastore_ui -h`
4. Run dashboard entrypoint:
   - `dms_datastore_ui show-repo <repo_dir>`

## Key Boundaries

- Domain and repository semantics belong in `dms_datastore_ui`.
- Reusable UI framework behavior belongs in `dvue`.
- Low-level data ingestion/formatting belongs in `dms_datastore`.
- Do not re-implement `dvue` base capabilities in UI modules.

## Architecture

| Class | Module | Role |
|---|---|---|
| `DatastoreUIMgr` | `datastore_uimgr.py` | Main dashboard manager; extends `dvue.TimeSeriesDataUIManager` |
| `DatastoreDataReference` | `datastore_uimgr.py` | dvue `DataReference` with datastore metadata (filepath, station_id, subloc, param, unit, geometry) |
| `DatastoreFilepathReader` | `datastore_uimgr.py` | dvue reader; flyweight shared across all references; calls `read_ts(filepath)` |
| `DatastoreCatalogBuilder` | `datastore_uimgr.py` | dvue builder; matches `StationDatastore` sources; builds reference list from inventory |
| `DatastorePlotAction` | `datastore_uimgr.py` | Customizes curve labels (`station_id@subloc`) and titles |
| `StationDatastore` | `map_inventory_explorer.py` | Runtime state: inventory, diskcache, unit conversion, filtering |
| `DataScreenerAction` | `datastore_actions.py` | Opens screener tabs for selected rows |
| `FlagEditorAction` | `datastore_actions.py` | Opens flag editor for selected rows |
| `GapVisualizerAction` | `datastore_actions.py` | Visualizes data gaps for selected rows |

## CLI Commands

| Command | Signature | Notes |
|---|---|---|
| `version` | `dms_datastore_ui version` | Prints package version |
| `show-repo` | `dms_datastore_ui show-repo <repo_dir>` | Launches full dashboard; opens browser |
| `data-screener` | `dms_datastore_ui data-screener <filepath>` | Opens screener for a single CSV file |
| `flag-editor` | `dms_datastore_ui flag-editor <filepath>` | Opens flag editor for a single CSV file |

## Primary Files To Read First

- [dms_datastore_ui/cli.py](dms_datastore_ui/cli.py) (Click command surface)
- [dms_datastore_ui/datastore_uimgr.py](dms_datastore_ui/datastore_uimgr.py) (`DatastoreUIMgr`, reference/reader/builder classes)
- [dms_datastore_ui/datastore_actions.py](dms_datastore_ui/datastore_actions.py) (action callbacks; module-level pn.extension() pattern)
- [dms_datastore_ui/map_inventory_explorer.py](dms_datastore_ui/map_inventory_explorer.py) (`StationDatastore`, caching, unit conversion)
- [dms_datastore_ui/datastore_catalog_datareference_design.md](dms_datastore_ui/datastore_catalog_datareference_design.md) (design rationale)
- [README.md](README.md) (supported commands and demos)

## Conventions And Pitfalls

### Catalog Initialization Order (Critical)
Build `DataCatalog` **before** calling `super().__init__()` in `DatastoreUIMgr`. The parent constructor calls `get_data_catalog()` during init.

### Panel Extension Imports (Critical)
In `datastore_actions.py`, `data_screener`, `flag_editor`, and `gap_visualizer` are imported **eagerly at module level** (not inside callbacks). Each calls `pn.extension()` at import time. Importing inside a live Bokeh session callback would reset `pn.state.curdoc` and orphan Bokeh models.

### Mixed Catalog — NaN Guards
`get_data_reference()` must check the `name` column **first** (math refs have `name` but not `filename`):
```python
def get_data_reference(self, row):
    if "name" in row.index:
        return self._dvue_catalog.get(row["name"])
    return self._dvue_catalog.get(self._ref_name(row))
```
Also filter NaN before calling `get_unique_short_names()` — see `../dvue/AGENTS.md` for full details.

### Other Conventions
- `repo_level` = datastore subdirectory level (e.g. `screened`, `formatted`).
- `DatastoreCatalogBuilder` creates refs with `cache=False`; `StationDatastore` handles disk caching via `diskcache`.
- Subloc normalization: `nan` → empty string (never `None`). Applied in `DatastoreDataReference.from_inventory_row()`.
- Caching hierarchy: `diskcache.Cache.memoize()` on `read_ts` → `DatastoreFilepathReader` → `DatastoreDataReference`. Do not add caching at multiple layers.
- `_sync_repo_level()` watcher (depends on `repo_level` param) keeps `StationDatastore` in sync when the UI param changes.
- Unit conversion is applied in `get_data_for_time_range()` via `to_uniform_units()`. Never apply it again downstream.
- Keep catalog metadata columns stable: `station_id`, `subloc`, `param`, `unit`, `min_year`, `max_year`, `geometry`, `repo_level`, `filename`. Filters and actions depend on them.
- Preserve CRS: `EPSG:26910` (UTM Zone 10N) used for map catalog creation. Change only if intentionally re-projecting.
- Inventory file glob: `inventory_datasets_{repo_level}*.csv`. Multiple versions → `find_lastest_fname()` picks newest.

## Testing

| File | What it tests | Requires `--repo`? |
|---|---|---|
| `tests/test_cli.py` | CLI parsing smoke test | No |
| `tests/test_catalog_datastore.py` | Reader / Reference / Builder / mixed-catalog interop | No |
| `tests/test_station_datastore_caching.py` | `StationDatastore` caching unit tests with fake repo | No |
| `tests/test_caching.py` | End-to-end caching performance (10x+ speedup assertion) | Yes (`--repo=<path>`) |

Key fixtures: `fake_repo(tmp_path)`, `datastore(fake_repo, monkeypatch)`. Integration tests self-skip: `pytest.skip("Pass --repo=<path> to run")`.

## External Docs (Link, Do Not Duplicate)

- [../dms_datastore/.github/copilot-instructions.md](../dms_datastore/.github/copilot-instructions.md) (datastore file conventions, download modules, test patterns)
- [../dvue/AGENTS.md](../dvue/AGENTS.md) (dvue manager/catalog/action conventions, mixed-catalog NaN handling, TransformToCatalogAction naming)
- [README.md](README.md) (UI command usage and demos)
