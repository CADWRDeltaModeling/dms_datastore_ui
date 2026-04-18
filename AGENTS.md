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

## Primary Files To Read First

- `dms_datastore_ui/cli.py` (Click command surface)
- `dms_datastore_ui/datastore_uimgr.py` (`DatastoreUIMgr` extends `dvue.tsdataui.TimeSeriesDataUIManager`)
- `dms_datastore_ui/datastore_actions.py` (custom action callbacks)
- `dms_datastore_ui/map_inventory_explorer.py` (`StationDatastore`, catalog/inventory glue)
- `README.md` (supported commands and demos)

## Conventions And Pitfalls

- `repo_level` means datastore subdirectory level (for example `screened`, `formatted`).
- Avoid duplicate caching. `DatastoreCatalogBuilder` creates refs with `cache=False` because datastore already caches on disk.
- Keep extension initialization timing safe for Panel sessions. Follow module-level import pattern in `datastore_actions.py`.
- Keep catalog metadata columns stable (`station_id`, `subloc`, `param`, `unit`, `min_year`, `max_year`, `geometry`, `repo_level`, `filename`) because filters/actions depend on them.
- Preserve CRS assumptions unless intentionally changed and tested (`EPSG:26910` currently used in catalog creation).

## External Docs (Link, Do Not Duplicate)

- `../dms_datastore/.github/copilot-instructions.md` (datastore conventions and test patterns)
- `../dvue/AGENTS.md` (dvue manager/catalog/action conventions)
- `README.md` (UI command usage)
