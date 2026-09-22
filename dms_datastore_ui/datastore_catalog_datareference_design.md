# Datastore Catalog and Data Reference Design

## Purpose

Provide a standalone catalog and reference design for dms_datastore_ui that remains compatible with the latest dvue catalog abstractions and supports mixed-reference catalogs.

## Goals

- Standalone references: each reference contains enough metadata to load data without depending on StationDatastore runtime state.
- Interoperability: datastore references can coexist with other DataReference objects in a single DataCatalog.
- Compatibility: conform to dvue DataReferenceReader, DataReference, CatalogBuilder, and DataCatalog contracts.
- Testability: behavior is validated with isolated tests that do not require a real repository.

## Baseline and Gaps

Previously:

- DatastoreReader loaded data by delegating to StationDatastore.get_data.
- DatastoreCatalogBuilder produced plain DataReference objects.
- No datastore-specific DataReference subclass existed.
- There were no standalone tests for datastore catalog/reference behavior.

Gap:

- References were not fully portable across contexts because loading behavior depended on StationDatastore.

## Design

### DatastoreFilepathReader

A dedicated DataReferenceReader implementation that loads logical repository series or standalone files.

Responsibilities:

- Implement load(**attributes).
- For repository references, require station, variable, repo, and repository-root attributes.
- Return `read_ts_repo(...)`, forwarding sublocation, modifier, and time range.
- For standalone CSV references, return `read_ts(filepath)`.

Notes:

- Uses the dvue reader contract directly.
- Supports flyweight usage, where one reader instance is shared across many references.
- Avoids StationDatastore coupling at load time.
- Preserves repository provider resolution, shard merging, flags, and regularity guarantees.

### DatastoreDataReference

A datastore-specific DataReference subclass with a focused convenience API.

Required metadata:

- repo_root
- repo_level
- series_id
- file_pattern
- station_id
- subloc
- modifier
- station_name
- param
- unit
- min_year
- max_year
- agency
- agency_id_dbase
- x
- y
- geometry

Convenience properties for now:

- filepath
- station_id
- subloc
- param
- unit
- geometry

Factory:

- from_inventory_row(row, repo_dir, repo_level, reader=None)
- Stores logical repository identity from the inventory row.
- Normalizes missing subloc values to empty string.
- Uses DatastoreFilepathReader by default when reader is not provided.

Caching:

- DataReference cache remains disabled for datastore-built references.
- Rationale: datastore already uses disk-backed caching.

### DatastoreCatalogBuilder

Builder behavior:

- can_handle(source) matches StationDatastore.
- build(source) merges inventory and constructs DatastoreDataReference objects.
- A shared DatastoreFilepathReader is used for all references created in one build call.
- DatastoreReader is removed entirely.

Result:

- Built references are portable and can load data independently through `read_ts_repo`.

## Mixed Catalog Interoperability

DatastoreDataReference must coexist with plain DataReference objects backed by any reader, including InMemoryDataReferenceReader and CallableDataReferenceReader.

Expected mixed-catalog behaviors:

- add, list, get, and remove operations function normally.
- search works on shared metadata keys like param and station_id.
- to_dataframe includes datastore metadata.
- arithmetic composition between references works via DataReference operators.

## Test Strategy

Standalone tests are implemented in tests/test_catalog_datastore.py.

Coverage groups:

1. DatastoreFilepathReader
- repository loads use `read_ts_repo` with identity and time range
- standalone loads use `read_ts` with filepath

2. DatastoreDataReference
- filepath or repository identity is required
- from_inventory_row builds normalized logical metadata
- getData works with the repository reader

3. DatastoreCatalogBuilder
- can_handle positive and negative cases
- build returns DatastoreDataReference objects with correct metadata
- reader sharing (flyweight) is preserved

4. Mixed catalog behavior
- datastore and non-datastore references coexist in one DataCatalog
- search and to_dataframe work across mixed sources
- arithmetic reference composition evaluates correctly

## Evolution Guidance

- Add convenience properties only when concrete downstream use cases require them.
- Keep metadata names aligned with existing filters/actions.
- If constructor or factory contracts change, update tests first to preserve portability and mixed-catalog guarantees.
