set -eu

python -m pip install --upgrade "panel>=1.11.1" "panel-material-ui>=0.17.0"
# Keep the screening dependency current when updating dms_datastore with --no-deps.
python -m pip install --upgrade --no-deps git+https://github.com/CADWRDeltaModeling/vtools3.git#egg=vtools
python -m pip install --upgrade --no-deps git+https://github.com/CADWRDeltaModeling/dvue.git#egg=dvue
python -m pip install --upgrade --no-deps git+https://github.com/CADWRDeltaModeling/dms_datastore.git#egg=dms_datastore
python -m pip install -e . --no-deps
python -c "from vtools.functions.error_detect import max_change_test; from dms_datastore import auto_screen"
# CARTO_API_KEY must be set in the environment (get a key at https://carto.com/basemaps/apikey)
# or basemap tiles will show CARTO's "API key required" watermark.
# The active cache stays on container-local disk; published snapshots (by default
# <repo parent>/cache_snapshots) are what survive a container restart.
export DMS_DATASTORE_UI_CACHE_ROOT="${DMS_DATASTORE_UI_CACHE_ROOT:-/tmp/dms_datastore_ui_cache}"
# Piped output is block-buffered by default, which hides warm-up progress in caching.log.
export PYTHONUNBUFFERED=1
repo_dir="${DMS_DATASTORE_UI_REPO_DIR:-continuous}"
repo_dir="$(cd "$repo_dir" && pwd)"
# Reuse or restore a cache snapshot for the current data before the UI opens the cache.
python repocache.py "$repo_dir" --restore-only || echo "Cache restore failed; the cache will be warmed instead"
# No-op when the cache is ready; otherwise warm it in the background and publish a snapshot.
python repocache.py "$repo_dir" 2>&1 | tee 'caching.log' &
# Use `python repoui.py` instead of `panel serve repoui.py`.
# The per_app_patterns cookie handler patch must run before BokehServer starts,
# which requires pn.serve() (module-level code runs once).  `panel serve` would
# re-execute the script per session, breaking the patch timing and the registry.
python repoui.py "$repo_dir"
