pip install git+https://github.com/CADWRDeltaModeling/dvue.git#egg=dvue --no-deps
pip install -e . --no-deps
# CARTO_API_KEY must be set in the environment (get a key at https://carto.com/basemaps/apikey)
# or basemap tiles will show CARTO's "API key required" watermark.
dir="continuous"
if [ -f "caching.log" ]; then
    echo "caching.log exists"
else
    python repocache.py $dir 2>&1 | tee 'caching.log' &
    sleep 10 # wait for the cache initialization to finish
fi
# Use `python repoui.py` instead of `panel serve repoui.py`.
# The per_app_patterns cookie handler patch must run before BokehServer starts,
# which requires pn.serve() (module-level code runs once).  `panel serve` would
# re-execute the script per session, breaking the patch timing and the registry.
python repoui.py $dir
