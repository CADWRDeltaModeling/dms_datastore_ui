pip install git+https://github.com/CADWRDeltaModeling/dvue.git#egg=dvue --no-deps
pip install -e . --no-deps
dir="continuous"
if [ -f "caching.log" ]; then
    echo "caching.log exists"
else
    python repocache.py $dir 2>&1 | tee 'caching.log' &
    sleep 10 # wait for the cache initialization to finish
fi
#panel serve repoui.py --num-procs 1 --address 0.0.0.0 --port 80 --allow-websocket-origin="*"
# --keep-alive 30000: server sends a WebSocket ping every 30 s; if the browser misses it
# the socket is closed and the browser does a full reload instead of silently reconnecting
# with stale Bokeh model IDs (which produces harmless but noisy UnknownReferenceError logs).
# --unused-session-lifetime / --session-token-expiration kept at 30 days so long-running
# interactive sessions are not evicted while the browser tab is still open.
panel serve repoui.py --num-procs 1 --address 0.0.0.0 --port 80 --allow-websocket-origin="*" \
  --keep-alive 30000 \
  --unused-session-lifetime 2592000000 \
  --session-token-expiration 2592000000
