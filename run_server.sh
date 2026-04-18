dir="continuous"
if [ -f "caching.log" ]; then
    echo "caching.log exists"
else
    python repocache.py $dir > 'caching.log' 2>&1 &
    sleep 10 # wait for the cache initialization to finish
fi
pip install git+https://github.com/CADWRDeltaModeling/dvue.git#egg=dvue --no-deps
pip install -e . --no-deps
#panel serve repoui.py --num-procs 1 --address 0.0.0.0 --port 80 --allow-websocket-origin="*"
panel serve repoui.py --num-procs 1 --address 0.0.0.0 --port 80 --allow-websocket-origin="*" --unused-session-lifetime 2592000000 --session-token-expiration 2592000000
