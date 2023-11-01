if [ -f "caching.log" ]; then
    echo "caching.log exists"
else
    python repocache.py continuous_station_repo_beta > 'caching.log' 2>&1 &
fi
panel serve repoui.py --num-procs 2 --address 0.0.0.0 --port 80 --allow-websocket-origin="*"