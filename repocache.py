import dms_datastore_ui.map_inventory_explorer as mie
import datetime as dt

if __name__ == '__main__':
    import sys
    print('Directory is ', sys.argv[1])
    explorer = mie.StationInventoryExplorer(sys.argv[1])
    print(f'Caching Task executed at: {dt.datetime.now()}')
    print('Clearing cache')
    explorer.station_datastore.clear_cache()
    print('Caching data')
    for repo_level in explorer.params.repo_level.objects:
        print(f'Cache level: {repo_level}')
        explorer.station_datastore.cache_repo_level(repo_level)
    print(f'Cache complete at: {dt.datetime.now()}')
#
