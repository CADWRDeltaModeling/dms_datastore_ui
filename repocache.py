import dms_datastore_ui.map_inventory_explorer as mie
import datetime as dt

if __name__ == '__main__':
    import sys
    print('Directory is ', sys.argv[1])
    explorer = mie.StationInventoryExplorer(sys.argv[1])
    print(f'Caching Task executed at: {dt.datetime.now()}')
    for repo_level in explorer.param.repo_level.objects:
        print(f'Cache level: {repo_level}')
        explorer.cache(repo_level)
    print(f'Cache complete at: {dt.datetime.now()}')
#
