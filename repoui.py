import dms_datastore_ui.map_inventory_explorer as mie
import panel as pn
import datetime as dt
import sys

dir = "continuous"
explorer = mie.StationInventoryExplorer(dir)
#
ui = explorer.create_view().servable(title="Station Inventory Explorer")
