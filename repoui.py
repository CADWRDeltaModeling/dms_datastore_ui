import panel as pn

pn.extension(
    "gridstack",
    "tabulator",
    "codeeditor",
    notifications=True,
    design="native",
    disconnect_notification="Connection lost, try reloading the page!",
    ready_notification="Application fully loaded.",
)
import datetime as dt
import sys
from dms_datastore_ui import fullscreen
import param
import holoviews as hv

hv.extension("bokeh")
main_panel = pn.Column(
    pn.indicators.LoadingSpinner(
        value=True, color="primary", size=50, name="Loading..."
    )
)
sidebar_panel = pn.Column(
    pn.indicators.LoadingSpinner(
        value=True, color="primary", size=50, name="Loading..."
    )
)
template = pn.template.VanillaTemplate(
    title="DMS Datastore",
    sidebar=[sidebar_panel],
    main=[main_panel],
    sidebar_width=650,
    header_color="blue",
    logo="https://sciencetracker.deltacouncil.ca.gov/themes/custom/basic/images/logos/DWR_Logo.png",
)


def load_explorer():
    import dms_datastore_ui.map_inventory_explorer as mie

    dir = "continuous"
    explorer = mie.StationInventoryExplorer(dir)
    te = explorer.create_view()

    # Clear existing content first
    main_panel.clear()
    sidebar_panel.clear()

    for obj in te.sidebar.objects:
        sidebar_panel.append(obj)

    # Add objects individually to ensure proper reactivity
    for obj in te.main.objects:
        main_panel.append(pn.panel(obj))

    # Add the disclaimer text to the modal
    template.modal.append(explorer.get_disclaimer_text())


def load_dataui():
    from dms_datastore_ui.datastore_uimgr import DatastoreUIMgr
    from pydelmod.dvue.dataui import DataUI

    dir = "continuous"
    uimgr = DatastoreUIMgr(dir)
    ui = DataUI(uimgr)
    view = ui.create_view()

    # Clear existing content first
    main_panel.clear()
    sidebar_panel.clear()

    # Add the DataUI components to the template
    if hasattr(view, "sidebar") and view.sidebar is not None:
        for obj in view.sidebar.objects:
            sidebar_panel.append(obj)

    if hasattr(view, "main") and view.main is not None:
        for obj in view.main.objects:
            main_panel.append(obj)
    else:
        # If no specific main panel structure, add the entire view
        main_panel.append(pn.panel(view))


def handle_hash_change(event):
    if event.new == "#dataui":
        load_dataui()
    else:
        load_explorer()


# Initialize the app based on current location hash
def init_app():
    # Access location via pn.state.location
    location = pn.state.location

    # Set up the hash watcher
    location.param.watch(handle_hash_change, "hash")

    # Initial load based on current hash
    if location.hash == "dataui":
        load_dataui()
    else:
        load_explorer()


pn.state.onload(init_app)

template.servable(title="DMS Datastore")
