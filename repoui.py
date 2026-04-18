import panel as pn

pn.extension(
    "gridstack",
    "tabulator",
    "codeeditor",
    notifications=True,
    reconnect=True,
    design="native",
    disconnect_notification="Connection lost, try reloading the page!",
    ready_notification="Application fully loaded.",
)
import datetime as dt
import sys
from dms_datastore_ui import fullscreen
import param
import holoviews as hv
import geoviews as gv

hv.extension("bokeh")
gv.extension("bokeh")

# Pre-import the heavy UI modules NOW (before any Panel session starts) so that
# their module-level hv/gv/pn.extension() calls happen here, not inside a live
# Bokeh session.  Importing them lazily inside load_explorer()/load_dataui()
# would re-run those extension calls mid-session and reset pn.state.curdoc,
# causing the main-area spinner to persist on first load.
import dms_datastore_ui.map_inventory_explorer  # noqa: F401 – triggers gv/hv extension init
from dms_datastore_ui.datastore_uimgr import DatastoreUIMgr  # noqa: F401
from dvue.dataui import DataUI  # noqa: F401 – triggers gv/hv extension init
import cartopy.crs as ccrs

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
    logo="dms_datastore_ui/california-department-of-water-resources-logo.png",
)


def load_explorer():
    explorer = dms_datastore_ui.map_inventory_explorer.StationInventoryExplorer("continuous")
    view = explorer.create_view()

    sidebar_items = list(view.sidebar)
    main_items = list(view.main)
    modal_items = list(view.modal)
    view.sidebar.clear()
    view.main.clear()
    view.modal.clear()

    sidebar_panel.objects = sidebar_items
    main_panel.objects = main_items

    template.modal.clear()
    for item in modal_items:
        template.modal.append(item)


def load_dataui():
    uimgr = DatastoreUIMgr("continuous")
    ui = DataUI(uimgr, crs=ccrs.epsg(26910))
    ui_template = ui.create_view()

    sidebar_items = list(ui_template.sidebar)
    main_items = list(ui_template.main)
    modal_items = list(ui_template.modal)
    ui_template.sidebar.clear()
    ui_template.main.clear()
    ui_template.modal.clear()

    sidebar_panel.objects = sidebar_items
    main_panel.objects = main_items

    template.modal.clear()
    for item in modal_items:
        template.modal.append(item)


def handle_hash_change(event):
    if event.new == "#newui":
        load_dataui()
    else:
        load_explorer()  # plain URL or #oldui both show the explorer


# Initialize the app based on current location hash
def init_app():
    # URL fragments are client-side only — not sent in the HTTP request.
    # Panel syncs location.hash via the WebSocket *after* onload fires.
    # For "#newui" this changes "" → "#newui", triggering the watcher below.
    # For a plain URL the hash stays "" and Panel may skip setting it (no
    # param event fired), so we add a one-shot fallback callback that runs
    # ~300 ms later — by then the WebSocket sync is complete and
    # location.hash has its definitive value.

    location = pn.state.location
    has_loaded = [False]  # guards the fallback timer only, not the watcher

    def _on_hash_change(event):
        # Fires for both the initial browser→Panel sync AND user-edited URL
        # changes, so never guard this with has_loaded.
        has_loaded[0] = True
        if event.new == "#newui":
            load_dataui()
        else:
            load_explorer()

    location.param.watch(_on_hash_change, "hash")

    # Fallback: fires once after 300 ms for plain URLs where the hash watcher
    # never triggers (location.hash stays "" and Panel skips the param update).
    def _fallback():
        if not has_loaded[0]:
            has_loaded[0] = True
            if location.hash == "#newui":
                load_dataui()
            else:
                load_explorer()

    pn.state.add_periodic_callback(_fallback, period=300, count=1)


pn.state.onload(init_app)

template.servable(title="DMS Datastore")
