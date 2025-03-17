import dms_datastore_ui.map_inventory_explorer as mie
import panel as pn

pn.extension("gridstack", "tabulator", notifications=True, design="native")
import datetime as dt
import sys


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
        print(obj)
        main_panel.append(pn.panel(obj))

    # Add the disclaimer text to the modal
    template.modal.append(explorer.get_disclaimer_text())


pn.state.onload(load_explorer)

template.servable(title="Station Inventory Explorer")
