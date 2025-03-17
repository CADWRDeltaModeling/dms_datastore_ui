import panel as pn
pn.extension()

pn.pane.Markdown(
    """
    # DMS Datastore UI

    Welcome to the DMS Datastore UI, a tool for managing and exploring data in the DMS datastore.

    ## Available Tools

    - [Repository UI](/repoui) - Manage and explore repositories
    - [Data Screener](/data_screener) - Screen data for quality control
    - [Flag Editor](/flag_editor) - Edit data flags
    - [Map Inventory Explorer](/map_inventory_explorer) - Explore map inventories

    ## Getting Started

    1. Choose one of the tools above to get started
    2. Refer to the documentation for more information
    3. Contact the administrator if you have any issues

    ---
    *Powered by DMS Datastore UI*
    """
).servable("Delta Modeling Section Data Repository")
