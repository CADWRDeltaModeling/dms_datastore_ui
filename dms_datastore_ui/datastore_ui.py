# %%
import click
import geopandas as gpd
from pydelmod.dvue.dataui import DataUI
from dms_datastore_ui.datastore_uimgr import DatastoreUIMgr


# %%
# @click.command()
def show_repo_ui(dir):
    """
    show the data ui for a repository
    """
    uimgr = DatastoreUIMgr(dir)
    ui = DataUI(uimgr)
    return ui.create_view().show()


# %%
# show_repo_ui("../tests/repo/continuous")
show_repo_ui("y:/repo/continuous")
# %%
