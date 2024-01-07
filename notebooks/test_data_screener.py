# %%
import dms_datastore_ui
from dms_datastore_ui import data_screener

ds = data_screener.DataScreener(
    "Y:/repo_staging/continuous/screened/ncro_ben_b94175_elev_*.csv"
)
# %%
ui = ds.view()
# %%
ui.show()
# %%
