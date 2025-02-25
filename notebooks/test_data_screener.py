# %%
import dms_datastore_ui
from dms_datastore_ui import data_screener

file_pattern = "Y:/repo/continuous/screened/des_anh@upper*_ec_*.csv"
# file_pattern = r"\\nasbdo\Modeling_Data\repo\continuous\screened\des_anh*_ec_*.csv"
ds = data_screener.DataScreener(file_pattern)
ui = ds.view()
ui.show()
# %%
