# %%
import dms_datastore
from dms_datastore import read_ts

# %%
dir = "Y:/repo_staging/continuous/screened"
# %%

meta, tsf = read_ts.read_flagged(
    f"{dir}/ncro_ben_b94175_elev_*.csv",
    apply_flags=False,
    return_flags=True,
    return_meta=True,
)

# %%
tsf[~tsf["user_flag"].isna()]
# %%
meta["screen"]
# %%
meta["screen"]["steps"]
# %%
import dms_datastore_ui
from dms_datastore_ui import flag_editor
import panel as pn

pn.extension()

# %%
tsf["user_flag"] = tsf["user_flag"].fillna("")
checker = flag_editor.FlagChecker(tsf.reset_index())
checker_dash = pn.Column(checker.param, checker.view)
# pn.Row(checker.view, checker.view_tabulator))
checker_dash.show()

# %%
