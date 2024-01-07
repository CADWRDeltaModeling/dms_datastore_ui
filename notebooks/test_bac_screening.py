# %%
import dms_datastore
from dms_datastore import read_ts, auto_screen
import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
import panel as pn

pn.extension()

# %%
repodir = "Y:/repo_staging/continuous"
fpath = f"{repodir}/screened/ncro*bac*ec*.csv"
meta, df = read_ts.read_flagged(
    fpath, apply_flags=False, return_flags=True, return_meta=True
)
# %%
df.head()
# Y:\repo_staging\continuous\plots\bac@_ec.png
# %%
df.hvplot()
# %%
# set df values to nan where flags are 1
dfs = df.copy()
dfs.mask(dfs.user_flag == "1", inplace=True)
# %%
dfs.head()
# %%
dfs.hvplot()
# %%
meta
# %%
dfscreened, dfanomaly = auto_screen.screener(
    df, "bac", "", "ec", meta["screen"], do_plot=False, return_anomaly=True
)
# %%
dfscreened.head()
# %%
# dfanomaly.head()

# %%
screened_mask = (df["user_flag"] == 1) & df["user_flag"].notna()
dfscreened = df["value"].mask(screened_mask)
dfbad = df["value"].mask(~screened_mask)
# %%
dfscreened.hvplot() * dfbad.hvplot(kind="scatter", color="red")
# %%
dfanomaly.columns
# %%
dfbads = [(df.mask(~dfanomaly.loc[:, col]), col) for col in dfanomaly.columns]
dfbads = pd.concat(
    [dfbad[["value"]].set_axis([col], axis=1) for dfbad, col in dfbads], axis=1
)
# %%
header_info = meta["original_header"].split("\n")
title = header_info[1]
# %%
(dfscreened.hvplot() * dfbads.hvplot.scatter()).opts(title=title)
# %%
