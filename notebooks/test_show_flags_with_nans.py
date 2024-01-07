# %%
import pandas as pd
import numpy as np
import dms_datastore_ui
from dms_datastore_ui import flag_editor
import holoviews as hv
from holoviews import streams
from holoviews import opts, dim

hv.extension("bokeh")
import panel as pn

pn.extension()


def create_test_dataframe(n=100):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(np.random.randn(n, 1), columns=["value"], index=dates)
    df["user_flag"] = np.nan
    df["user_flag"] = df["user_flag"].astype("object")
    df = df.reset_index().rename(columns={"index": "datetime"})
    return df


# opts.defaults(opts.Points(tools=["box_select", "lasso_select", "tap"]))
df = create_test_dataframe()
# %%
df
# %%
df.hvplot()
# %%
hv.Curve(df.mask((df["user_flag"].notna()) & (df["user_flag"] == "1")))
# %%
hv.Curve(df.mask(df["user_flag"] == "1"))
# %%
df.loc[5:20, "user_flag"] = "1"
df.loc[30:40, "user_flag"] = "0"
# %%
bad_flagged = df["user_flag"] == "1"
not_bad_flagged = df["user_flag"] == "0"
# %%
pts_cycle = hv.Cycle(hv.Cycle().values[1:])
hv.Curve(df.mask(bad_flagged)) * (
    hv.Points(df.mask(~bad_flagged)) * hv.Points(df.mask(~not_bad_flagged))
).opts(opts.Points(color=pts_cycle, size=5))
# %%
# %%
