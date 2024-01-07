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
    df.index.rename("datetime", inplace=True)
    df["user_flag"] = np.nan
    df["user_flag"] = df["user_flag"].astype("object")
    df.iloc[5:20, 1] = "1"
    df.iloc[30:40, 1] = "0"
    return df


# opts.defaults(opts.Points(tools=["box_select", "lasso_select", "tap"]))
df = create_test_dataframe()
print(df.dtypes)
editor = flag_editor.FlagEditor(df)
editor.view().show()
