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
    df["user_flag"] = ""
    df = df.reset_index().rename(columns={"index": "datetime"})
    return df


# opts.defaults(opts.Points(tools=["box_select", "lasso_select", "tap"]))
df = create_test_dataframe()
checker = flag_editor.FlagChecker(df)
checker_dash = pn.Column(checker.param, checker.view)
# pn.Row(checker.view, checker.view_tabulator))
checker_dash.show()
