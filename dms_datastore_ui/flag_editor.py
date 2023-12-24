import panel as pn
import param
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import streams
from holoviews import opts, dim

hv.extension("bokeh")
pn.extension()


class FlagChecker(param.Parameterized):
    flag = param.ObjectSelector(default="BAD", objects=["UNCHECKED", "BAD"])

    def __init__(self, df, **kwargs):
        super().__init__(
            **kwargs
        )  # param.Parameterized requires calling their super first
        self.init(df)

    def init(self, df):
        self.x_col_name = "datetime"
        self.y_col_name = "value"
        self.flag_col_name = "user_flag"
        self.flag_map = {"BAD": "1", "UNCHECKED": ""}
        self.flag2markers = dict(zip(["1", ""], ["x", "o"]))
        self.flag2colors = dict(zip(["1", ""], ["red", "blue"]))
        self.df = df
        self.value_col_index = df.columns.get_loc(self.y_col_name)
        self.flag_col_index = df.columns.get_loc(self.flag_col_name)
        # copy with flags setting col 2 values for NA or BAD values
        self.dfg = self.df.copy()
        self.dff = self.df.copy()
        self._update_dfg_vals()
        self.points = hv.Points(self.df, kdims=[self.x_col_name, self.y_col_name]).opts(
            alpha=0
        )
        # Declare points as source of selection stream
        self.selection = streams.Selection1D(source=self.points)
        self.dmap = hv.DynamicMap(self.mark_flag, streams=[self.selection])
        self.tabulator = pn.widgets.Tabulator(self.df)

    @param.depends("flag")
    def view(self):
        curves = self.points * self.dmap
        return curves.opts(
            width=700, height=400, title="Ready to mark selections: " + self.flag
        ).opts(
            opts.Points(
                tools=["box_select", "lasso_select", "tap"],
                active_tools=["lasso_select", "tap", "wheel_zoom"],
            )
        )

    def view_tabulator(self):
        return self.tabulator

    def _update_dfg_vals(self):
        self.dfg.iloc[:, self.value_col_index] = self.df.iloc[:, self.value_col_index]
        self.dfg[self.y_col_name] = self.dfg[self.y_col_name].mask(
            self.df[self.flag_col_name] != ""
        )
        self.dff = self.df[self.df[self.flag_col_name] != ""]

    def update_flags(self, index, flag_value):
        if len(index) > 0:
            self.df.iloc[index, [self.flag_col_index]] = flag_value
            self.tabulator.patch(
                {self.flag_col_name: [(idx, flag_value) for idx in index]}
            )
            self._update_dfg_vals()
            self.selection.param.trigger("index")

    def mark_flag(self, index):
        self.update_flags(index, self.flag_map[self.flag])
        crv_layout = hv.Curve(self.dfg, kdims=[self.x_col_name]).opts(color="blue")
        pts_layout = hv.Points(
            self.dff,
            kdims=[self.x_col_name, self.y_col_name],
        ).opts(
            marker=dim(self.flag_col_name).categorize(self.flag2markers),
            color=dim(self.flag_col_name).categorize(self.flag2colors),
            size=5,
        )
        return crv_layout * pts_layout
