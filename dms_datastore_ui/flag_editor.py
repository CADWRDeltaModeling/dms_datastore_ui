import panel as pn
import param
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import streams
from holoviews import opts, dim
from dms_datastore import read_ts, auto_screen, write_ts
from datetime import datetime, timedelta
import io

hv.extension("bokeh")
pn.extension()

help_text = """
# Flag Editor
This app is for flagging data. It reads the data and associated meta data from the file path and displays
the time series data as well as the flags in a graph.
The user can then select individual points or a rectangle of points or a lasso of points and mark them as BAD or NOT BAD.
The flags are stored in the user_flag column of the data frame.
The user can then save the data frame to a file.
## Instructions
1. Click on the button to plot the data.
2. Select whether to mark the selected points as BAD or NOT BAD.
2. Select points or a rectangle or a lasso of points.
4. Repeat steps 2 and 3 until satisfied with the flags.
5. Click on the Save button to save the data frame to a file.
"""


class FlagEditor(param.Parameterized):
    flag = param.ObjectSelector(default="BAD", objects=["NOT BAD", "BAD"])
    time_range = param.CalendarDateRange(
        default=(datetime.now() - timedelta(days=1000), datetime.now()),
        doc="Time window for data. Default is last 1000 days",
    )

    def __init__(self, filepath, **kwargs):
        super().__init__(
            **kwargs
        )  # param.Parameterized requires calling their super first
        self.filepath = filepath
        meta, df = read_ts.read_flagged(
            filepath, apply_flags=False, return_flags=True, return_meta=True
        )

        self.init(df, meta)

    def init(self, df, meta):
        self.meta = meta
        self.x_col_name = "datetime"
        self.y_col_name = "value"
        self.y_label = (
            f"{self.meta['station_id']}/{self.meta['param']}({self.meta['unit']})"
        )
        self.flag_col_name = "user_flag"
        self.flag_map = {"NOT BAD": "0", "BAD": "1"}
        self.dforiginal = df
        self.time_range = (
            self.dforiginal.last_valid_index() - pd.to_timedelta("1000D"),
            self.dforiginal.last_valid_index(),
        )
        df = df.reset_index()  # done to get the indexes correctly
        self.value_col_index = df.columns.get_loc(self.y_col_name)
        self.flag_col_index = df.columns.get_loc(self.flag_col_name)
        self.time_range_changed()

    @param.depends("time_range", watch=True)
    def time_range_changed(self):
        self.df = self.dforiginal.loc[slice(*self.time_range)].copy().reset_index()
        self.dff = self.df.copy()
        self.points = hv.Points(self.df, kdims=[self.x_col_name, self.y_col_name]).opts(
            alpha=0,
        )
        # Declare points as source of selection stream
        self.selection = streams.Selection1D(source=self.points)
        self.dmap = hv.DynamicMap(self.mark_flag, streams=[self.selection])
        self.tabulator = pn.widgets.Tabulator(self.df)

    def view(self):
        self.plot_button = pn.widgets.Button(
            name="Plot", button_type="primary", icon="chart-line"
        )
        self.plot_button.on_click(self.do_plot)
        self.plot_panel = pn.panel(
            hv.Div("<h3>Click on button</h3>"),
            sizing_mode="stretch_both",
        )
        self.flag_button = pn.widgets.Button(
            name="Mark Flags", button_type="primary", icon="flag"
        )
        self.flag_button.on_click(self.do_mark_on_selected)
        self.download_button = pn.widgets.FileDownload(
            file=f"{self.meta['station_id']}{self.meta['sublocation']}_{self.meta['param']}_flagged_{self.time_range[1].year}.csv",
            button_type="success",
            icon="file-download",
            callback=self.save_data,
        )
        time_range_widget = pn.Param(
            self.param.time_range,
            widgets={
                "time_range": {
                    "widget_type": pn.widgets.DatetimeRangeInput,
                    "format": "%Y-%m-%d %H:%M",
                }
            },
        )
        flag_widget = pn.Param(self.param.flag)
        row1 = pn.Row(
            pn.Column(
                time_range_widget,
                flag_widget,
                pn.Row(self.plot_button, self.download_button),
            ),
            pn.Row(
                pn.pane.Markdown(help_text),
                sizing_mode="stretch_width",
                align="center",
            ),
        )
        row2 = pn.Row(self.plot_panel)
        return pn.Column(row1, row2)

    def do_plot(self, event):
        self.plot_panel.loading = True
        self.plot_panel.object = self.make_plot()
        self.plot_panel.loading = False

    def save_data(self):
        buffer = io.StringIO()
        write_ts.write_ts_csv(self.dff, buffer, self.meta)
        buffer.seek(0)
        return buffer

    def make_plot(self):
        plot = self.points * self.dmap
        return plot.opts(
            responsive=True,
            title=f"{self.meta['station_name']} ({self.meta['param']} [{self.meta['unit']}])",
        ).opts(
            opts.Points(
                tools=["box_select", "lasso_select", "tap"],
                active_tools=["box_select", "tap", "wheel_zoom"],
                ylabel=self.y_label,
            )
        )

    def view_tabulator(self):
        return self.tabulator

    def update_flags(self, index, flag_value):
        if len(index) > 0:
            if flag_value == "0":  # only update if flag_value is 1
                # find values that are already flagged as 1
                self.dff.loc[
                    self.dff.index.isin(index)
                    & (self.dff.iloc[:, self.flag_col_index] == "1"),
                    self.flag_col_name,
                ] = "0"  # set the values that are already flagged as 1 to 0
            else:
                self.dff.iloc[index, [self.flag_col_index]] = flag_value
            self.tabulator.patch(
                {self.flag_col_name: [(idx, flag_value) for idx in index]}
            )
            # self.selection.param.trigger("index")

    def do_mark_on_selected(self, event):
        self.mark_flag(self.selection.index)

    def mark_flag(self, index):
        self.plot_panel.loading = True
        self.update_flags(index, self.flag_map[self.flag])
        bad_flagged = self.dff["user_flag"] == "1"
        not_bad_flagged = self.dff["user_flag"] == "0"

        crv = hv.Curve(self.dff.mask(bad_flagged), kdims=[self.x_col_name])
        bad_pts = hv.Points(
            self.dff.loc[bad_flagged], kdims=[self.x_col_name, self.y_col_name]
        ).opts(marker="x", color="red", size=10)
        not_bad_pts = hv.Points(
            self.dff.loc[not_bad_flagged], kdims=[self.x_col_name, self.y_col_name]
        ).opts(marker="o", color="green", size=10)
        self.plot_panel.loading = False
        return crv * (bad_pts * not_bad_pts)


# create using argparse that takes in a file path and then shows the data screener app
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="path to the data file")
    args = parser.parse_args()
    editor = FlagEditor(args.filepath)
    editor.view().show()
