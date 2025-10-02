import panel as pn
import param
import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
from holoviews import streams
from holoviews import opts, dim
from dms_datastore.read_ts import read_flagged, read_ts
from dms_datastore import auto_screen, write_ts
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
import io

hv.extension("bokeh")
pn.extension("codeeditor")

scatter_cycle = hv.Cycle(hv.Cycle("Category10").values[1:])
marker_cycle = hv.Cycle(
    [
        "o",
        "s",
        "d",
        "x",
        "*",
        "h",
        "v",
        "^",
        "<",
        ">",
        "p",
        "+",
        "D",
        "1",
        "2",
        "3",
        "4",
        "8",
        "|",
        "_",
    ]
)

data_screening_help_text = """
# Data Screening
This app is for screening data. It reads the data and associated meta data from the file path and displays
the values as well as the screening steps in yaml format.

## Instructions
1. Click on the button to plot the data.
2. Edit the yaml text to change the screening steps. The yaml text is a list of screening steps.
3. Click on the button to plot the data again.
4. Repeat steps 2 and 3 until satisfied with the screening."""


class DataScreener(param.Parameterized):
    """
    This class is a panel app for screening data.
    It reads the data and associated meta data from the file path and displays
    the values as well as the screening steps in yaml format.
    """

    time_range = param.CalendarDateRange(
        default=(datetime.now() - timedelta(days=1000), datetime.now()),
        doc="Time window for data. Default is last 1000 days",
    )

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.load(filepath)
        self.time_range = (
            self.df.last_valid_index() - pd.to_timedelta("1000D"),
            self.df.last_valid_index(),
        )

    def load(self, filepath):
        self.meta, self.df = read_flagged(
            filepath, apply_flags=False, return_flags=True, return_meta=True
        )

    def create_editor(self):
        screen = self.meta["screen"]
        text = yaml.dump(screen)
        self.editor = pn.widgets.CodeEditor(value=text, language="yaml")
        return self.editor

    @param.depends("time_range")
    def screen_and_plot(self):
        self.meta["screen"] = yaml.safe_load(self.editor.value)
        station_name = self.meta["station_name"]
        station_id = self.meta["station_id"]
        sublocation = self.meta["sublocation"]
        param = self.meta["param"]
        title = f"""{param}@{station_id}/{sublocation} :: {station_name}"""
        dfscreened, dfanomaly = auto_screen.screener(
            self.df,
            self.meta["station_id"],
            self.meta["sublocation"],
            self.meta["param"],
            self.meta["screen"],
            do_plot=False,
            return_anomaly=True,
        )
        screened_mask = (self.df["user_flag"] == 1) & self.df["user_flag"].notna()
        dfscreened = self.df["value"].mask(screened_mask)
        dfbad = self.df["value"].mask(~screened_mask)
        plot_with_screened = dfscreened.hvplot() * dfbad.hvplot(
            kind="scatter", color="red"
        )
        dfbads = [
            (self.df.mask(~dfanomaly.loc[:, col]), col) for col in dfanomaly.columns
        ]
        dfbads = pd.concat(
            [dfbad[["value"]].set_axis([col], axis=1) for dfbad, col in dfbads], axis=1
        )
        return (
            dfscreened.loc[slice(*self.time_range)].hvplot()
            * dfbads.loc[slice(*self.time_range)].hvplot.scatter(
                color=scatter_cycle, marker=marker_cycle
            )
        ).opts(title=title, ylabel=f'{self.meta["unit"]}')

    def do_screening(self, event):
        self.plot_panel.loading = True
        try:
            self.plot_panel.object = self.screen_and_plot()
        except Exception as e:
            self.plot_panel.object = hv.Div(f"<h3>Error: {e}</h3>")
        self.plot_panel.loading = False

    def save_screening(self):
        buffer = io.StringIO()
        write_ts.write_ts_csv(self.df, buffer, self.meta)
        buffer.seek(0)
        return buffer

    def view(self):
        self.plot_button = pn.widgets.Button(
            name="Plot", button_type="primary", icon="chart-line"
        )
        self.plot_button.on_click(self.do_screening)
        self.download_button = pn.widgets.FileDownload(
            button_type="success",
            icon="file-download",
            filename=f"{self.meta['station_id']}{self.meta['sublocation']}_{self.meta['param']}_screened_{self.time_range[1].year}.csv",
            callback=self.save_screening,
        )
        self.plot_panel = pn.panel(
            hv.Div("<h3>Click on button</h3>"),
            sizing_mode="stretch_both",
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
        self.create_editor()
        row1 = pn.Row(
            pn.Column(
                time_range_widget, pn.Row(self.plot_button, self.download_button)
            ),
            pn.Row(
                pn.pane.Markdown(data_screening_help_text),
                sizing_mode="stretch_width",
                align="center",
            ),
        )
        row2 = pn.Row(self.editor, self.plot_panel)
        return pn.Column(row1, row2)


def get_filepath(base_dir, station_id, subloc, param):
    # Implement the logic to generate the file path based on base_dir, station_id, subloc, and param
    # For example:
    return f"{base_dir}/{station_id}_{subloc}_{param}.csv"


# create using argparse that takes in a file path and then shows the data screener app
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("station_id", help="ID of the station")
    parser.add_argument("param", help="Parameter to screen")
    parser.add_argument("--subloc", help="Sublocation (optional)", default="")
    parser.add_argument(
        "--base_dir", help="Base directory for data files", default="/path/to/data"
    )
    args = parser.parse_args()

    print(f"Station ID: {args.station_id}")
    print(f"Parameter: {args.param}")
    print(f"Sublocation: {args.subloc}")
    print(f"Base Directory: {args.base_dir}")

    # Assuming you have a method to get the filepath from station_id, subloc, param, and base_dir
    filepath = get_filepath(args.base_dir, args.station_id, args.subloc, args.param)
    screener = DataScreener(filepath)
    screener.view().show()
