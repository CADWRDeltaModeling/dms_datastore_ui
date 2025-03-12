import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
import panel as pn

hv.extension("bokeh")
pn.extension()


class GapVisualizer:
    def __init__(self, df, row_info, column_name="value"):
        self.df = df
        self.row_info = row_info
        self.column_name = column_name

    def visualize_gap(self):
        """
        Using curve and spikes display the missing data.
        """
        missing_df = self.df[self.df[self.column_name].isna()][[self.column_name]]
        index_name = self.df.index.name if self.df.index.name is not None else "index"
        # Create a curve plot
        curve = hv.Curve(
            self.df,
            index_name,
            self.column_name,
            label=f"{self.row_info['station_id']}_{self.row_info['subloc']}_{self.row_info['param']}",
        ).opts(
            height=300,
            title="Time Series Data",
            xlabel="Time",
            ylabel=self.column_name,
            responsive=True,
        )
        # Create a spikes plot
        spikes = hv.Spikes(missing_df, index_name, [], label="Missing").opts(
            height=120,
            title="Missing Data Visualization",
            xlabel="Time",
            ylabel="Series",
            responsive=True,
        )
        return (
            (curve + spikes)
            .opts(
                opts.Curve(
                    xaxis=None,
                    line_width=1.50,
                    color="red",
                    tools=["hover"],
                ),
                opts.Spikes(yaxis=None, line_width=0.5, color="grey"),
            )
            .opts(
                sizing_mode="stretch_width",
            )
            .cols(1)
        )
