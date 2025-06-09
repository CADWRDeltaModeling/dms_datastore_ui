import os
from pathlib import Path
from datetime import datetime, timedelta
from io import StringIO
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
# viz and ui
import hvplot.pandas
import holoviews as hv
from holoviews import opts, dim, streams

hv.extension("bokeh")
import cartopy
import geoviews as gv

gv.extension("bokeh")
import panel as pn

pn.extension("tabulator", notifications=True, design="native")

import param

#!pip install diskcache
import diskcache
import dms_datastore
from dms_datastore.read_ts import read_ts, read_flagged
from dms_datastore_ui import data_screener, flag_editor

#
from vtools.functions.filter import godin, cosine_lanczos

#
from . import fullscreen


def uniform_unit_for(param):
    if param == "elev":
        return "feet"
    elif param == "flow":
        return "ft^3/s"
    elif param == "ec":
        return "microS/cm"
    elif param == "temp":
        return "deg_c"
    elif param == "do":
        return "mg/l"
    elif param == "ssc":
        return "mg/l"
    elif param == "turbidity":
        return "NTU"
    elif param == "ph":
        return "pH"
    elif param == "velocity":
        return "ft/s"
    elif param == "cla":
        return "ug/l"
    else:
        return "std unit"


def to_uniform_units(df, param, unit):
    """
    elev, feet, meters
    flow, ft^3/s
    ec, uS/cm, microS/cm
    temp, deg_f, deg_c
    do, mg/l
    ssc, mg/l
    turbidity, FNU, NTU
    ph, pH, std unit
    velocity, ft/s
    cla, ug/l
    """
    if param == "elev":
        if unit == "meters":
            df["value"] = df["value"] * 3.28084
            unit = "feet"
    elif param == "ec":
        if unit == "uS/cm":
            unit = "microS/cm"
    elif param == "temp":
        if unit == "deg_f":
            df["value"] = (df["value"] - 32) * 5 / 9
            unit = "deg_c"
    elif param == "turbidity":
        if unit == "FNU":
            unit = "NTU"
    elif param == "ph":
        if unit == "std unit":
            unit = "pH"
    return df, unit


# this should be a util function
def find_lastest_fname(pattern, dir="."):
    d = Path(dir)
    fname, mtime = None, 0
    for f in d.glob(pattern):
        fmtime = f.stat().st_mtime
        if fmtime > mtime:
            mtime = fmtime
            fname = f.absolute()
    return fname, mtime


# from stackoverflow.com https://stackoverflow.com/questions/6086976/how-to-get-a-complete-exception-stack-trace-in-python
def full_stack():
    import traceback, sys

    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]  # remove call of full_stack, the printed exception
        # will contain the caught exception caller instead
    trc = "Traceback (most recent call last):\n"
    stackstr = trc + "".join(traceback.format_list(stack))
    if exc is not None:
        stackstr += "  " + traceback.format_exc().lstrip(trc)
    return stackstr


class StationDatastore(param.Parameterized):
    # define a class to hold the station inventory and retrieve data. Move the caching code here
    repo_level = param.ListSelector(
        objects=["screened"],
        default=["screened"],
        doc="repository level (sub directory) under which data is found. You can select multiple repo levels (ctrl+click)",
    )
    parameter_type = param.ListSelector(
        objects=["all"],
        default=["all"],
        doc="parameter type of data, e.g. flow, elev, temp, etc. You can select multiple parameters (ctrl+click) or all",
    )
    apply_filter = param.Boolean(default=False, doc="Apply tidal filter to data")
    filter_type = param.Selector(
        objects=["cosine_lanczos", "godin"],
        default="cosine_lanczos",
        doc="Filter type is cosine lanczos with a 40 hour cutoff or godin",
    )
    fill_gap = param.Integer(default=4, doc="Maximum fill gap in data")
    convert_units = param.Boolean(default=True, doc="Convert units to uniform units")
    caching = param.Boolean(default=True, doc="Use caching")

    def __init__(self, dir, **kwargs):
        super().__init__(**kwargs)
        self.dir = os.path.normpath(dir)
        if not os.path.exists(self.dir):
            raise Exception(f"Directory {self.dir} does not exist")
        self.cache = diskcache.Cache(
            "cache_" + self.last_part_path(self.dir), size_limit=1e11
        )
        self.caching_read_ts = self.cache.memoize()(read_ts)
        # check that repo_levels are valid and set default to first valid
        valid_repo_levels = []
        for repo_level in self.param.repo_level.objects:
            if os.path.exists(os.path.join(self.dir, repo_level)):
                valid_repo_levels.append(repo_level)
            else:
                print(f"{repo_level} doesn't exist in {self.dir}")
        if len(valid_repo_levels) == 0:
            raise ValueError(
                f"No valid repos found in {self.dir}: valid repos are {self.param.repo_level.objects}"
            )
        self.param.repo_level.objects = valid_repo_levels
        self.param.repo_level.default = valid_repo_levels[0]
        # self.repo_level = valid_repo_levels[0]  # select the first valid repo_level
        # read inventory file for each repo level
        self.inventory_file, mtime = find_lastest_fname(
            f"inventory_datasets_{self.repo_level}*.csv", self.dir
        )
        if not self.inventory_file:
            raise FileNotFoundError(
                f"Could not find inventory_datasets_{self.repo_level}*.csv file in {self.dir}"
            )
        print("Using inventory file: ", self.inventory_file)
        self.df_dataset_inventory = pd.read_csv(
            os.path.join(self.dir, self.inventory_file)
        )
        # replace nan with empty string for column subloc
        self.df_dataset_inventory["subloc"] = self.df_dataset_inventory[
            "subloc"
        ].fillna("")
        self.unique_params = self.df_dataset_inventory["param"].unique()
        self.param.parameter_type.objects = ["all"] + list(self.unique_params)
        group_cols = [
            "station_id",
            "subloc",
            "name",
            "unit",
            "param",
            "min_year",
            "max_year",
            "agency",
            "agency_id_dbase",
            "lat",
            "lon",
        ]
        self.df_station_inventory = (
            self.df_dataset_inventory.groupby(group_cols)
            .count()
            .reset_index()[group_cols]
        )
        # calculate min (min year) and max of max_year
        self.min_year = self.df_station_inventory["min_year"].min()
        self.max_year = self.df_station_inventory["max_year"].max()

    def last_part_path(self, dir):
        return os.path.basename(os.path.normpath(dir))

    def get_data(self, repo_level, filename):
        if self.caching:
            return self.caching_read_ts(self.get_data_filepath(repo_level, filename))
        else:
            return read_ts(self.get_data_filepath(repo_level, filename))

    def get_data_filepath(self, repo_level, filename):
        filepath = os.path.join(self.dir, repo_level, filename)
        return filepath

    def clear_cache(self):
        if self.caching:
            self.cache.clear()
            print("Cache cleared")

    def cache_repo_level(self, repo_level):
        # get unique filenames
        if not self.caching:
            raise Exception("Caching is not enabled")
        filenames = self.df_dataset_inventory["filename"].unique()
        print("Caching: ", len(filenames), " files")
        for i, filename in enumerate(filenames):
            print(f"Caching {i} ::{filename}")
            try:
                self.get_data(repo_level, filename)
            except Exception as e:
                print(e)
                print("Skipping", filename, "due to error")

    def get_uniform_units_data(self, df, param, unit):
        if self.convert_units:
            df, unit = to_uniform_units(df, param, unit)
        return df, unit

    def get_filtered_data(self, df):
        if self.apply_filter:
            if self.fill_gap > 0:
                df = df.interpolate(limit=self.fill_gap)
            if self.filter_type == "cosine_lanczos":
                if len(df) > 0:
                    df["value"] = cosine_lanczos(df["value"], "40H")
            else:
                if len(df) > 0:
                    df["value"] = godin(df["value"])
        return df


from bokeh.models import HoverTool
from bokeh.core.enums import MarkerType

# print(list(MarkerType))
# ['asterisk', 'circle', 'circle_cross', 'circle_dot', 'circle_x', 'circle_y', 'cross', 'dash',
# 'diamond', 'diamond_cross', 'diamond_dot', 'dot', 'hex', 'hex_dot', 'inverted_triangle', 'plus',
# 'square', 'square_cross', 'square_dot', 'square_pin', 'square_x', 'star', 'star_dot',
# 'triangle', 'triangle_dot', 'triangle_pin', 'x', 'y']
param_to_marker_map = {
    "elev": "square",
    "predictions": "square_x",
    "turbidity": "diamond",
    "flow": "circle",
    "velocity": "circle_dot",
    "temp": "cross",
    "do": "asterisk",
    "ec": "triangle",
    "ssc": "diamond",
    "ph": "plus",
    "salinity": "inverted_triangle",
    "cla": "dot",
    "fdom": "hex",
}


def get_color_dataframe(stations, color_cycle=hv.Cycle()):
    """
    Create a dataframe with station names and colors
    """
    cc = color_cycle.values
    # extend cc to the size of stations
    while len(cc) < len(stations):
        cc = cc + cc
    dfc = pd.DataFrame({"stations": stations, "color": cc[: len(stations)]})
    dfc.set_index("stations", inplace=True)
    return dfc


def get_colors(stations, dfc):
    """
    Create a dictionary with station names and colors
    """
    return hv.Cycle(list(dfc.loc[stations].values.flatten()))


class StationInventoryExplorer(param.Parameterized):
    """
    Show station inventory on map and select to display data available
    Furthermore select the data rows and click on button to display plots for selected rows
    """

    time_range = param.CalendarDateRange(
        default=(datetime.now() - timedelta(days=10), datetime.now()),
        doc="Time window for data. Default is last 10 days",
    )
    map_color_category = param.Selector(
        objects=["param", "agency"], default="param", doc="Color by parameter or agency"
    )
    use_symbols_for_params = param.Boolean(
        default=False,
        doc="Use symbols for parameters. If not selected, all parameters will be shown as circles",
    )
    search_text = param.String(default="", doc="Search text to filter stations")
    show_legend = param.Boolean(default=True, doc="Show legend")
    legend_position = param.Selector(
        objects=["top_right", "top_left", "bottom_right", "bottom_left"],
        default="top_right",
        doc="Legend position",
    )
    sensible_range_yaxis = param.Boolean(
        default=False,
        doc="Sensible range (1st and 99th percentile) or auto range for y axis",
    )
    year_range = param.Range(step=1, bounds=(2000, 2010), doc="Year range for data")
    query = param.String(
        default="",
        doc='Query to filter stations. See <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html">Pandas Query</a> for details. E.g. max_year <= 2023',
    )

    def __init__(self, dir, **kwargs):
        super().__init__(**kwargs)
        self.station_datastore = pn.state.as_cached(
            "datastore", lambda x: StationDatastore(x), x=dir
        )
        self.param.year_range.bounds = (
            self.station_datastore.min_year,
            self.station_datastore.max_year,
        )
        self.year_range = (
            self.station_datastore.min_year,
            self.station_datastore.max_year,
        )
        self.tmap = gv.tile_sources.CartoLight
        tooltips = [
            ("Station ID", "@station_id"),
            ("SubLoc", "@subloc"),
            ("Name", "@name"),
            ("Years", "@min_year to @max_year"),
            ("Agency", "@agency - @agency_id_dbase"),
            ("Parameter", "@param"),
            ("Unit", "@unit"),
        ]
        hover = HoverTool(tooltips=tooltips)
        self.current_station_inventory = self.station_datastore.df_station_inventory
        self.map_station_inventory = gv.Points(
            self.current_station_inventory, kdims=["lon", "lat"]
        ).opts(
            size=6,
            color=dim(self.map_color_category),
            cmap="Category10",
            # marker=dim('param').categorize(param_to_marker_map),
            tools=[hover],
            height=800,
            projection=cartopy.crs.GOOGLE_MERCATOR,
        )
        self.map_station_inventory = self.map_station_inventory.opts(
            opts.Points(
                tools=["tap", hover, "lasso_select", "box_select"],
                nonselection_alpha=0.3,  # nonselection_color='gray',
                size=10,
            )
        ).opts(frame_width=500, active_tools=["wheel_zoom"])

        self.station_select = streams.Selection1D(
            source=self.map_station_inventory
        )  # .Points.I)

    def show_inventory(self, index):
        if len(index) == 0:
            index = slice(None)
        dfs = self.current_station_inventory.iloc[index]
        # return a UI with controls to plot and show data
        return self.update_data_table(dfs)

    def get_param_to_marker_map(self):
        if self.use_symbols_for_params:
            return param_to_marker_map
        else:
            return {p: "circle" for p in self.station_datastore.unique_params}

    @param.depends("search_text", watch=True)
    def do_search(self):
        # Create a boolean mask to select rows with matching text
        mask = self.current_station_inventory.apply(
            lambda row: row.astype(str)
            .str.contains(self.search_text, case=False)
            .any(),
            axis=1,
        )
        # Use the boolean mask to select the matching rows
        index = self.current_station_inventory.index[mask]
        self.station_select.event(
            index=list(index)
        )  # this should trigger show_inventory

    def _append_to_title_map(self, title_map, unit, r, repo_level):
        value = title_map[unit]
        if repo_level not in value[0]:
            value[0] += f",{repo_level}"
        if r["station_id"] not in value[2]:
            value[2] += f',{r["station_id"]}'
        if r["agency"] not in value[3]:
            value[3] += f',{r["agency"]}'
        title_map[unit] = value

    def _create_title(self, v):
        title = f"{v[1]} @ {v[2]} ({v[3]}::{v[0]})"
        return title

    def _calculate_range(self, current_range, df, factor=0.1):
        if df.empty:
            return current_range
        else:
            new_range = df.iloc[:, 1].quantile([0.05, 0.995]).values
            scaleval = new_range[1] - new_range[0]
            new_range = [
                new_range[0] - scaleval * factor,
                new_range[1] + scaleval * factor,
            ]
        if current_range is not None:
            new_range = [
                min(current_range[0], new_range[0]),
                max(current_range[1], new_range[1]),
            ]
        return new_range

    def create_plots(self, event):
        # df = self.display_table.selected_dataframe # buggy
        df = self.display_table.value.iloc[self.display_table.selection]
        df = df.merge(self.station_datastore.df_dataset_inventory)
        try:
            layout_map = {}
            title_map = {}
            range_map = {}
            station_map = {}  # list of stations for each unit
            stationids = list(
                (df["station_id"].astype(str) + df["subloc"].astype(str)).unique()
            )
            color_df = get_color_dataframe(stationids, hv.Cycle())
            for i, r in df.iterrows():
                for repo_level in self.station_datastore.repo_level:
                    crvs = self.create_curves(r, repo_level)
                    unit = r["unit"]
                    if self.station_datastore.convert_units:
                        unit = uniform_unit_for(r["param"])
                    if unit not in layout_map:
                        layout_map[unit] = []
                        title_map[unit] = [
                            repo_level,
                            r["param"],
                            r["station_id"],
                            r["agency"],
                            r["subloc"],
                        ]
                        range_map[unit] = None
                        station_map[unit] = []
                    layout_map[unit].extend(crvs)
                    station_map[unit].append(r["station_id"] + r["subloc"])
                    if self.sensible_range_yaxis:
                        for crv in crvs:
                            range_map[unit] = self._calculate_range(
                                range_map[unit], crv.data
                            )
                    self._append_to_title_map(title_map, unit, r, repo_level)
            if len(layout_map) == 0:
                return hv.Div("<h3>Select rows from table and click on button</h3>")
            else:
                return (
                    hv.Layout(
                        [
                            hv.Overlay(layout_map[k])
                            .opts(
                                opts.Curve(color=get_colors(station_map[k], color_df))
                            )
                            .opts(
                                show_legend=self.show_legend,
                                legend_position=self.legend_position,
                                ylim=(
                                    tuple(range_map[k])
                                    if range_map[k] is not None
                                    else (None, None)
                                ),
                                title=self._create_title(title_map[k]),
                                min_height=400,
                            )
                            for k in layout_map
                        ]
                    )
                    .cols(1)
                    .opts(
                        axiswise=True,
                        sizing_mode="stretch_both",
                    )
                )
        except Exception as e:
            stackmsg = full_stack()
            print(stackmsg)
            pn.state.notifications.error(f"Error while fetching data for {e}")
            return hv.Div(f"<h3> Exception while fetching data </h3> <pre>{e}</pre>")

    def _slice_df(self, df, time_range):
        sdf = df.loc[slice(*time_range), :]
        if sdf.empty:
            return pd.DataFrame(
                columns=["value"],
                index=pd.date_range(*time_range, freq="D"),
                dtype=float,
            )
        else:
            return sdf

    def get_data_for_time_range(self, repo_level, filename, meta=False):
        """
        Gets data for time range for repo_level (screened/formatted) for the filename
        If meta is True it returns a tuple of the dataframe and meta
        """
        try:
            df = self.station_datastore.get_data(repo_level, filename)
        except Exception as e:
            print(full_stack())
            if pn.state.notifications:
                pn.state.notifications.error(
                    f"Error while fetching data for {repo_level}/{filename}: {e}"
                )
            df = pd.DataFrame(columns=["value"], dtype=float)
        df = self._slice_df(df, self.time_range)
        return df

    def do_unit_and_filtering(self, df):
        df, unit = self.station_datastore.get_uniform_units_data(df, param, unit)
        df = self.station_datastore.get_filtered_data(df)
        return df, unit

    def do_create_crv(
        self, df, repo_level, param, unit, station_id, subloc, agency, agency_id_dbase
    ):
        crvlabel = f"{repo_level}/{station_id}{subloc}/{param}"
        crv = hv.Curve(df[["value"]], label=crvlabel).redim(value=crvlabel)
        return crv.opts(
            xlabel="Time",
            ylabel=f"{param}({unit})",
            title=f"{repo_level}/{station_id}{subloc}::{agency}/{agency_id_dbase}",
            responsive=True,
            active_tools=["wheel_zoom"],
            tools=["hover"],
        )

    def create_curves(self, r, repo_level):
        filename = r["filename"]
        param = r["param"]
        unit = r["unit"]
        station_id = r["station_id"]
        subloc = r["subloc"] if len(r["subloc"]) == 0 else f'/{r["subloc"]}'
        agency = r["agency"]
        agency_id_dbase = r["agency_id_dbase"]
        df = self.get_data_for_time_range(repo_level, filename)
        # if df doesn't have value column # work around for issue https://github.com/CADWRDeltaModeling/dms_datastore/issues/15
        if "value" not in df.columns:
            dflist = [df.loc[:, [c]].rename(columns={c: "value"}) for c in df.columns]
            sublocs = list(df.columns)
        else:
            dflist = [df]
            sublocs = [subloc]
        dflist, unit = zip(
            *[
                self.station_datastore.get_uniform_units_data(df, param, unit)
                for df in dflist
            ]
        )
        dflist = [self.station_datastore.get_filtered_data(df) for df in dflist]
        unit = list(unit)
        return [
            self.do_create_crv(
                df, repo_level, param, unit, station_id, subloc, agency, agency_id_dbase
            )
            for df, subloc, unit in zip(dflist, sublocs, unit)
        ]

    def update_plots(self, event):
        self.display_area.loading = True
        self.plot_panel.object = self.create_plots(event)
        self.display_area.clear()
        self.display_area.append(self.plot_panel)
        self.display_area.loading = False

    def permalink_callback(self, event):
        if pn.state.location:
            pn.state.location.update_query(**self.get_permalink_params())

    def get_permalink_params(self):
        # get the current state
        urlparams = {}
        sdate, edate = (r.isoformat().rsplit(":", 1)[0] for r in self.time_range)
        urlparams["sdate"] = sdate
        urlparams["edate"] = edate
        urlparams["repo_level"] = self.station_datastore.repo_level
        df = self.display_table.value.iloc[self.display_table.selection]
        selections = (
            df["station_id"].astype(str)
            + "|"
            + df["subloc"].astype(str)
            + "|"
            + df["param"].astype(str)
        )
        selections = selections.str.cat(sep=",")
        urlparams["selections"] = selections
        return urlparams

    def set_ui_state_from_url(self, urlparams):
        if "sdate" in urlparams and "edate" in urlparams:
            self.time_range = (
                datetime.fromisoformat(urlparams["sdate"][0].decode()),
                datetime.fromisoformat(urlparams["edate"][0].decode()),
            )
        if "repo_level" in urlparams:
            self.station_datastore.repo_level = eval(
                urlparams["repo_level"][0].decode()
            )
        if "selections" in urlparams:
            selections = urlparams["selections"][0].decode().split(",")
            tuples = [tuple(s.split("|")) for s in selections]
            columns = ["station_id", "subloc", "param"]
            df = self.current_station_inventory.loc[:, columns]
            tuples_df = pd.DataFrame(tuples, columns=columns)
            # Get index of matching stations
            df["subloc"] = df["subloc"].str.strip()
            tuples_df["subloc"] = tuples_df["subloc"].str.strip()

            # Convert all string columns to the same case if necessary (e.g., lower case)
            df["station_id"] = df["station_id"].str.lower()
            df["param"] = df["param"].str.lower()
            tuples_df["station_id"] = tuples_df["station_id"].str.lower()
            tuples_df["param"] = tuples_df["param"].str.lower()
            #
            self.query = f"station_id in {tuple(tuples_df['station_id'].unique())}"
            # Use merge to find exact matches, include an indicator
            merged_df = pd.merge(
                df.reset_index(),
                tuples_df,
                on=["station_id", "subloc", "param"],
                how="left",
                indicator=True,
            )
            # Filter the merged DataFrame to only matched rows and get the original indices
            matched_indices = merged_df[merged_df["_merge"] == "both"]["index"].tolist()
            # select the stations after reseting station inventory to full index
            self.current_station_inventory = self.station_datastore.df_station_inventory
            self.station_select.event(index=matched_indices)
            # select the rows in the table -- can't select stations till they are displayed ?
            # self.display_table.selection = list(index)

    def download_data(self):
        self.download_button.loading = True
        try:
            df = self.display_table.value.iloc[self.display_table.selection]
            df = df.merge(self.station_datastore.df_dataset_inventory)
            dflist = []
            dfflist = []
            for i, r in df.iterrows():
                for repo_level in self.station_datastore.repo_level:
                    dfdata = self.get_data_for_time_range(repo_level, r["filename"])
                    # ignore user_flag for now
                    dfdata = dfdata[["value"]]
                    # check if user_flag exists first...
                    # dfflags = dfdata[['user_flag']]
                    param = r["param"]
                    unit = r["unit"]
                    subloc = r["subloc"]
                    station_id = r["station_id"]
                    agency = r["agency"]
                    agency_id_dbase = r["agency_id_dbase"]
                    dfdata.columns = [
                        f"{repo_level}/{station_id}/{subloc}/{agency}/{agency_id_dbase}/{param}/{unit}"
                    ]
                    # dfflags.columns = dfdata.columns
                    dflist.append(dfdata)
                    # dfflist.append(dfflags)
            dfdata = pd.concat(dflist, axis=1)
            # dfflag = pd.concat(dfflist, axis=1)
            sio = StringIO()
            dfdata.to_csv(sio)
            # ??? dfflag.to_csv(sio)
            sio.seek(0)
            return sio
        finally:
            self.download_button.loading = False

    def show_data_screener(self, event):
        self.display_area.loading = True
        try:
            df = self.display_table.value.iloc[self.display_table.selection]
            df = df.merge(self.station_datastore.df_dataset_inventory)
            view = pn.Tabs()
            for i, r in df.iterrows():
                for repo_level in self.station_datastore.repo_level:
                    filepath = self.station_datastore.get_data_filepath(
                        repo_level, r["filename"]
                    )
                    screener = data_screener.DataScreener(filepath)
                    screener.time_range = self.time_range
                    view.append(
                        (
                            f"{r['station_id']}_{r['subloc']}_{r['param']}",
                            screener.view(),
                        )
                    )
            self.display_area.clear()
            self.display_area.append(view)
        finally:
            self.display_area.loading = False

    def show_flag_editor(self, event):
        self.display_area.loading = True
        try:
            df = self.display_table.value.iloc[self.display_table.selection]
            df = df.merge(self.station_datastore.df_dataset_inventory)
            view = pn.Tabs()
            for i, r in df.iterrows():
                for repo_level in self.station_datastore.repo_level:
                    filepath = self.station_datastore.get_data_filepath(
                        repo_level, r["filename"]
                    )
                    editor = flag_editor.FlagEditor(filepath)
                    editor.time_range = self.time_range
                    view.append(
                        (f"{r['station_id']}_{r['subloc']}_{r['param']}", editor.view())
                    )
            self.display_area.clear()
            self.display_area.append(view)
        finally:
            self.display_area.loading = False

    def show_gap_visualizer(self, event):
        self.display_area.loading = True
        try:
            df = self.display_table.value.iloc[self.display_table.selection]
            df = df.merge(self.station_datastore.df_dataset_inventory)
            from dms_datastore_ui import gap_visualizer

            if df is None or df.empty:
                return hv.Div("<h3>Select rows from table and click on button</h3")

            # Create individual views with linked x ranges
            views = []
            for i, r in df.iterrows():
                for repo_level in self.station_datastore.repo_level:
                    dfdata = self.get_data_for_time_range(repo_level, r["filename"])
                    gv = gap_visualizer.GapVisualizer(dfdata, r)
                    crv, spike = gv.visualize_gap()
                    views.append(crv)
                    views.append(spike)

            layout = (
                hv.Layout(views)
                .cols(1)
                .opts(shared_axes=True, sizing_mode="stretch_width")
            )

            view = pn.Column(layout, sizing_mode="stretch_width")
            self.display_area.clear()
            self.display_area.append(view)
        except Exception as e:
            stackmsg = full_stack()
            print(stackmsg)
            pn.state.notifications.error(f"Error while fetching data for {e}")
            self.display_area.clear()
            self.display_area.append(
                hv.Div(f"<h3> Exception while fetching data </h3> <pre>{e}</pre>")
            )
        finally:
            self.display_area.loading = False

    def update_data_table(self, dfs):
        # if attribute display_table is not set, create it
        if not hasattr(self, "display_table"):
            column_width_map = {
                "station_id": "10%",
                "subloc": "5%",
                "lat": "10%",
                "lon": "10%",
                "name": "25%",
                "min_year": "5%",
                "max_year": "5%",
                "agency": "5%",
                "agency_id_dbase": "5%",
                "param": "10%",
                "unit": "10%",
            }
            from bokeh.models.widgets.tables import NumberFormatter

            self.display_table = pn.widgets.Tabulator(
                dfs,
                disabled=True,
                widths=column_width_map,
                show_index=False,
                sizing_mode="stretch_width",
                formatters={
                    "min_year": NumberFormatter(format="0"),
                    "max_year": NumberFormatter(format="0"),
                },
                header_filters=True,
            )
            self.display_area = pn.Column()
            self.plot_button = pn.widgets.Button(
                name="Plot", button_type="primary", icon="chart-line"
            )
            self.plot_button.on_click(self.update_plots)
            self.plot_panel = pn.panel(
                hv.Div("<h3>Select rows from table and click on button</h3>"),
                sizing_mode="stretch_both",
            )
            self.screener_button = pn.widgets.Button(
                name="Data Screener", button_type="primary", icon="table"
            )
            self.screener_button.on_click(self.show_data_screener)
            self.gap_biz_button = pn.widgets.Button(
                name="Gap Visualizer", button_type="primary", icon="table"
            )
            self.gap_biz_button.on_click(self.show_gap_visualizer)
            self.editor_button = pn.widgets.Button(
                name="Flag Editor", button_type="primary", icon="flag"
            )
            self.editor_button.on_click(self.show_flag_editor)
            # add a button to trigger the save function
            self.download_button = pn.widgets.FileDownload(
                label="Download",
                callback=self.download_data,
                filename="dms_data.csv",
                button_type="primary",
                icon="file-download",
                embed=False,
            )
            self.permalink_button = pn.widgets.Button(
                name="Permalink", button_type="primary", icon="link"
            )

            self.permalink_button.on_click(self.permalink_callback)

            gspec = pn.GridStack(
                sizing_mode="stretch_both", allow_resize=True, allow_drag=False
            )  # ,
            gspec[0, 0:10] = pn.Row(
                self.plot_button,
                self.screener_button,
                self.editor_button,
                self.gap_biz_button,
                self.download_button,
                self.permalink_button,
            )
            self.display_area.clear()
            self.display_area.append(self.plot_panel)
            gspec[1:5, 0:10] = fullscreen.FullScreen(pn.Row(self.display_table))
            gspec[6:15, 0:10] = fullscreen.FullScreen(self.display_area)
            self.plots_panel = pn.Row(
                gspec
            )  # fails with object of type 'GridSpec' has no len()
        else:
            self.display_table.value = dfs

        return self.plots_panel

    def get_map_of_stations(self, vartype, color_category, symbol_category, query):
        if len(vartype) == 1 and vartype[0] == "all":
            dfs = self.station_datastore.df_station_inventory
        else:
            dfs = self.station_datastore.df_station_inventory[
                self.station_datastore.df_station_inventory["param"].isin(vartype)
            ]
        # limit the current station inventory to those rows whose min_year and max_year are within the year_range
        # dfs = dfs.query(
        #    f"min_year >= {self.year_range[0]} and max_year <= {self.year_range[1]}"
        # )
        query = query.strip()
        if len(query) > 0:
            dfs = dfs.query(query)
        self.current_station_inventory = dfs
        self.map_station_inventory.data = self.current_station_inventory
        return self.tmap * self.map_station_inventory.opts(
            color=dim(color_category),
            marker=dim("param").categorize(self.get_param_to_marker_map()),
        )

    def get_disclaimer_text(self):
        from ._version import get_versions

        # Add disclaimer about data hosted here
        # insert app version with date time of last commit and commit id
        version_string = f"DMS Datastore UI: {get_versions()['version']}"
        disclaimer_text = f"""
        ## App version:
        ### {version_string}

        ## Data Disclaimer

        The data here is not the original data as provided by the agencies. The original data should be obtained from the agencies.

        The data presented here is an aggregation of data from various sources. The various sources are listed in the inventory file as agency and agency_id_dbase.

        The data here has been modified and corrected as needed by the Delta Modeling Section for use in the Delta Modeling Section's models and analysis.
        """
        return disclaimer_text

    def create_about_button(self, template):
        about_btn = pn.widgets.Button(
            name="About this Site", button_type="primary", icon="info-circle"
        )

        def about_callback(event):
            template.open_modal()

        about_btn.on_click(about_callback)
        return about_btn

    def create_view(self):
        control_widgets = pn.Row(
            pn.Column(
                pn.WidgetBox(
                    "Map Options",
                    self.station_datastore.param.parameter_type,
                    self.param.use_symbols_for_params,
                    self.param.map_color_category,
                    self.param.query,
                ),
                self.param.search_text,
            ),
            pn.Column(
                pn.Param(
                    self.param.time_range,
                    widgets={
                        "time_range": {
                            "widget_type": pn.widgets.DatetimeRangeInput,
                            "format": "%Y-%m-%d %H:%M",
                        }
                    },
                ),
                self.station_datastore.param.repo_level,
                pn.WidgetBox(
                    self.station_datastore.param.apply_filter,
                    self.station_datastore.param.filter_type,
                    self.station_datastore.param.fill_gap,
                ),
                pn.WidgetBox(
                    self.param.show_legend,
                    self.param.legend_position,
                ),
                self.station_datastore.param.convert_units,
                self.param.sensible_range_yaxis,
            ),
        )
        map_tooltip = pn.widgets.TooltipIcon(
            value='Map of stations. Click on a station to see data available in the table. See <a href="https://docs.bokeh.org/en/latest/docs/user_guide/interaction/tools.html">Bokeh Tools</a> for toolbar operation'
        )
        map_display = pn.bind(
            self.get_map_of_stations,
            vartype=self.station_datastore.param.parameter_type,
            color_category=self.param.map_color_category,
            symbol_category=self.param.use_symbols_for_params,
            query=self.param.query,
        )
        sidebar_view = pn.Column(
            control_widgets,
            pn.Column(pn.Row("Station Map", map_tooltip), map_display),
        )
        main_view = pn.Column(
            pn.bind(self.show_inventory, index=self.station_select.param.index),
        )
        self.set_ui_state_from_url(pn.state.session_args)
        template = pn.template.VanillaTemplate(
            title="DMS Datastore",
            sidebar=[sidebar_view],
            sidebar_width=650,
            header_color="blue",
            logo="https://sciencetracker.deltacouncil.ca.gov/themes/custom/basic/images/logos/DWR_Logo.png",
        )
        template.main.append(main_view)
        # Adding about button
        template.modal.append(self.get_disclaimer_text())
        control_widgets[0].append(self.create_about_button(template))
        return template


#!conda install -y -c conda-forge jupyter_bokeh
if __name__ == "__main__":
    # using argparse to get the directory
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="directory with station inventory")
    # add argument optional to run caching
    parser.add_argument("--cache", help="use caching", action="store_true")
    args = parser.parse_args()
    # if no args exit with help message
    if args.dir is None:
        parser.print_help()
        exit(0)
    else:
        dir = args.dir
    explorer = StationInventoryExplorer(dir)
    if args.cache:
        # run caching
        print("Clearing cache")
        explorer.station_datastore.clear_cache()
        print("Caching data")
        for repo_level in explorer.params.repo_level.objects:
            print("Caching ", repo_level)
            explorer.station_datastore.cache_repo_level(repo_level)
    else:  # display ui
        explorer.create_view().show(title="Station Inventory Explorer")
