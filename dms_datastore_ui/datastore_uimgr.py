import param
import panel as pn
import pandas as pd
from pydelmod.dvue.actions import (
    PlotAction,
    DownloadDataAction,
    PermalinkAction,
    DownloadDataCatalogAction,
)
from pydelmod.dvue.dataui import DataUIManager
from pydelmod.dvue.tsdataui import TimeSeriesDataUIManager
from dms_datastore_ui.map_inventory_explorer import StationDatastore
import holoviews as hv
import geopandas as gpd


class DatastoreUIMgr(TimeSeriesDataUIManager):
    repo_level = param.ListSelector(
        objects=["screened"],
        default=["screened"],
        doc="repository level (sub directory) under which data is found. You can select multiple repo levels (ctrl+click)",
    )

    def __init__(self, dir, repo_level="screened", **kwargs):
        self.dir = dir
        self.datastore = StationDatastore(dir)
        kwargs["filename_column"] = "filename"
        # Call the parent class's __init__ method with kwargs
        super().__init__(**kwargs)
        self.color_cycle_column = "station_id"
        self.dashed_line_cycle_column = "subloc"
        self.marker_cycle_column = "param"

    # data related methods
    def get_data_catalog(self):
        """return a dataframe or geodataframe with the data catalog"""
        inventory = self.datastore.df_station_inventory
        inventory = inventory.merge(self.datastore.df_dataset_inventory)
        # use lon,lat to convert inventory to geodataframe
        inventory = gpd.GeoDataFrame(
            inventory,
            geometry=gpd.points_from_xy(inventory.lon, inventory.lat),
            crs="EPSG:4326",  # WGS84 coordinate reference system
        )
        return inventory

    def get_time_range(self, dfcat):
        """Convert year integers to datetime objects for CalendarDateRange parameter"""
        import datetime

        return (
            datetime.datetime.now() - datetime.timedelta(days=30),
            datetime.datetime.now(),
        )

    def build_station_name(self, r):
        # Build a unique station name from row data
        subloc = r["subloc"] if pd.notna(r["subloc"]) and r["subloc"] else ""
        return f"{r['station_id']}{subloc}"

    # display related support for tables
    def get_table_columns(self):
        return list(self.get_table_column_width_map().keys()) + ["filename"]

    def get_table_column_width_map(self):
        return {
            "station_id": "10%",
            "subloc": "5%",
            # "lat": "10%",
            # "lon": "10%",
            "name": "25%",
            "min_year": "5%",
            "max_year": "5%",
            "agency": "5%",
            "agency_id_dbase": "5%",
            "param": "10%",
            "unit": "10%",
        }

    def get_table_filters(self):
        filters = {
            "station_id": {"type": "input", "func": "like"},
            "subloc": {"type": "input", "func": "like"},
            "name": {"type": "input", "func": "like"},
            "param": {
                "type": "list",
                "valuesLookup": True,
                "sort": "asc",
                "multiselect": True,
            },
            "agency": {
                "type": "list",
                "func": "in",
                "valuesLookup": True,
                "sort": "asc",
                "multiselect": True,
            },
            "min_year": {"type": "number"},
            "max_year": {"type": "number"},
        }
        return filters

    def is_irregular(self, r):
        return False  # only regular time series data in example

    def get_data_for_time_range(self, r, time_range):
        repo_level = self.datastore.repo_level[0]  # Use first repo level by default
        filename = r["filename"]

        try:
            ts_data = self.datastore.get_data(repo_level, filename)
            param = r["param"]
            unit = r["unit"]
            station_id = r["station_id"]

            result_data = ts_data[slice(time_range[0], time_range[1])]
        except Exception as e:
            print(
                f"Error retrieving data for {station_id}/{param} using {filename}: {e}"
            )

        return (
            result_data,
            unit,
            "inst-val",
        )

    def get_station_ids(self, df):
        return list((df.apply(self.build_station_name, axis=1).astype(str).unique()))

    # methods below if geolocation data is available
    def get_tooltips(self):
        # Define tooltips for hover functionality
        return [
            ("Station ID", "@station_id"),
            ("SubLoc", "@subloc"),
            ("Name", "@name"),
            ("Years", "@min_year to @max_year"),
            ("Agency", "@agency - @agency_id_dbase"),
            ("Parameter", "@param"),
            ("Unit", "@unit"),
        ]

    def get_map_color_columns(self):
        """return the columns that can be used to color the map"""
        return ["param"]

    def get_map_marker_columns(self):
        """return the columns that can be used to color the map"""
        return ["param", "agency"]

    def get_map_color_columns(self):
        """return the columns that can be used to color the map"""
        return ["param", "agency", "min_year", "max_year"]

    def get_name_to_color(self):
        """return a dictionary mapping column names to color names"""
        return {
            "param": "Category10",
            "agency": "Category20",
            "min_year": "Viridis",
            "max_year": "Viridis",
        }

    def get_map_marker_columns(self):
        """return the columns that can be used to color the map"""
        return ["param"]

    def get_name_to_marker(self):
        """return a dictionary mapping column names to marker names"""
        from .map_inventory_explorer import param_to_marker_map

        return {"param": param_to_marker_map}

    def create_curve(self, df, r, unit, file_index=None):
        file_index_label = f"{self.repo_level}:" if file_index is not None else ""
        crvlabel = f'{file_index_label}{r["station_id"]}/{r["subloc"]}/{r["param"]}'
        ylabel = f'{r["param"]} ({unit})'
        title = f'{r["station_id"]}{r["subloc"]}::{r["agency"]}/{r["agency_id_dbase"]}'
        crv = hv.Curve(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        return crv.opts(
            xlabel="Time",
            ylabel=ylabel,
            title=title,
            responsive=True,
            active_tools=["wheel_zoom"],
            tools=["hover"],
        )

    def _append_value(self, new_value, value):
        if new_value not in value:
            value += f'{", " if value else ""}{new_value}'
        return value

    def append_to_title_map(self, title_map, unit, r):
        if unit in title_map:
            value = title_map[unit]
        else:
            value = ["", ""]
        value[0] = self._append_value(r["param"], value[0])
        value[1] = self._append_value(r["station_id"], value[1])
        title_map[unit] = value

    def create_title(self, v):
        title = f"{v[1]}({v[0]})"
        return title


dir = "y:/repo/continuous"
uimgr = DatastoreUIMgr(dir)
from pydelmod.dvue import dataui

ui = dataui.DataUI(uimgr, station_id_column="station_id")
ui.create_view(title="DMS Datastore Data UI").servable()
