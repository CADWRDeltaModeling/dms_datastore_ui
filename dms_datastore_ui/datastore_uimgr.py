import os
import param
import panel as pn
import pandas as pd
from shapely.geometry import Point
from dvue.actions import (
    PlotAction,
    DownloadDataAction,
    PermalinkAction,
    DownloadDataCatalogAction,
)
from dvue.dataui import DataUIManager
from dvue.tsdataui import TimeSeriesDataUIManager, TimeSeriesPlotAction
from dvue.catalog import (
    DataReferenceReader,
    DataReference,
    CatalogBuilder,
    DataCatalog,
)
from dms_datastore.read_ts import read_ts
from dms_datastore_ui.map_inventory_explorer import StationDatastore
from dms_datastore_ui.map_inventory_explorer import to_uniform_units
from dms_datastore_ui.datastore_actions import (
    DataScreenerAction,
    FlagEditorAction,
    GapVisualizerAction,
)
import holoviews as hv
import geopandas as gpd


class DatastoreFilepathReader(DataReferenceReader):
    """Reads time series data from an absolute ``filepath`` attribute.

    This reader is standalone and does not depend on :class:`StationDatastore`
    state. A single instance can be shared across many references.
    """

    def load(self, **attributes) -> pd.DataFrame:
        filepath = attributes.get("filepath")
        if not filepath:
            raise KeyError("DatastoreFilepathReader requires a 'filepath' attribute")
        return read_ts(filepath)

    def __repr__(self) -> str:
        return "DatastoreFilepathReader()"


class DatastoreDataReference(DataReference):
    """Datastore-specific :class:`~dvue.catalog.DataReference`.

    References are standalone by carrying an absolute ``filepath`` and the
    metadata required by filtering, map display, and mixed-catalog workflows.
    """

    def __init__(self, reader=None, name: str = "", cache: bool = False, **attributes):
        if not attributes.get("filepath"):
            raise ValueError("DatastoreDataReference requires a non-empty 'filepath'")
        if reader is None:
            reader = DatastoreFilepathReader()
        super().__init__(reader=reader, name=name, cache=cache, **attributes)

    @classmethod
    def from_inventory_row(cls, row, repo_dir, repo_level, reader=None):
        filename = row["filename"]
        subloc = row["subloc"] if pd.notna(row["subloc"]) and row["subloc"] else ""
        filepath = os.path.join(repo_dir, repo_level, filename)
        return cls(
            reader=reader,
            name=filename,
            cache=False,
            filepath=filepath,
            repo_level=repo_level,
            filename=filename,
            station_id=row["station_id"],
            subloc=subloc,
            station_name=row["name"],
            param=row["param"],
            unit=row["unit"],
            min_year=row["min_year"],
            max_year=row["max_year"],
            agency=row["agency"],
            agency_id_dbase=row["agency_id_dbase"],
            x=row["x"],
            y=row["y"],
            geometry=Point(row["x"], row["y"]),
        )

    @property
    def filepath(self):
        return self.get_attribute("filepath")

    @property
    def station_id(self):
        return self.get_attribute("station_id")

    @property
    def subloc(self):
        return self.get_attribute("subloc")

    @property
    def parameter(self):
        """Return the parameter/variable name (e.g. 'flow', 'temp')."""
        return self.get_attribute("param")

    @property
    def unit(self):
        return self.get_attribute("unit")

    @property
    def geometry(self):
        return self.get_attribute("geometry")


class DatastoreCatalogBuilder(CatalogBuilder):
    """Builds :class:`DatastoreDataReference` objects from a :class:`StationDatastore`.

    Each row in the merged station/dataset inventory becomes one
    ``DatastoreDataReference``. A shared :class:`DatastoreFilepathReader`
    lazily loads each row's file using its absolute ``filepath`` attribute.

    In-memory caching on each reference is disabled (``cache=False``)
    because the :class:`StationDatastore` already maintains an on-disk
    LRU cache via *diskcache*.
    """

    def can_handle(self, source) -> bool:
        return isinstance(source, StationDatastore)

    def build(self, source: StationDatastore):
        reader = DatastoreFilepathReader()
        # Merge station inventory with dataset inventory on common columns.
        # df_dataset_inventory has additional columns like filename.
        merge_keys = ["station_id", "subloc", "name", "unit", "param", "min_year", "max_year", "agency", "agency_id_dbase", "x", "y"]
        inventory = source.df_dataset_inventory  # dataset inventory already includes all needed columns
        repo_level = source.repo_level[0]
        refs = []
        for _, row in inventory.iterrows():
            ref = DatastoreDataReference.from_inventory_row(
                row=row,
                repo_dir=source.dir,
                repo_level=repo_level,
                reader=reader,
            )
            refs.append(ref)
        return refs

    def __repr__(self) -> str:
        return "DatastoreCatalogBuilder()"


class DatastorePlotAction(TimeSeriesPlotAction):
    """TimeSeriesPlotAction with datastore-specific curve labels and titles."""

    @staticmethod
    def _append_value(new_value, value):
        if new_value not in value:
            value += f'{", " if value else ""}{new_value}'
        return value

    def create_curve(self, data, row, unit, file_index=""):
        subloc = (row.get("subloc") or "") if hasattr(row, "get") else ""
        station_id = row["station_id"]
        param = row["param"]
        station_label = f"{station_id}@{subloc}" if subloc else station_id
        crvlabel = f"{station_label}/{param} ({unit})"
        ylabel = f"{param} ({unit})"
        crv = hv.Curve(data.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        return crv.opts(
            xlabel="Time",
            ylabel=ylabel,
            responsive=True,
            active_tools=["wheel_zoom"],
            tools=["hover"],
        )

    def append_to_title_map(self, title_map, group_key, row):
        # value = [params, station_ids, unit]
        value = title_map.get(group_key, ["", "", str(group_key)])
        value[0] = self._append_value(row["param"], value[0])
        value[1] = self._append_value(row["station_id"], value[1])
        title_map[group_key] = value

    def create_title(self, title_info) -> str:
        params, station_ids, unit = title_info
        return f"{params} ({unit}) — {station_ids}"


class DatastoreUIMgr(TimeSeriesDataUIManager):
    show_math_ref_editor = param.Boolean(default=False)
    repo_level = param.ListSelector(
        objects=["screened"],
        default=["screened"],
        doc="Repository level (sub-directory) under which data is found. Ctrl+click to select multiple.",
    )

    unit_conversion = param.Boolean(
        default=False,
        doc="Convert units to standard units (e.g. feet to meters, cfs to cms)",
    )

    parameter_type = param.ListSelector(
        objects=["all"],
        default=["all"],
        doc="Filter map and table to these parameter types. Ctrl+click to select multiple, or choose 'all'.",
    )

    year_range = param.Range(
        default=(2000, 2010),
        step=1,
        bounds=(2000, 2010),
        doc="Filter map to stations whose data overlaps this year range.",
    )

    def __init__(self, dir, repo_level="screened", **kwargs):
        self.dir = dir
        self.datastore = StationDatastore(dir)
        # Build catalog before super().__init__() because the parent calls
        # get_data_catalog() during initialisation.
        self._catalog = (
            DataCatalog(crs="EPSG:26910")
            .add_builder(DatastoreCatalogBuilder())
            .add_source(self.datastore)
        )
        kwargs["filename_column"] = "filename"
        # Sync repo_level choices from the validated datastore objects.
        valid_levels = self.datastore.param.repo_level.objects
        self.param.repo_level.objects = valid_levels
        self.repo_level = valid_levels[:1]  # default to first valid level
        # Sync parameter_type choices from the datastore inventory.
        unique_params = list(self.datastore.unique_params)
        self.param.parameter_type.objects = ["all"] + unique_params
        # Sync year_range bounds from inventory.
        min_yr = int(self.datastore.min_year)
        max_yr = int(self.datastore.max_year)
        self.param.year_range.bounds = (min_yr, max_yr)
        self.year_range = (min_yr, max_yr)
        # Call the parent class's __init__ method with kwargs
        super().__init__(**kwargs)
        self.color_cycle_column = "station_id"
        self.dashed_line_cycle_column = "subloc"
        self.marker_cycle_column = "param"

    @param.depends("repo_level", watch=True)
    def _sync_repo_level(self):
        """Keep the StationDatastore in sync when repo_level changes."""
        self.datastore.repo_level = self.repo_level

    @param.depends("parameter_type", "year_range", watch=True)
    def _sync_map_query(self):
        """Push a pandas query string to DataUI when map filters change."""
        if not hasattr(self, "_dataui"):
            return
        parts = []
        if self.parameter_type and "all" not in self.parameter_type:
            param_list = ", ".join(f'"{p}"' for p in self.parameter_type)
            parts.append(f"param in [{param_list}]")
        min_yr, max_yr = self.year_range
        parts.append(f"min_year <= {max_yr} and max_year >= {min_yr}")
        self._dataui.query = " and ".join(parts)

    # ------------------------------------------------------------------
    # DataCatalog integration
    # ------------------------------------------------------------------

    @property
    def data_catalog(self):
        """Expose the underlying :class:`~dvue.catalog.DataCatalog`.

        :meth:`~dvue.dataui.DataProvider.get_data_catalog` automatically
        delegates to this, returning a :class:`~geopandas.GeoDataFrame`
        (because references carry a ``geometry`` attribute).
        """
        return self._catalog

    def get_widgets(self):
        widget_tabs = super().get_widgets()
        # "Data" tab: repo_level selector + unit conversion toggle
        widget_tabs.append(
            (
                "Data",
                pn.WidgetBox(
                    self.param.repo_level,
                    self.param.unit_conversion,
                ),
            )
        )
        return widget_tabs

    def get_map_option_widgets(self):
        """Extra widgets injected into the Map Options sidebar tab."""
        return pn.WidgetBox(
            "Filter Map",
            self.param.parameter_type,
            self.param.year_range,
        )

    def get_data_actions(self):
        actions = super().get_data_actions()
        actions.extend([
            dict(
                name="Data Screener",
                button_type="primary",
                icon="table",
                action_type="display",
                callback=DataScreenerAction().callback,
            ),
            dict(
                name="Flag Editor",
                button_type="primary",
                icon="flag",
                action_type="display",
                callback=FlagEditorAction().callback,
            ),
            dict(
                name="Gap Visualizer",
                button_type="primary",
                icon="chart-bar",
                action_type="display",
                callback=GapVisualizerAction().callback,
            ),
        ])
        return actions

    def _make_plot_action(self):
        return DatastorePlotAction()

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

    def get_data_reference(self, row):
        # The display table includes 'filename' but may not include 'name'.
        # For datastore refs, ref.name == filename.  For mixed catalogs the
        # 'name' column (present when the full catalog DF is used) is preferred.
        key = row.get("name") if "name" in row.index else row["filename"]
        return self._catalog.get(key)

    def get_table_column_width_map(self):
        return {
            "station_id": "10%",
            "subloc": "5%",
            # "lat": "10%",
            # "lon": "10%",
            "station_name": "25%",
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
            "station_name": {"type": "input", "func": "like"},
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
        # Look up the DataReference for this row using the catalog's universal
        # key (ref.name). For datastore refs name==filename; for any other ref
        # type in a mixed catalog name is always present after reset_index().
        ref = self.data_catalog.get(r["name"])
        current_repo_level = self.repo_level[0] if self.repo_level else "screened"
        if ref.get_attribute("repo_level") != current_repo_level:
            ref.set_attribute("repo_level", current_repo_level)

        unit = r["unit"]
        result_data = pd.DataFrame()
        try:
            ts_data = ref.getData()
            if self.unit_conversion:
                ts_data, unit = to_uniform_units(ts_data, r["param"], unit)
            result_data = ts_data[slice(time_range[0], time_range[1])]
        except Exception as e:
            print(
                f"Error retrieving data for {r['station_id']}/{r['param']} using {r['filename']}: {e}"
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
            ("Name", "@station_name"),
            ("Years", "@min_year to @max_year"),
            ("Agency", "@agency - @agency_id_dbase"),
            ("Parameter", "@param"),
            ("Unit", "@unit"),
        ]

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


# dir = "y:/repo/continuous"
# uimgr = DatastoreUIMgr(dir)
# from dvue import dataui

# ui = dataui.DataUI(uimgr, station_id_column="station_id")
# ui.create_view(title="DMS Datastore Data UI").servable()
