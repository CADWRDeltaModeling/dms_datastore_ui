"""dvue plugin entry points for dms_datastore_ui readers.

Exports
-------
register_readers
    No-arg function called by dvue at startup via the ``dvue.plugins``
    entry point group.  Registers :class:`DatastoreFilepathReader` for the
    ``"datastore_csv"`` ref-type and the ``.csv`` extension.

DVueUIManager
    Optional module-level symbol read by ``dvue ui``; overrides the default
    :class:`~dvue.registry_ui.RegistryUIManager` with a subclass that shows
    datastore-specific table columns (station_id, param, unit, agency …)
    and map tooltips.
"""

from dvue.registry import ReaderRegistry
from dvue.registry_ui import RegistryUIManager

from dms_datastore_ui.datastore_uimgr import DatastoreFilepathReader


def register_readers() -> None:
    """Register dms_datastore_ui readers with dvue ReaderRegistry."""
    ReaderRegistry.register(
        "datastore_csv",
        DatastoreFilepathReader,
        extensions=[".csv"],
    )


class DatastoreRegistryUIManager(RegistryUIManager):
    """RegistryUIManager with DatastoreUIMgr-compatible table/map presentation.

    Used automatically by ``dvue ui`` when the ``dms_datastore_ui`` plugin is
    installed.  Overrides column definitions, filters, tooltips and map-color
    settings so inventory-derived refs display the same fields as the full
    :class:`~dms_datastore_ui.datastore_uimgr.DatastoreUIMgr`.
    """

    def __init__(self, files=(), **kwargs):
        super().__init__(files=files, **kwargs)
        # Override RegistryUIManager defaults.  Must happen after super().__init__()
        # because that sets these as instance attributes (class-level attrs would be
        # silenced by the instance assignment in RegistryUIManager.__init__).
        # station_id_column="station_id" ensures _update_map_geo_selection takes
        # the isin() branch — "station_id" is a real display-table column, whereas
        # "station" (the RegistryUIManager default) is absent from our column map
        # which would cause a KeyError via the fallback dfs.loc[current_view.index].
        self.station_id_column = "station_id"
        self.color_cycle_column = "station_id"
        self.dashed_line_cycle_column = "subloc"
        self.marker_cycle_column = "param"

    def _get_table_column_width_map(self):
        return {
            "station_id": "10%",
            "subloc": "5%",
            "station_name": "25%",
            "min_year": "5%",
            "max_year": "5%",
            "agency": "5%",
            "agency_id_dbase": "5%",
            "param": "10%",
            "unit": "10%",
        }

    def get_table_columns(self):
        # Include "name" as a data-only column so get_data_reference() can
        # look up refs by their catalog key even though it is not displayed.
        return list(self.get_table_column_width_map().keys()) + ["name"]

    def get_table_filters(self):
        return {
            "station_id": {"type": "input", "func": "like", "placeholder": "filter"},
            "subloc": {"type": "input", "func": "like", "placeholder": "filter"},
            "station_name": {"type": "input", "func": "like", "placeholder": "filter"},
            "param": {"type": "input", "func": "like", "placeholder": "filter"},
            "agency_id_dbase": {"type": "input", "func": "like", "placeholder": "filter"},
            "agency": {"type": "input", "func": "like", "placeholder": "filter"},
            "unit": {"type": "input", "func": "like", "placeholder": "filter"},
            "min_year": {"type": "number"},
            "max_year": {"type": "number"},
        }

    def build_station_name(self, r):
        subloc = r.get("subloc") or ""
        station_id = str(r.get("station_id") or r.get("station") or "")
        return f"{station_id}@{subloc}" if subloc else station_id

    def get_tooltips(self):
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
        return ["param", "agency", "min_year", "max_year"]

    def get_name_to_color(self):
        return {
            "param": "Category10",
            "agency": "Category20",
            "min_year": "Viridis",
            "max_year": "Viridis",
        }

    def get_map_marker_columns(self):
        return ["param"]

    def get_map_color_category(self):
        return "param"


# Picked up by ``dvue ui`` CLI — overrides the default RegistryUIManager.
DVueUIManager = DatastoreRegistryUIManager
