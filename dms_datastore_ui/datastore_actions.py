"""Custom action classes for the DatastoreUIMgr.

These plug into :meth:`DatastoreUIMgr.get_data_actions` and appear as
buttons in the DataUI action bar.  Each class follows the same pattern as
the built-in dvue action classes: a single ``callback(event, dataui)``
method that receives the current :class:`~dvue.dataui.DataUI` instance and
opens a new closable tab in the display panel.

Imports of data_screener, flag_editor, and gap_visualizer MUST remain at
module level (not inside callbacks).  Each of those modules calls
``pn.extension()`` at import time; if they were imported lazily inside a
live Bokeh session callback, the ``pn.extension()`` call would fire during
the callback and reset Panel's internal document state (``pn.state.curdoc``),
orphaning subsequently-created Bokeh models from the session document and
causing ``UnknownReferenceError`` when interactive buttons are clicked.
"""

import panel as pn
import holoviews as hv
from dvue.utils import full_stack
import logging

# Eager imports so that module-level pn.extension() calls in each module
# run at server startup, before any session is alive.
from dms_datastore_ui import data_screener  # noqa: E402 – pn.extension("codeeditor") at module level
from dms_datastore_ui import flag_editor    # noqa: E402 – pn.extension() at module level
from dms_datastore_ui import gap_visualizer # noqa: E402 – pn.extension() at module level

logger = logging.getLogger(__name__)


def _open_display_tab(dataui, title, content):
    """Helper: add *content* as a new closable tab in the DataUI display panel."""
    if len(dataui._display_panel.objects) > 0 and isinstance(
        dataui._display_panel.objects[0], pn.Tabs
    ):
        tabs = dataui._display_panel.objects[0]
        dataui._tab_count += 1
        tabs.append((title, content))
        tabs.active = len(tabs) - 1
    else:
        dataui._tab_count = 0
        dataui._display_panel.objects = [
            pn.Tabs((title, content), closable=True, dynamic=True)
        ]


def _selected_rows(dataui):
    """Return the selected catalog rows merged with the dataset inventory, or None."""
    if not dataui.display_table.selection:
        if pn.state.notifications is not None:
            pn.state.notifications.warning(
                "Please select at least one row from the table.", duration=3000
            )
        return None
    df = dataui.display_table.value.iloc[dataui.display_table.selection]
    # merge with dataset inventory to get filename column if not already present
    uimgr = dataui._dataui_manager
    if "filename" not in df.columns and hasattr(uimgr, "datastore"):
        df = df.merge(uimgr.datastore.df_dataset_inventory)
    return df


class DataScreenerAction:
    """Open a :class:`~dms_datastore_ui.data_screener.DataScreener` tab for
    each selected row, using the current time range from the manager."""

    def callback(self, event, dataui):
        dataui._display_panel.loading = True
        try:
            df = _selected_rows(dataui)
            if df is None:
                return
            uimgr = dataui._dataui_manager
            time_range = uimgr.time_range
            repo_levels = uimgr.datastore.repo_level

            view = pn.Tabs()
            for _, r in df.iterrows():
                for repo_level in repo_levels:
                    filepath = uimgr.datastore.get_data_filepath(repo_level, r["filename"])
                    screener = data_screener.DataScreener(filepath)
                    screener.time_range = time_range
                    tab_label = f"{r['station_id']}_{r['subloc']}_{r['param']}"
                    view.append((tab_label, screener.view()))

            _open_display_tab(dataui, "Data Screener", view)
        except Exception as e:
            stack_str = full_stack()
            logger.error(stack_str)
            if pn.state.notifications is not None:
                pn.state.notifications.error("Data Screener error: " + str(e), duration=0)
        finally:
            dataui._display_panel.loading = False


class FlagEditorAction:
    """Open a :class:`~dms_datastore_ui.flag_editor.FlagEditor` tab for
    each selected row, using the current time range from the manager."""

    def callback(self, event, dataui):
        dataui._display_panel.loading = True
        try:
            df = _selected_rows(dataui)
            if df is None:
                return
            uimgr = dataui._dataui_manager
            time_range = uimgr.time_range
            repo_levels = uimgr.datastore.repo_level

            view = pn.Tabs()
            for _, r in df.iterrows():
                for repo_level in repo_levels:
                    filepath = uimgr.datastore.get_data_filepath(repo_level, r["filename"])
                    editor = flag_editor.FlagEditor(filepath)
                    editor.time_range = time_range
                    tab_label = f"{r['station_id']}_{r['subloc']}_{r['param']}"
                    view.append((tab_label, editor.view()))

            _open_display_tab(dataui, "Flag Editor", view)
        except Exception as e:
            stack_str = full_stack()
            logger.error(stack_str)
            if pn.state.notifications is not None:
                pn.state.notifications.error("Flag Editor error: " + str(e), duration=0)
        finally:
            dataui._display_panel.loading = False


class GapVisualizerAction:
    """Open a gap-visualization panel for each selected row.

    Uses the time-range-sliced data already available via the catalog so the
    view respects whatever time window the user has configured.
    """

    def callback(self, event, dataui):
        dataui._display_panel.loading = True
        try:
            df = _selected_rows(dataui)
            if df is None:
                return
            uimgr = dataui._dataui_manager
            time_range = uimgr.time_range
            repo_levels = uimgr.datastore.repo_level

            views = []
            for _, r in df.iterrows():
                for repo_level in repo_levels:
                    ref = uimgr.data_catalog.get(r["filename"])
                    # Sync repo_level on the reference
                    if ref.get_attribute("repo_level") != repo_level:
                        ref.set_attribute("repo_level", repo_level)
                    try:
                        dfdata = ref.getData()
                        dfdata = dfdata[slice(time_range[0], time_range[1])]
                    except Exception as e:
                        logger.error(f"Could not load data for {r['filename']}: {e}")
                        continue
                    gv = gap_visualizer.GapVisualizer(dfdata, r)
                    crv, spike = gv.visualize_gap()
                    views.append(crv)
                    views.append(spike)

            if not views:
                _open_display_tab(
                    dataui,
                    "Gap Visualizer",
                    pn.pane.HTML("<h3>No data available for selected rows.</h3>"),
                )
                return

            layout = hv.Layout(views).cols(1).opts(shared_axes=True, sizing_mode="stretch_width")
            _open_display_tab(dataui, "Gap Visualizer", pn.Column(layout, sizing_mode="stretch_width"))
        except Exception as e:
            stack_str = full_stack()
            logger.error(stack_str)
            if pn.state.notifications is not None:
                pn.state.notifications.error("Gap Visualizer error: " + str(e), duration=0)
        finally:
            dataui._display_panel.loading = False
