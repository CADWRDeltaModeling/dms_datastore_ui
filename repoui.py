import logging
import sys

# Logging setup runs once at server start (not per-session as with panel serve).
_dms_logger = logging.getLogger("dms_datastore_ui")
if not getattr(_dms_logger, "_dms_handler_installed", False):
    _dms_logger.setLevel(logging.DEBUG)
    _dms_handler = logging.StreamHandler(sys.stderr)
    _dms_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    _dms_logger.addHandler(_dms_handler)
    _dms_logger.propagate = False  # avoid double-printing via Bokeh's root handler
    _dms_logger._dms_handler_installed = True
_dms_logger.warning("dms_datastore_ui logging active (level=%s)", logging.getLevelName(_dms_logger.level))
print(f"[repoui] dms_datastore_ui logging active, level={logging.getLevelName(_dms_logger.level)}", flush=True)

# ── [1] Session persistence ─────────────────────────────────────────────────
#
# Run as:  python repoui.py [REPO_DIR] [--port PORT] [--address ADDRESS]
# Do NOT use `panel serve` — install_session_handler() patches Bokeh's
# per_app_patterns before BokehServer starts; pn.serve() ensures module-level
# code runs exactly once.
#
# Two-layer persistence (both managed by SessionManager):
#   Layer 1 — in-memory registry (same server process)
#   Layer 2 — diskcache (server-restart fallback, persist=True)

import panel as pn
from pathlib import Path
import pandas as pd
from dvue.session_persistence import (
    install_session_handler,
    SessionManager,
    snapshot as _snapshot,
    restore as _restore,
)

install_session_handler()

_session_mgr = SessionManager(
    cookie_name="dvue_user_id",
    cache_dir=Path(__file__).parent / ".session_cache",
    persist=True,
)

# ── [2] Panel extension + heavy imports (run once at server start) ───────────

pn.extension(
    "gridstack",
    "tabulator",
    "codeeditor",
    notifications=True,
    reconnect=True,
    design="native",
    disconnect_notification="Connection lost, try reloading the page!",
    ready_notification="Application fully loaded.",
)
import datetime as dt
from dms_datastore_ui import fullscreen
import param
import holoviews as hv
import geoviews as gv

hv.extension("bokeh")
gv.extension("bokeh")

# Pre-import the heavy UI modules NOW so that their module-level extension
# calls happen once here, not inside a live Bokeh session.
import dms_datastore_ui.map_inventory_explorer  # noqa: F401
from dms_datastore_ui.datastore_uimgr import DatastoreUIMgr  # noqa: F401
from dvue.dataui import DataUI  # noqa: F401
import cartopy.crs as ccrs

# ── [3] Repo directory (override via CLI arg or environment) ─────────────────
_REPO_DIR = "continuous"


# ── [5] App factories (called once per Bokeh session / browser tab) ──────────
#
# make_newui_app  → DataUI (new UI); reset button lives in the DataUI action row
# make_oldui_app  → classic StationInventoryExplorer; reset button in header
#
# Registry key scheme: "{user_id}:newui" / "{user_id}:oldui"


def make_newui_app():
    user_id  = _session_mgr.current_user_id
    reg_key  = _session_mgr.make_reg_key(user_id, "newui")
    entry    = _session_mgr.get_entry(reg_key)
    reuse_dataui = bool(entry and entry.get("mode") == "dataui")

    header_link = pn.pane.HTML(
        '<a href="/oldui" style="color:white; font-size:0.9em; '
        'text-decoration:none; margin-left:1em; white-space:nowrap;">'
        "&#8594; Classic Explorer</a>",
        sizing_mode="fixed",
    )

    if reuse_dataui:
        # ── Registry hit: DataUI already built ──────────────────────────────
        template   = entry["template"]
        ui         = entry["ui"]
        main_panel = entry.get("main_panel")

        if main_panel is not None:
            main_panel.loading = True

        def _reattach():
            try:
                if ui:
                    ui.setup_location_sync()
                    ui.setup_url_sync()
            finally:
                if main_panel is not None:
                    main_panel.loading = False

        pn.state.onload(_reattach)
        template.servable(title="DMS Datastore")
        return

    # ── Build a fresh session ─────────────────────────────────────────────────
    main_panel = pn.Column(
        pn.indicators.LoadingSpinner(
            value=True, color="primary", size=50, name="Loading..."
        ),
        sizing_mode="stretch_both",
    )
    sidebar_panel = pn.Column(
        pn.indicators.LoadingSpinner(
            value=True, color="primary", size=50, name="Loading..."
        )
    )
    template = pn.template.VanillaTemplate(
        title="DMS Datastore",
        sidebar=[sidebar_panel],
        main=[main_panel],
        sidebar_width=650,
        header_color="blue",
        logo="dms_datastore_ui/california-department-of-water-resources-logo.png",
        header=[header_link],
    )

    def load_dataui():
        saved = _session_mgr.load_state(user_id)
        from dms_datastore_ui.d1641_standards import get_default_wyt_file
        uimgr = DatastoreUIMgr(_REPO_DIR, wyt_file=get_default_wyt_file())
        uimgr.show_reset_session_button = True  # reset button wired in DataUI action row
        if saved:
            _restore(uimgr, saved)

        ui = DataUI(uimgr, crs=ccrs.epsg(26910))
        ui_template = ui.create_view()

        sidebar_items = list(ui_template.sidebar)
        main_items    = list(ui_template.main)
        ui_template.sidebar.clear()
        ui_template.main.clear()
        ui_template.modal.clear()

        sidebar_panel.objects = sidebar_items
        main_panel.objects    = main_items

        # Wire About (and optional Disclaimer) buttons to the outer VanillaTemplate
        # so that open_modal() targets the correct served template.
        # add_header_buttons also populates template.modal with the about text.
        template.modal.clear()
        ui.add_header_buttons(template)

        _session_mgr.set_entry(reg_key, {
            "template": template, "mode": "dataui",
            "mgr": uimgr, "ui": ui, "main_panel": main_panel,
        })

        def _save(event=None):
            _session_mgr.save_state(user_id, _snapshot(uimgr, ui))

        uimgr.param.watch(_save, "time_range")
        if hasattr(ui, "display_table"):
            ui.display_table.param.watch(_save, "selection")

    pn.state.onload(load_dataui)
    template.servable(title="DMS Datastore")


def make_oldui_app():
    user_id = _session_mgr.current_user_id
    reg_key = _session_mgr.make_reg_key(user_id, "oldui")

    header_link = pn.pane.HTML(
        '<a href="/" style="color:white; font-size:0.9em; '
        'text-decoration:none; margin-left:1em; white-space:nowrap;">'
        "&#8592; New UI</a>",
        sizing_mode="fixed",
    )
    reset_btn = _session_mgr.make_reset_button(reg_key, sizing_mode="fixed")

    main_panel = pn.Column(
        pn.indicators.LoadingSpinner(
            value=True, color="primary", size=50, name="Loading..."
        ),
        sizing_mode="stretch_both",
    )
    sidebar_panel = pn.Column(
        pn.indicators.LoadingSpinner(
            value=True, color="primary", size=50, name="Loading..."
        )
    )
    template = pn.template.VanillaTemplate(
        title="DMS Datastore \u2014 Classic Explorer",
        sidebar=[sidebar_panel],
        main=[main_panel],
        sidebar_width=650,
        header_color="blue",
        logo="dms_datastore_ui/california-department-of-water-resources-logo.png",
        header=[header_link, reset_btn],
    )

    def load_explorer():
        explorer = dms_datastore_ui.map_inventory_explorer.StationInventoryExplorer(_REPO_DIR)
        view = explorer.create_view()

        sidebar_items = list(view.sidebar)
        main_items    = list(view.main)
        modal_items   = list(view.modal)
        view.sidebar.clear()
        view.main.clear()
        view.modal.clear()

        sidebar_panel.objects = sidebar_items
        main_panel.objects    = main_items

        template.modal.clear()
        for item in modal_items:
            template.modal.append(item)

        _session_mgr.set_entry(reg_key, {
            "template": template, "mode": "explorer", "mgr": None, "ui": None
        })

    pn.state.onload(load_explorer)
    template.servable(title="DMS Datastore \u2014 Classic Explorer")


# ── [6] Entry point ───────────────────────────────────────────────────────────
#
# Run:  python repoui.py [REPO_DIR] [--port PORT] [--address ADDRESS]
# The per_app_patterns patch above must execute before BokehServer starts;
# pn.serve(...) ensures module-level code runs only once.
#
# Routes:
#   /repoui  → new DataUI  (make_newui_app)
#   /oldui   → classic StationInventoryExplorer  (make_oldui_app)

import click


@click.command()
@click.argument("repo_dir", default="continuous", required=False)
@click.option("--port", default=80, show_default=True, help="Port to serve on.")
@click.option(
    "--address",
    default="0.0.0.0",
    show_default=True,
    help="Network address to bind to.",
)
def _serve(repo_dir: str, port: int, address: str) -> None:
    """Serve the DMS Datastore UI.

    REPO_DIR is the path to the continuous data repository
    (default: 'continuous').
    """
    global _REPO_DIR
    _REPO_DIR = repo_dir
    pn.serve(
        {"": make_newui_app, "oldui": make_oldui_app},
        port=port,
        address=address,
        allow_websocket_origin=["*"],
        keep_alive=30000,
        unused_session_lifetime_milliseconds=2_592_000_000,
        show=False,
    )


if __name__ == "__main__":
    _serve()
