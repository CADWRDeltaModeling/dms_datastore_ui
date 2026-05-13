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
# Run as:  python repoui.py [repo_dir]
# Do NOT use `panel serve` — the per_app_patterns patch must run before
# BokehServer.__init__().  pn.serve(make_app) handles this correctly.
#
# Two-layer persistence:
#   Layer 1 — _APP_REGISTRY (in-memory, same server process):
#     UUID cookie → registry entry → reuse existing mgr + ui + template.
#     Panel mirrors all Python widget state (plot tabs, selections, etc.)
#     into the new Bokeh Document automatically when template.servable() is
#     called.  No replay logic needed.
#   Layer 2 — diskcache (server restart):
#     Picklable params (time_range) are restored to a freshly created manager.
#     Dynamic tab content is not serialisable and is not restored.

import panel as pn
from uuid import uuid4
from bokeh.server.urls import per_app_patterns
from panel.io.server import DocHandler
import diskcache
from pathlib import Path
import pandas as pd


class _SessionAwareDocHandler(DocHandler):
    """Sets a persistent 'dvue_user_id' UUID cookie on first visit."""

    _COOKIE_NAME = "dvue_user_id"

    async def get(self, *args, **kwargs):
        user_id = self.get_cookie(self._COOKIE_NAME)
        if not user_id:
            user_id = uuid4().hex
            self.set_cookie(self._COOKIE_NAME, user_id, expires_days=365, path="/")
            self.request.cookies[self._COOKIE_NAME] = user_id
        await super().get(*args, **kwargs)


per_app_patterns[0] = (r"/?", _SessionAwareDocHandler)

# -- diskcache state store (server-restart fallback) -------------------------
_CACHE_DIR = Path(__file__).parent / ".session_cache"
_SESSION_CACHE = diskcache.Cache(str(_CACHE_DIR))
_TTL = 30 * 24 * 3600  # 30 days


def _load_state(user_id: str) -> dict:
    return _SESSION_CACHE.get(user_id, default={})


def _save_state(user_id: str, state: dict) -> None:
    _SESSION_CACHE.set(user_id, state, expire=_TTL)


# -- In-memory registry: user_id → {"template", "mode", "mgr", "ui"} --------
_APP_REGISTRY: dict = {}

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

# ── [4] Diskcache snapshot / restore ─────────────────────────────────────────


def _snapshot(mgr, ui) -> dict:
    """Picklable snapshot for diskcache (server-restart fallback only)."""
    tr = getattr(mgr, "time_range", None)
    tbl = getattr(ui, "display_table", None)
    return {
        "time_range": (
            [pd.Timestamp(tr[0]).isoformat(), pd.Timestamp(tr[1]).isoformat()]
            if tr else None
        ),
        "selection": list(tbl.selection or []) if tbl is not None else [],
    }


def _restore(mgr, saved: dict) -> None:
    """Apply diskcache params to a freshly created manager."""
    tr = saved.get("time_range")
    if tr:
        try:
            mgr.time_range = (
                pd.Timestamp(tr[0]).to_pydatetime(),
                pd.Timestamp(tr[1]).to_pydatetime(),
            )
        except Exception:
            pass


# ── [5] App factories (called once per Bokeh session / browser tab) ──────────
#
# make_newui_app  → served at "/"   (the default, DataUI)
# make_oldui_app  → served at "/oldui"  (classic StationInventoryExplorer)
#
# Registry key scheme: "{user_id}:newui" / "{user_id}:oldui" — keeps the two
# apps independent even though they share the same cookie-based user_id.
#
# Registry hit — DataUI mode (same server, returning user):
#   Reuse existing mgr + ui + template.  Panel automatically mirrors all
#   widget state into the new Bokeh Document when template.servable() is
#   called.  Only per-Document setup is re-registered via pn.state.onload.


def make_newui_app():
    user_id = pn.state.cookies.get("dvue_user_id", "")
    reg_key = f"{user_id}:newui" if user_id else ""
    entry = _APP_REGISTRY.get(reg_key) if reg_key else None
    reuse_dataui = bool(entry and entry.get("mode") == "dataui")

    header_link = pn.pane.HTML(
        '<a href="/oldui" style="color:white; font-size:0.9em; '
        'text-decoration:none; margin-left:1em; white-space:nowrap;">'
        "&#8594; Classic Explorer</a>",
        sizing_mode="fixed",
    )

    if reuse_dataui:
        # ── Registry hit: DataUI already built ──────────────────────────────
        template = entry["template"]
        ui = entry["ui"]

        def _reattach():
            if ui:
                ui.setup_location_sync()
                ui.setup_url_sync()

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
        saved = _load_state(user_id) if user_id else {}
        uimgr = DatastoreUIMgr(_REPO_DIR)
        if saved:
            _restore(uimgr, saved)

        ui = DataUI(uimgr, crs=ccrs.epsg(26910))
        ui_template = ui.create_view()

        sidebar_items = list(ui_template.sidebar)
        main_items = list(ui_template.main)
        modal_items = list(ui_template.modal)
        ui_template.sidebar.clear()
        ui_template.main.clear()
        ui_template.modal.clear()

        sidebar_panel.objects = sidebar_items
        main_panel.objects = main_items

        template.modal.clear()
        for item in modal_items:
            template.modal.append(item)

        if reg_key:
            _APP_REGISTRY[reg_key] = {
                "template": template, "mode": "dataui", "mgr": uimgr, "ui": ui
            }

        # Wire live-persistence watchers (diskcache save on state change).
        def _save(event=None):
            if user_id:
                _save_state(user_id, _snapshot(uimgr, ui))

        uimgr.param.watch(_save, "time_range")
        if hasattr(ui, "display_table"):
            ui.display_table.param.watch(_save, "selection")

    pn.state.onload(load_dataui)
    template.servable(title="DMS Datastore")


def make_oldui_app():
    user_id = pn.state.cookies.get("dvue_user_id", "")
    reg_key = f"{user_id}:oldui" if user_id else ""

    header_link = pn.pane.HTML(
        '<a href="/repoui" style="color:white; font-size:0.9em; '
        'text-decoration:none; margin-left:1em; white-space:nowrap;">'
        "&#8592; New UI</a>",
        sizing_mode="fixed",
    )

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
        header=[header_link],
    )

    def load_explorer():
        explorer = dms_datastore_ui.map_inventory_explorer.StationInventoryExplorer(_REPO_DIR)
        view = explorer.create_view()

        sidebar_items = list(view.sidebar)
        main_items = list(view.main)
        modal_items = list(view.modal)
        view.sidebar.clear()
        view.main.clear()
        view.modal.clear()

        sidebar_panel.objects = sidebar_items
        main_panel.objects = main_items

        template.modal.clear()
        for item in modal_items:
            template.modal.append(item)

        if reg_key:
            _APP_REGISTRY[reg_key] = {
                "template": template, "mode": "explorer", "mgr": None, "ui": None
            }

    pn.state.onload(load_explorer)
    template.servable(title="DMS Datastore \u2014 Classic Explorer")


# ── [6] Entry point ───────────────────────────────────────────────────────────
#
# Run:  python repoui.py [repo_dir]
# The per_app_patterns patch above must execute before BokehServer starts;
# pn.serve(...) ensures module-level code runs only once.
#
# Routes:
#   /repoui  → new DataUI  (make_newui_app)
#   /oldui   → classic StationInventoryExplorer  (make_oldui_app)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        _REPO_DIR = sys.argv[1]
    pn.serve(
        {"": make_newui_app, "oldui": make_oldui_app},
        port=80,
        address="0.0.0.0",
        allow_websocket_origin=["*"],
        keep_alive=30000,
        unused_session_lifetime_milliseconds=2_592_000_000,
        show=False,
    )
