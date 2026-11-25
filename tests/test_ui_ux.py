from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import panel_material_ui as pmui
import param

from dms_datastore_ui.datastore_uimgr import (
    DatastoreUIMgr,
    _shared_d1641_reference_templates,
)
from dms_datastore_ui.map_inventory_explorer import _shared_inventory_snapshot


def _bare_manager():
    manager = DatastoreUIMgr.__new__(DatastoreUIMgr)
    param.Parameterized.__init__(manager)
    return manager


def test_datastore_multi_selects_are_touch_friendly():
    manager = _bare_manager()
    manager._wyt_df = None

    map_widget = manager.get_map_option_widgets()[1]
    data_widget = manager.get_widgets()["Data"][0]

    assert isinstance(map_widget, pmui.MultiChoice)
    assert map_widget.label == "Parameter types"
    assert isinstance(data_widget, pmui.MultiChoice)
    assert data_widget.label == "Repository levels"
    assert "ctrl" not in manager.param.repo_level.doc.lower()
    assert "ctrl" not in manager.param.parameter_type.doc.lower()


def test_default_time_range_uses_catalog_years_and_is_configurable():
    manager = _bare_manager()
    catalog = pd.DataFrame(
        {
            "max_year": [2010, 2024, 2039, 2027],
            "source": ["archive", "archive", "forecast", "d1641"],
        }
    )

    start, end = manager.get_time_range(catalog)
    assert (start.year, end.year) == (2023, 2024)

    manager.default_window_years = 3
    start, end = manager.get_time_range(catalog)
    assert (start.year, end.year) == (2022, 2024)


def test_wyt_opacity_change_replots_current_selection(monkeypatch):
    manager = _bare_manager()
    calls = []
    fake_action = SimpleNamespace(callback=lambda event, dataui: calls.append(dataui))
    fake_table = SimpleNamespace(selection=[0])
    fake_ui = SimpleNamespace(display_table=fake_table)
    manager._dataui = fake_ui
    monkeypatch.setattr(manager, "_make_plot_action", lambda: fake_action)

    manager.wyt_band_alpha = 0.1

    assert calls == [fake_ui]


def test_secondary_actions_are_menu_driven_and_low_emphasis(monkeypatch):
    manager = _bare_manager()
    monkeypatch.setattr(
        manager,
        "_make_plot_action",
        lambda: SimpleNamespace(callback=lambda event, dataui: None),
    )
    actions = manager.get_data_actions()
    by_name = {action["name"]: action for action in actions}

    assert by_name["Plot"]["button_type"] == "primary"
    assert by_name["Data Tools"]["action_type"] == "menu"
    assert set(by_name["Data Tools"]["items"]) == {
        "Data Screener",
        "Flag Editor",
        "Gap Visualizer",
    }
    assert all(
        action.get("button_type") == "light"
        for name, action in by_name.items()
        if name != "Plot"
    )


def test_low_value_table_columns_are_hidden_by_default():
    manager = _bare_manager()
    schema = manager.get_table_schema(df=pd.DataFrame())
    hidden = manager.get_hidden_table_columns(pd.DataFrame())
    assert "agency_id_dbase" in schema["hidden_by_default"]
    assert "agency_id_dbase" in hidden


def test_inventory_snapshot_is_shared_by_inventory_version(monkeypatch):
    import dms_datastore_ui.map_inventory_explorer as module

    row = {
        "file_pattern": "*.csv",
        "filename": "station.csv",
        "series_id": "station_flow",
        "station_id": "sta",
        "subloc": "",
        "name": "Station",
        "unit": "cfs",
        "param": "flow",
        "modifier": "",
        "min_year": 2020,
        "max_year": 2024,
        "agency": "test",
        "agency_id_dbase": "1",
        "x": 1.0,
        "y": 2.0,
    }
    calls = []

    def fake_read_csv(path):
        calls.append(path)
        return pd.DataFrame([row])

    _shared_inventory_snapshot.cache_clear()
    monkeypatch.setattr(module.pd, "read_csv", fake_read_csv)

    first = _shared_inventory_snapshot("inventory_datasets_screened_test.csv", 1.0)
    second = _shared_inventory_snapshot("inventory_datasets_screened_test.csv", 1.0)

    assert calls == ["inventory_datasets_screened_test.csv"]
    assert second is first
    assert first[2] == ("flow",)


def test_shared_d1641_reference_templates_accept_default_specs():
    from dms_datastore_ui.d1641_standards import DEFAULT_STANDARDS, get_default_wyt_file

    wyt_file = get_default_wyt_file()
    specs_key = tuple(
        (
            spec.name,
            spec.csv_filename,
            spec.unit_scale,
            spec.unit,
            tuple(spec.dsm2_locations),
        )
        for spec in DEFAULT_STANDARDS
    )

    _shared_d1641_reference_templates.cache_clear()
    refs = _shared_d1641_reference_templates(
        wyt_file,
        Path(wyt_file).stat().st_mtime,
        specs_key,
        2026,
    )

    assert len(refs) == 19


def test_dead_home_page_is_removed():
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "home.py").exists()


def test_root_ui_defers_manager_construction_until_onload(monkeypatch):
    import panel as pn
    import repoui

    callbacks = []
    constructions = []

    class FakeSessionManager:
        current_user_id = "test-user"

        @staticmethod
        def make_reg_key(user_id, mode):
            return f"{user_id}:{mode}"

        @staticmethod
        def get_entry(key):
                raise AssertionError("Panel models must not be reused across documents")

    monkeypatch.setattr(repoui, "_session_mgr", FakeSessionManager())
    monkeypatch.setattr(
        repoui,
        "DatastoreUIMgr",
        lambda *args, **kwargs: constructions.append((args, kwargs)),
    )
    monkeypatch.setattr(pn.state, "onload", lambda callback: callbacks.append(callback))

    repoui.make_newui_app()

    assert constructions == []
    assert len(callbacks) == 1
