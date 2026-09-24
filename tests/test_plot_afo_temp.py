"""Visual regression test for the DatastorePlotAction pipeline.

Plots station_id="afo", param="temp" over the manager's default time
window and writes an HTML artifact for manual inspection, guarding
against the dvue reference-axis "blank plot" regression (JS syntax error
from redeclaring the `tick` formatter parameter).

Run from the dms_datastore_ui root directory:
    conda activate dsm2ui
    python -m pytest tests/test_plot_afo_temp.py -v -s --repo=continuous
"""
import os

import pytest


@pytest.fixture(scope="module")
def repo_dir(request):
    d = request.config.getoption("--repo", default=None)
    if d is None:
        pytest.skip("Pass --repo=<path> to run this test")
    return d


@pytest.fixture(scope="module")
def manager(repo_dir):
    from dms_datastore_ui.datastore_uimgr import DatastoreUIMgr
    return DatastoreUIMgr(dir=repo_dir)


def test_plot_afo_temp_default_window(manager):
    """Render the afo/temp curve and assert it is non-empty and free of the
    known JS-formatter syntax error that produced a blank plot."""
    import holoviews as hv

    hv.extension("bokeh")

    df = manager.get_data_catalog()
    sel = df[(df["station_id"] == "afo") & (df["param"] == "temp")].reset_index(drop=True)
    assert not sel.empty, "afo/temp not found in catalog"

    action = manager._make_plot_action()
    refs_and_data = list(action.get_refs_and_data(sel, manager))
    assert refs_and_data and refs_and_data[0][2] is not None, "no data loaded for afo/temp"
    assert not refs_and_data[0][2].empty, "afo/temp data is empty for default window"

    layout = action.render(sel, refs_and_data, manager)

    plot = hv.renderer("bokeh").get_plot(layout)
    from bokeh.models import Axis

    for axis in plot.state.select({"type": Axis}):
        formatter = getattr(axis, "formatter", None)
        code = getattr(formatter, "code", None)
        if code:
            assert "const tick" not in code, "formatter redeclares `tick` -- causes blank plot"
            assert "let tick" not in code, "formatter redeclares `tick` -- causes blank plot"

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "afo_temp_plot.html")
    hv.save(layout, out_path)
    print(f"\nSaved visual output: {out_path}")
    assert os.path.getsize(out_path) > 0
