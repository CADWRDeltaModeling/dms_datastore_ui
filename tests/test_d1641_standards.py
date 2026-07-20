"""Tests for dms_datastore_ui.d1641_standards."""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dms_datastore_ui.d1641_standards import (
    DEFAULT_STANDARDS,
    DSM2_TO_DATASTORE_ID,
    STATION_COORDS,
    D1641StandardSpec,
    _classify_svi,
    _no_wsi_result,
    _parse_wsi_text,
    build_d1641_references,
    fetch_current_wyt,
    read_hist_wateryear_types,
    rsl_ec_to_cl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal WSIHIST-format text: 14 header lines then 5 data rows covering WYs
# 2015–2019 with all five Sacramento Valley water year types.
_WSIHIST_CONTENT = textwrap.dedent("""\
    CDEC Water Supply Index History - Sacramento Valley
    ====================================================
    (This file is a test fixture; values are illustrative only)

    Sac Valley Water Year Runoff Indices (1000s of acre-feet)
    and Water Year Classifications

    Oct-Mar  Apr-Jul  WY Sum  WY Index  WY Type
    -------------------------------------------------
    Columns: WY  SAC-Oct/Mar  SAC-Apr/Jul  SAC-WYsum  SAC-Index  SAC-Type
             SJR-Oct/Mar  SJR-Apr/Jul  SJR-WYsum  SJR-Index  SJR-Type
    -------------------------------------------------
    (source: http://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST)
    WY   SAC_OM   SAC_AJ  SAC_WYS SAC_IDX SAC_TYP  SJR_OM   SJR_AJ  SJR_WYS SJR_IDX SJR_TYP
    2015  6.43    5.12    11.55   11.55    D    2.51    3.04    5.55    5.55    D
    2016  9.02    9.15    18.17   18.17    W    4.62    6.21   10.83   10.83    W
    2017  5.12    6.33    11.45   11.45   BN    3.10    4.45    7.55    7.55   BN
    2018  4.22    5.11     9.33    9.33   AN    2.88    3.55    6.43    6.43   AN
    2019  2.11    2.05     4.16    4.16    C    1.20    1.55    2.75    2.75    C
""")


@pytest.fixture
def wyt_file(tmp_path):
    """Write a minimal WSIHIST fixture to tmp_path and return its path."""
    fpath = tmp_path / "wsihist.txt"
    fpath.write_text(_WSIHIST_CONTENT)
    return str(fpath)


# ---------------------------------------------------------------------------
# read_hist_wateryear_types
# ---------------------------------------------------------------------------


def test_read_hist_wateryear_types_columns(wyt_file):
    df = read_hist_wateryear_types(wyt_file)
    assert "wy" in df.columns
    assert "sac_yrtype" in df.columns


def test_read_hist_wateryear_types_wyt_values(wyt_file):
    df = read_hist_wateryear_types(wyt_file)
    # All five year types should be present.
    assert set(df["sac_yrtype"].dropna().unique()) == {"D", "W", "BN", "AN", "C"}
    assert 2015 in df["wy"].values
    assert 2019 in df["wy"].values


# ---------------------------------------------------------------------------
# rsl_ec_to_cl
# ---------------------------------------------------------------------------


def test_rsl_ec_to_cl_scalar():
    # At EC=1053 µS/cm: max(1053*0.15-12, 1053*0.285-50) = max(145.95, 250.1) ≈ 250
    cl = rsl_ec_to_cl(1053.0)
    assert abs(cl - 250.1) < 0.5


def test_rsl_ec_to_cl_series():
    s = pd.Series([500.0, 1000.0, 1500.0])
    result = rsl_ec_to_cl(s)
    assert isinstance(result, pd.Series)
    assert len(result) == 3
    # At EC=1000: max(1000*0.15-12, 1000*0.285-50) = max(138, 235) = 235
    assert abs(result.iloc[1] - 235.0) < 0.1


def test_rsl_ec_to_cl_array():
    arr = np.array([1000.0, 1500.0])
    result = rsl_ec_to_cl(arr)
    assert isinstance(result, np.ndarray)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# build_d1641_references — basic structure
# ---------------------------------------------------------------------------


def test_build_d1641_references_count(wyt_file):
    refs = build_d1641_references(wyt_file)
    # AG WI:         emm + jer = 2
    # FWS SJR:       jer       = 1
    # AG S DELTA:    ver + bdt + old = 3
    # AG EXPORT:     trp + hbp = 2
    # FWS SUISUN:    cse + nsl + bdl = 3
    # MI 250:        rsl + hbp + trp + bks + ccs = 5
    # MI 150:        rsl = 1
    # Total = 17
    assert len(refs) == 17


def test_build_d1641_references_names(wyt_file):
    refs = build_d1641_references(wyt_file)
    names = {r.name for r in refs}
    # Original standards
    assert "emm_D1641_AG_WI" in names
    assert "jer_D1641_AG_WI" in names
    assert "jer_D1641_FWS_SJR" in names
    assert "emm_D1641_FWS_SJR" not in names  # Emmaton not in FWS SJR standard
    assert "rsl_D1641_MI_250" in names
    # New standards
    assert "ver_D1641_AG_S_DELTA" in names
    assert "bdt_D1641_AG_S_DELTA" in names
    assert "old_D1641_AG_S_DELTA" in names
    assert "trp_D1641_AG_EXPORT" in names
    assert "hbp_D1641_AG_EXPORT" in names
    assert "cse_D1641_FWS_SUISUN" in names
    assert "nsl_D1641_FWS_SUISUN" in names
    assert "bdl_D1641_FWS_SUISUN" in names
    assert "hbp_D1641_MI_250" in names
    assert "trp_D1641_MI_250" in names
    assert "bks_D1641_MI_250" in names
    assert "ccs_D1641_MI_250" in names
    assert "rsl_D1641_MI_150" in names


def test_build_d1641_references_station_ids(wyt_file):
    refs = build_d1641_references(wyt_file)
    station_ids = {r.get_attribute("station_id") for r in refs}
    assert {"emm", "jer", "rsl", "ver", "bdt", "old",
            "trp", "hbp", "cse", "nsl", "bdl", "bks", "ccs"} == station_ids


def test_build_d1641_references_params(wyt_file):
    refs = build_d1641_references(wyt_file)
    params = {r.get_attribute("param") for r in refs}
    assert params == {
        "D1641_AG_WI", "D1641_FWS_SJR",
        "D1641_AG_S_DELTA", "D1641_AG_EXPORT",
        "D1641_FWS_SUISUN",
        "D1641_MI_250", "D1641_MI_150",
    }


def test_build_d1641_references_units(wyt_file):
    refs = build_d1641_references(wyt_file)
    for ref in refs:
        param = ref.get_attribute("param")
        unit  = ref.get_attribute("unit")
        if param in ("D1641_MI_250", "D1641_MI_150"):
            assert unit == "mg/L", f"Expected mg/L for {param}, got {unit}"
        else:
            assert unit == "microS/cm", f"Expected microS/cm for {param}, got {unit}"


def test_build_d1641_references_agency(wyt_file):
    refs = build_d1641_references(wyt_file)
    for ref in refs:
        assert ref.get_attribute("agency") == "swrcb"


def test_build_d1641_references_agency_id_dbase(wyt_file):
    """agency_id_dbase should carry the DSM2 model station ID for traceability."""
    refs = build_d1641_references(wyt_file)
    by_name = {r.name: r.get_attribute("agency_id_dbase") for r in refs}
    assert by_name["emm_D1641_AG_WI"]      == "RSAC092"
    assert by_name["jer_D1641_AG_WI"]      == "RSAN018"
    assert by_name["jer_D1641_FWS_SJR"]    == "RSAN018"
    assert by_name["rsl_D1641_MI_250"]     == "ROLD024"
    assert by_name["ver_D1641_AG_S_DELTA"] == "RSAN112"
    assert by_name["rsl_D1641_MI_150"]     == "ROLD024"


def test_build_d1641_references_subloc_empty(wyt_file):
    refs = build_d1641_references(wyt_file)
    for ref in refs:
        assert ref.get_attribute("subloc") == ""


def test_build_d1641_references_geometry(wyt_file):
    refs = build_d1641_references(wyt_file)
    for ref in refs:
        geom = ref.get_attribute("geometry")
        assert geom is not None
        sid = ref.get_attribute("station_id")
        expected_x, expected_y = STATION_COORDS[sid]
        assert abs(geom.x - expected_x) < 1.0
        assert abs(geom.y - expected_y) < 1.0


def test_build_d1641_references_year_range(wyt_file):
    refs = build_d1641_references(wyt_file)
    for ref in refs:
        min_yr = ref.get_attribute("min_year")
        max_yr = ref.get_attribute("max_year")
        assert min_yr is not None
        assert max_yr is not None
        assert min_yr <= max_yr


# ---------------------------------------------------------------------------
# getData — series content
# ---------------------------------------------------------------------------


def test_get_data_emmaton_ag_wi(wyt_file):
    """Emmaton AG WI series should be non-empty with active regulation values and NaN gaps."""
    refs = build_d1641_references(wyt_file)
    ref = next(r for r in refs if r.name == "emm_D1641_AG_WI")
    df = ref.getData()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "value" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    # No zero values — off-season zeros are replaced with NaN.
    assert (df["value"] == 0.0).sum() == 0, "Off-season zeros should be NaN, not 0"
    # Active regulation periods exist (positive values present).
    active = df["value"].dropna()
    assert len(active) > 0
    assert active.max() <= 4000.0  # headroom above max regulation value (2780 µS/cm)
    # NaN gaps exist during off-season months.
    assert df["value"].isna().any(), "Expected NaN gaps for off-season periods"


def test_get_data_mi_250_constant(wyt_file):
    """Rock Slough MI 250 standard should be a constant 250 mg/L series."""
    refs = build_d1641_references(wyt_file)
    ref = next(r for r in refs if r.name == "rsl_D1641_MI_250")
    df = ref.getData()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "value" in df.columns
    # All values must be exactly 250 mg/L (constant across all WY types)
    assert (df["value"] == 250.0).all(), "MI 250 standard should be a constant 250 mg/L"


def test_get_data_datetime_index(wyt_file):
    """All standard series should have a DatetimeIndex named 'datetime'."""
    refs = build_d1641_references(wyt_file)
    for ref in refs:
        df = ref.getData()
        if not df.empty:
            assert isinstance(df.index, pd.DatetimeIndex)


def test_get_data_covers_wyt_period(wyt_file):
    """AG WI series should span the WY period in the fixture (2015–2019)."""
    refs = build_d1641_references(wyt_file)
    ref = next(r for r in refs if r.name == "emm_D1641_AG_WI")
    df = ref.getData()
    # Use the full index (including NaN rows) for span check.
    assert df.index.min() <= pd.Timestamp("2015-10-01")
    assert df.index.max() >= pd.Timestamp("2019-09-01")


# ---------------------------------------------------------------------------
# Custom standards and station_id_map override
# ---------------------------------------------------------------------------


def test_custom_standards_single(wyt_file):
    """Passing a single standard spec returns only that standard's references."""
    specs = [s for s in DEFAULT_STANDARDS if s.name == "D1641_AG_WI"]
    refs = build_d1641_references(wyt_file, standards=specs)
    assert len(refs) == 2  # emm + jer
    assert all(r.get_attribute("param") == "D1641_AG_WI" for r in refs)


def test_custom_station_id_map(wyt_file):
    """Override station_id_map should remap DSM2 IDs to custom station IDs."""
    custom_map = {"RSAC092": "emm_custom", "RSAN018": "jer", "ROLD024": "rsl"}
    custom_coords = {
        "emm_custom": (610598.8, 4215917.6),
        "jer": (615024.3, 4212414.2),
        "rsl": (619885.2, 4204031.0),
    }
    specs = [s for s in DEFAULT_STANDARDS if s.name == "D1641_AG_WI"]
    refs = build_d1641_references(wyt_file, standards=specs,
                                  station_id_map=custom_map,
                                  station_coords=custom_coords)
    names = {r.name for r in refs}
    assert "emm_custom_D1641_AG_WI" in names


# ---------------------------------------------------------------------------
# _classify_svi
# ---------------------------------------------------------------------------


def test_classify_svi_wet():
    assert _classify_svi(9.2)  == "W"
    assert _classify_svi(15.0) == "W"


def test_classify_svi_above_normal():
    assert _classify_svi(7.81) == "AN"   # just above AN threshold
    assert _classify_svi(9.1)  == "AN"


def test_classify_svi_below_normal():
    assert _classify_svi(6.51) == "BN"
    assert _classify_svi(7.8)  == "BN"   # exactly 7.8 is BN (≤ 7.8, not AN)


def test_classify_svi_dry():
    assert _classify_svi(5.41) == "D"
    assert _classify_svi(6.5)  == "D"    # exactly 6.5 is D (≤ 6.5, not BN)


def test_classify_svi_critical():
    assert _classify_svi(5.4)  == "C"   # exactly 5.4 is C (≤ 5.4)
    assert _classify_svi(0.0)  == "C"


# ---------------------------------------------------------------------------
# _parse_wsi_text
# ---------------------------------------------------------------------------

# Minimal sample matching the CDEC WSI pre-block format.
_WSI_SAMPLE_FINAL = """
2026 Water Year Hydrologic Classification Indices
                    2026 Water Year Forecast as of May 1, 2026

SACRAMENTO VALLEY WATER YEAR TYPE INDEX  40-30-30   (SVI)
                             Probability of Exceedance
Forecast Date      99%        90%        75%        50%        25%        10%
--------------------------------------------------------------------------------
Dec 1, 2025       5.03       5.71       6.63       8.09      10.13      12.50
Jan 1, 2026       6.22       6.87       7.81       9.08      10.90      12.92
May 1, 2026       7.48       7.56       7.66       7.81       8.15       8.36

     Index =     0.4 * Current Apr-Jul Runoff   (1)
SAN JOAQUIN VALLEY WATER YEAR TYPE INDEX
"""

_WSI_SAMPLE_PROVISIONAL = """
2027 Water Year Hydrologic Classification Indices
                    2027 Water Year Forecast as of Mar 1, 2027

SACRAMENTO VALLEY WATER YEAR TYPE INDEX  40-30-30   (SVI)
                             Probability of Exceedance
Forecast Date      99%        90%        75%        50%        25%        10%
--------------------------------------------------------------------------------
Dec 1, 2026       3.10       3.80       4.70       5.80       7.20       9.00
Mar 1, 2027       4.50       4.90       5.20       5.60       6.10       6.80

     Index =     0.4 * Current Apr-Jul Runoff   (1)
SAN JOAQUIN VALLEY WATER YEAR TYPE INDEX
"""


def test_parse_wsi_text_final():
    """May 1 forecast → WY 2026 = AN, is_final=True."""
    result = _parse_wsi_text(_WSI_SAMPLE_FINAL, 2026)
    assert result["wy"] == 2026
    assert result["wyt"] == "AN"        # 7.81 > 7.8 → AN
    assert result["is_final"] is True
    assert result["source"] == "wsi_live"
    assert abs(result["svi_50pct"] - 7.81) < 0.01
    assert result["forecast_date"] == pd.Timestamp("2026-05-01")


def test_parse_wsi_text_provisional():
    """Mar 1 forecast → WY 2027 = D, is_final=False."""
    result = _parse_wsi_text(_WSI_SAMPLE_PROVISIONAL, 2027)
    assert result["wy"] == 2027
    assert result["wyt"] == "D"         # 5.60 → D (5.4 < 5.60 ≤ 6.5)
    assert result["is_final"] is False
    assert result["forecast_date"] == pd.Timestamp("2027-03-01")


def test_parse_wsi_text_bad_input():
    """Malformed text returns a no-data result."""
    result = _parse_wsi_text("no data here", 2026)
    assert result["wyt"] is None
    assert result["source"] == "none"


# ---------------------------------------------------------------------------
# fetch_current_wyt
# ---------------------------------------------------------------------------


def test_fetch_current_wyt_oct_nov_returns_none(monkeypatch):
    """Oct and Nov return no-data without any network call."""
    import dms_datastore_ui.d1641_standards as m
    monkeypatch.setattr("pandas.Timestamp.now",
                        lambda: pd.Timestamp("2026-10-15"))
    result = m.fetch_current_wyt(cache_dir=None)
    assert result["wyt"] is None
    assert result["source"] == "none"
    assert result["wy"] == 2027   # Oct 2026 is WY 2027


def test_fetch_current_wyt_uses_cache(tmp_path, monkeypatch):
    """Second call within the same month returns cached result."""
    import diskcache
    import dms_datastore_ui.d1641_standards as m

    # Pretend it's April 2026
    monkeypatch.setattr("pandas.Timestamp.now",
                        lambda: pd.Timestamp("2026-04-10"))

    # Seed the cache with a known result
    fake_result = {
        "wy": 2026, "wyt": "BN", "svi_50pct": 7.0,
        "forecast_date": pd.Timestamp("2026-04-01"),
        "is_final": False, "source": "wsi_live",
    }
    cache = diskcache.Cache(str(tmp_path))
    cache.set("wsi_wyt_2026_04", fake_result, expire=60 * 60 * 24 * 30)

    result = m.fetch_current_wyt(cache_dir=str(tmp_path))
    assert result["source"] == "wsi_cached"
    assert result["wyt"] == "BN"
