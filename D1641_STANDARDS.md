# D-1641 Water Quality Standards in dms_datastore_ui

This document describes the State Water Resources Control Board (SWRCB) Decision 1641
(Revised Decision 1641, December 1999) regulatory standards that are generated as
time series and added to the `DatastoreUIMgr` data catalog.

## Overview

D-1641 specifies water quality objectives for the Sacramento–San Joaquin Delta and
San Francisco Bay–Delta Estuary to protect agricultural, municipal & industrial, and
fish & wildlife uses.  Standards are grouped into three categories:

| Category | Metric | Stations |
|----------|--------|---------|
| Agricultural (AG) | EC (µS/cm) – running average | Multiple Delta locations |
| Municipal & Industrial (MI) | Chloride (mg/L) – daily | Export intakes |
| Fish & Wildlife Service (FWS) | EC (µS/cm) – monthly maximum | Suisun Marsh, SJR |

## How WYT-Dependent Standards Work

Most D-1641 AG and FWS standards are conditioned on the **Sacramento Valley Water Year
Type (WYT)**, which is classified annually based on the Sacramento Valley Index (SVI):

```
SVI = 0.4 × Apr–Jul Runoff Forecast (MAF)
    + 0.3 × Oct–Mar Runoff (MAF)
    + 0.3 × Previous Year's Index (max 10.0)
```

| WYT | Index threshold | Color in UI |
|-----|----------------|-------------|
| W (Wet) | SVI ≥ 9.2 | dark blue |
| AN (Above Normal) | SVI > 7.8 | light blue |
| BN (Below Normal) | SVI > 6.5 | grey |
| D (Dry) | SVI > 5.4 | orange |
| C (Critical) | SVI ≤ 5.4 | red |

The WYT is determined progressively:

* **Dec 1 – Apr 30**: provisional monthly forecast; standards are set but may change.
* **May 1**: *final* determination (50 % exceedance forecast).  Standards are fixed for the
  remaining water year.
* **Oct – Nov**: no forecast yet; the previous year's type is used as a proxy.

Historical WYTs (1906–present) are read from a bundled copy of the CDEC WSIHIST
file (`dms_datastore_ui/data/wsihist.txt`).  The current year's type is fetched from
the live CDEC WSI report and cached monthly.

---

## Standard Sets

### 1. D1641_AG_WI — Agricultural, Western & Interior Delta

**Metric**: 14-day running average of daily mean EC (µS/cm)  
**D-1641 reference**: Table 2-1, D-22, D-15  
**Applicable period**: April – August (off-season values = 0 → displayed as NaN)

| dms_datastore ID | DSM2 ID | Location | D-1641 Compliance Point |
|-----------------|---------|----------|------------------------|
| `emm` | RSAC092 | Sacramento River at Emmaton | D-22 |
| `jer` | RSAN018 | San Joaquin River at Jersey Point | D-15 |

**Standard values** (µS/cm, after × 1000 conversion from mmhos/cm):

| WYT | Apr–Aug period threshold |
|-----|--------------------------|
| W   | 450 µS/cm (Apr 1 – Aug 15) |
| AN  | 450 → 630 µS/cm (Apr 1 → Jul 2) |
| BN  | 450 → 1140 µS/cm (Apr 1 → Jun 21) |
| D   | 450 → 1670 µS/cm (Apr 1 → Jun 16) |
| C   | 2780 µS/cm (Apr 1 – Aug 15) |

---

### 2. D1641_FWS_SJR — Fish & Wildlife Service, San Joaquin River

**Metric**: 14-day running average of daily mean EC (µS/cm)  
**D-1641 reference**: Table 2-2, D-15  
**Applicable stations**: Jersey Point only (Emmaton is not a FWS SJR compliance point)

| dms_datastore ID | DSM2 ID | Location |
|-----------------|---------|----------|
| `jer` | RSAN018 | San Joaquin River at Jersey Point |

**Notes**: The FWS SJR standard also includes a conditional low-flow provision under
D-1641 that tightens the standard at Jersey Point and Prisoners Point when Sacramento
River flows are critically low.  That conditional is **not** applied in this implementation;
only the base standard is shown.

---

### 3. D1641_AG_S_DELTA — Agricultural, South Delta

**Metric**: 30-day running average of daily mean EC (µS/cm)  
**D-1641 reference**: Table 2-1  
**Standard**: 1000 µS/cm (Oct–Apr); **700 µS/cm** (Apr 30 – Sep 1)  
**WYT dependency**: None — same standard regardless of water year type.

| dms_datastore ID | DSM2 ID | Location | D-1641 Ref |
|-----------------|---------|----------|-----------|
| `ver` | RSAN112 | SJR at Vernalis | C-10 |
| `bdt` | RSAN072 | SJR at Brandt Bridge | C-6 |
| `old` | ROLD059 | Old River at Tracy Blvd | P-12 |

**Note**: The model aggregate point OLD_MID (Old River near Middle River, C-8) is omitted
because no single dms_datastore monitoring station corresponds to that model location.

---

### 4. D1641_AG_EXPORT — Agricultural, Export Pumping Plants

**Metric**: Monthly average of daily mean EC (µS/cm)  
**D-1641 reference**: Table 2-1 export objectives  
**Standard**: 1000 µS/cm year-round  
**WYT dependency**: None.

| dms_datastore ID | DSM2 ID | Location | D-1641 Ref |
|-----------------|---------|----------|-----------|
| `trp` | CHDMC006 | Tracy / Delta-Mendota Canal PP | DMC-1 |
| `hbp` | CHSWP003 | Harvey O. Banks Pumping Plant | C-9 |

---

### 5. D1641_FWS_SUISUN — Fish & Wildlife Service, Suisun Marsh

**Metric**: Monthly mean of daily-maximum EC (or average of two high tides), in µS/cm  
**D-1641 reference**: Table 2-2, Suisun Marsh Objectives  
**WYT dependency**: None — the standard values are identical for all WYTs.

| dms_datastore ID | DSM2 ID | Location | D-1641 Ref |
|-----------------|---------|----------|-----------|
| `cse` | RSAC081 | Sacramento River at Collinsville | C-2 |
| `nsl` | SLMZU025 | Montezuma Slough at National Steel | S-64 |
| `bdl` | SLMZU011 | Montezuma Slough near Beldons Landing | S-49 |

**Seasonal standard schedule** (same for all WYTs; values in µS/cm):

| Month start | EC maximum |
|------------|-----------|
| Oct 1 | 19 000 µS/cm |
| Nov 1 | 15 500 µS/cm |
| Jan 1 | 12 500 µS/cm |
| Feb 1 | 8 000 µS/cm |
| Apr 1 | 11 000 µS/cm |
| Jun 1 | — (no standard; NaN) |

**Interpretation**: These are *maximum* EC ceilings to protect managed wetland habitat in
Suisun Marsh from over-salination in fall and under-freshening in winter.  Higher EC is
permitted in October (early fall) and in April (spring inflow).  No standard applies
June through September (summer).

**Notes**: D-1641 also includes a conditional tightening of the Suisun Marsh standard
during low-flow years (when Sacramento River runoff is below certain thresholds).  That
conditional requires the Sacramento River Index and is **not** applied here.

---

### 6. D1641_MI_250 — Municipal & Industrial, 250 mg/L Chloride

**Metric**: Daily chloride ≤ 250 mg/L  
**D-1641 reference**: Table 2-3  
**WYT dependency**: None — constant 250 mg/L throughout the year.

| dms_datastore ID | DSM2 ID | Location | D-1641 Ref |
|-----------------|---------|----------|-----------|
| `rsl` | ROLD024 | Rock Slough at Contra Costa Canal | C-5 |
| `hbp` | CHSWP003 | Harvey O. Banks Pumping Plant | C-9 |
| `trp` | CHDMC006 | Tracy / Delta-Mendota Canal PP | DMC-1 |
| `bks` | SLBAR002 | Barker Slough Pumping Plant | C-16 (SLSAR3) |
| `ccs` | SLCCH016 | Cache Slough at Vallejo | C-19 |

**Comparison with observed data**: The observed monitoring at these stations is EC
(µS/cm).  To compare EC against the 250 mg/L chloride standard, the Rock Slough
regression `rsl_ec_to_cl(EC) = max(EC × 0.15 − 12, EC × 0.285 − 50)` converts Rock
Slough EC to chloride.  In the UI, selecting both an observed EC series and the MI 250
standard will display them on **dual Y-axes** (EC left, mg/L right) without conversion.

#### Rock Slough Compliance Notes

Rock Slough (C-5) is the Contra Costa Water District (CCWD) intake at the head of the
Contra Costa Canal.  It is one of the most challenging D-1641 compliance points because
the southern Delta is susceptible to saline intrusion, particularly during dry and critical
years.

Compliance is managed through:

1. **Temporary Rock Slough Barrier** — DWR installs a rock-fill closure to block saline
   tidal water from entering the Old River / Rock Slough arm, typically in spring.
2. **Los Vaqueros Reservoir** — CCWD stores water when Delta quality is acceptable; draws
   from reservoir when Rock Slough exceeds 250 mg/L.
3. **CVP/SWP coordinated operations** — export operators increase Delta outflow to maintain
   adequate salinity buffer at C-5.

There is no formal alternative compliance location in D-1641 for C-5; the standard is
measured at the intake itself.

---

### 7. D1641_MI_150 — Municipal & Industrial, Stricter Seasonal Chloride (Rock Slough)

**Metric**: Daily chloride (mg/L) — more stringent than the 250 mg/L baseline  
**D-1641 reference**: Additional provision for Contra Costa Canal  
**Station**: Rock Slough (`rsl`, ROLD024) only

The standard tightens from January 1 depending on the water year type:

| WYT | Threshold (mg/L from Jan 1) |
|-----|-----------------------------|
| W   | 240 mg/L |
| AN  | 190 mg/L |
| BN  | 175 mg/L |
| D   | 165 mg/L |
| C   | 155 mg/L |

This means that in a Critical year, CCWD must maintain chloride **below 155 mg/L** at
Rock Slough from January onward — far more restrictive than the year-round 250 mg/L
baseline.  During Oct–Dec the previous year's WYT standard carries forward until January.

---

### 8. D1641_AG_TERMINOUS — Agricultural, Interior Delta (Terminous)

**Metric**: 14-day running average of daily mean EC (µS/cm)  
**D-1641 reference**: Table 2-3, C-13  
**Applicable period**: April 1 – August 15 (off-season values = 0 → displayed as NaN)  
**WYT dependency**: Yes — Critical Dry year uses a higher (relaxed) threshold.

| dms_datastore ID | DSM2 ID | Location | D-1641 Ref |
|-----------------|---------|----------|-----------|
| `sti` | RSMKL008 | South Fork Mokelumne River at Terminous (Staten Island) | C-13 |

**Standard values** (µS/cm, after × 1000 conversion from mmhos/cm):

| WYT | Apr 1 – Aug 15 threshold |
|-----|--------------------------|
| W   | 450 µS/cm |
| AN  | 450 µS/cm |
| BN  | 450 µS/cm |
| D   | 450 µS/cm |
| C   | 540 µS/cm |

Unlike the Western Delta stations (Emmaton, Jersey Point), the Terminous standard does
**not** have a mid-season transition date.  The same EC ceiling applies from April 1
continuously through August 15 for all water year types.  The Critical Dry relaxation
(0.54 vs. 0.45 mmhos/cm) reflects reduced Sacramento River and Mokelumne River
inflows in critically dry years.

**Current compliance history** (from DWR D-1641 Compliance Reports): All EC objectives
at this location have been met since D-1641 was adopted.

---

### 9. D1641_AG_SAN_ANDREAS — Agricultural, Interior Delta (San Andreas Landing)

**Metric**: 14-day running average of daily mean EC (µS/cm)  
**D-1641 reference**: Table 2-4, C-4  
**Applicable period**: April 1 – August 15 (off-season values = 0 → displayed as NaN)  
**WYT dependency**: Yes — Dry year transitions to a higher threshold mid-season; Critical Dry year uses a higher threshold for the entire period.

| dms_datastore ID | DSM2 ID | Location | D-1641 Ref |
|-----------------|---------|----------|-----------|
| `sal` | RSAN032 | San Joaquin River at San Andreas Landing | C-4 |

**Standard values** (µS/cm, after × 1000 conversion from mmhos/cm):

| WYT | Apr 1 – transition date | Transition date | Transition – Aug 15 |
|-----|------------------------|-----------------|----------------------|
| W   | 450 µS/cm | — | 450 µS/cm |
| AN  | 450 µS/cm | — | 450 µS/cm |
| BN  | 450 µS/cm | — | 450 µS/cm |
| D   | 450 µS/cm | Jun 25 | 580 µS/cm |
| C   | 870 µS/cm | — | 870 µS/cm |

In Wet, Above Normal, and Below Normal years the 0.45 mmhos/cm standard holds for the
full April 1 – August 15 period with no mid-season relaxation.  In a Dry year, the
standard relaxes to 0.58 mmhos/cm on June 25.  In a Critical Dry year, a single higher
threshold of 0.87 mmhos/cm applies for the entire April 1 – August 15 period (no initial
0.45 period).

**Current compliance history** (from DWR D-1641 Compliance Reports): All EC objectives
at this location have been met since D-1641 was adopted.

---

## Usage

Pass the bundled WSIHIST file (or a custom one) to `DatastoreUIMgr` at construction:

```python
from dms_datastore_ui.datastore_uimgr import DatastoreUIMgr
from dms_datastore_ui.d1641_standards import get_default_wyt_file

mgr = DatastoreUIMgr(
    dir="/path/to/repo",
    wyt_file=get_default_wyt_file(),   # bundled WSIHIST 1906–present
)
```

The D-1641 standard time series are then available as catalog entries alongside the
observed station data.  Selecting an observed EC (or chloride) series and the
corresponding D-1641 standard in the same plot enables direct compliance comparison.

### Updating the Bundled WSIHIST File

The bundled `dms_datastore_ui/data/wsihist.txt` covers water years 1906–2025.  It should
be refreshed each fall (after the new water year completes in September) by re-downloading
from CDEC:

```
http://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST
```

The current water year's standard is always kept up to date automatically via a
monthly-cached live fetch from the CDEC WSI report
(`http://cdec.water.ca.gov/reportapp/javareports?name=WSI`), so manual refresh is only
needed once per year.

---

## Data Sources

| Source | URL |
|--------|-----|
| SWRCB Revised Decision 1641 (1999) | https://www.waterboards.ca.gov/waterrights/water_issues/programs/bay_delta/decision_1641/ |
| CDEC WSIHIST (historical WYTs) | http://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST |
| CDEC WSI (current year forecast) | http://cdec.water.ca.gov/reportapp/javareports?name=WSI |
| D-1641 lookup CSVs | `dsm2_calsim_analysis/data/info/` in [dsm2-calsim-analysis](https://github.com/CADWRDeltaModeling/dsm2-calsim-analysis) |
| Observed station registry | `dms_datastore/config_data/station_dbase.csv` |
