"""
County- and state-level choropleths of cumulative NamUs cases.

Produces two log-color-scaled maps of the continental U.S. for the 2010-2024
window:

    plots/demographics/[2010-2024]/[2010-2024]_mp_county_choropleth.png
        County-level cumulative case counts, joined to the 2024 Census
        county shapefile by GEOID.

    plots/demographics/[2010-2024]/[2010-2024]_mp_state_choropleth.png
        State-level cumulative case counts, joined to the 2024 Census
        state shapefile by STATEFP (the first two digits of every FIPS).

Log color scale because the case-count distribution is heavy-tailed -- a
linear scale would push every county that isn't LA into the same shade.
Alaska, Hawaii, and Puerto Rico (STATEFP 02 / 15 / 72) are excluded so the
continental-US bounding box stays tight.

Run as:
    python scripts/us/visualization/choropleth.py
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Load cases + the 2024 Census shapefiles for counties and states.
# ---------------------------------------------------------------------------
df_namus = pd.read_csv(
    r'export/mp_term.csv'
)

gdf_2024 = gpd.read_file(
    r'source/shape files/2024/counties/tl_2024_us_county.shp'
)

gdf_states_2024 = gpd.read_file(
    r'source/shape files/2024/states/tl_2024_us_state.shp'
)

# Restrict to the 2010-2024 window for the modern-era view.
df_namus['DisappearanceDate'] = pd.to_datetime(df_namus['DisappearanceDate'])
df_namus = df_namus[
    (df_namus['DisappearanceDate'] > pd.to_datetime('2009-12-31')) &
    (df_namus['DisappearanceDate'] < pd.to_datetime('2025-01-01'))
]

# Stringify the join keys so the merge below doesn't trip over int/str
# mismatches between the shapefile and our CSV.
gdf_2024['GEOID'] = gdf_2024['GEOID'].astype(str)
gdf_2024['STATEFP'] = gdf_2024['STATEFP'].astype(str)
gdf_states_2024['GEOID'] = gdf_states_2024['GEOID'].astype(str)
gdf_states_2024['STATEFP'] = gdf_states_2024['STATEFP'].astype(str)

# ---------------------------------------------------------------------------
# Drop AK, HI, PR (STATEFP 02, 15, 72). AK and HI distort the bounding box
# and PR has no SEER population panel for normalization.
# ---------------------------------------------------------------------------
conus_states = {'02', '15', '72'}
gdf_2024 = gdf_2024[~gdf_2024['STATEFP'].isin(conus_states)].copy()
gdf_states_2024 = gdf_states_2024[~gdf_states_2024['STATEFP'].isin(conus_states)].copy()

# ---------------------------------------------------------------------------
# Build per-county case counts and merge into the county shapefile.
# ---------------------------------------------------------------------------
df_namus['FIPS'] = (
    df_namus['FIPS']
    .astype(str)
    .str.zfill(5)
)

county_counts = (
    df_namus
    .groupby('FIPS')
    .size()
    .reset_index(name='case_count')
)

gdf = gdf_2024.merge(
    county_counts,
    left_on='GEOID',
    right_on='FIPS',
    how='left'
)
gdf['case_count'] = gdf['case_count'].fillna(0)

# ---------------------------------------------------------------------------
# Log color norm. vmin comes from the smallest *nonzero* case count; without
# this, zero-case counties would be -inf in log space.
# ---------------------------------------------------------------------------
vmin = gdf.loc[gdf['case_count'] > 0, 'case_count'].min()
vmax = gdf['case_count'].max()
norm = LogNorm(vmin=vmin, vmax=vmax)

# ---------------------------------------------------------------------------
# Render the county-level choropleth.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

gdf.plot(
    column='case_count',
    ax=ax,
    cmap='viridis',
    linewidth=0.1,
    edgecolor='gray',
    norm=norm,
    legend=False  # horizontal colorbar added manually below
)

# Zoom to the continental-US bounding box.
ax.set_xlim([-125, -66])
ax.set_ylim([24, 50])

# Horizontal colorbar at the bottom of the map.
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm._A = []  # required for ScalarMappable to work as a colorbar source
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05)
cbar.set_label('$log_{10}$(Cumulative Missing Person Cases) (2010–2024)', fontsize=12)

ax.set_title(
    "Cumulative NamUs Missing Person Cases by County (Continental U.S., 2010-2024)",
    fontsize=24,
    fontweight='bold'
)
ax.axis('off')

plt.tight_layout()
plt.savefig(
    r'plots/demographics/[2010-2024]/[2010-2024]_mp_county_choropleth.png',
    dpi=1200,
    bbox_inches='tight'
)
plt.show()


# ---------------------------------------------------------------------------
# State-level view.
#
# Same data, aggregated to state FIPS (first two digits) and joined into the
# 2024 state shapefile. Mostly a sanity check that the county map isn't
# being driven by FIPS-resolution artifacts.
# ---------------------------------------------------------------------------
df_namus['FIPS'] = (
    df_namus['FIPS']
    .astype(str)
    .str.zfill(5)
)
df_namus['STATEFP'] = df_namus['FIPS'].str[:2]

state_counts = (
    df_namus
    .groupby('STATEFP')
    .size()
    .reset_index(name='case_count')
)

gdf = gdf_states_2024.merge(
    state_counts,
    on='STATEFP',
    how='left'
)
gdf['case_count'] = gdf['case_count'].fillna(0)

vmin = gdf.loc[gdf['case_count'] > 0, 'case_count'].min()
vmax = gdf['case_count'].max()
norm = LogNorm(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(14, 8))

gdf.plot(
    column='case_count',
    ax=ax,
    cmap='viridis',
    linewidth=0.6,
    edgecolor='gray',
    norm=norm,
    legend=False
)

ax.set_xlim([-125, -66])
ax.set_ylim([24, 50])

sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm._A = []
cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation='horizontal',
    fraction=0.05,
    pad=0.05
)
cbar.set_label(
    '$log_{10}$(Cumulative Missing Person Cases) (2010–2024)',
    fontsize=12
)

ax.set_title(
    "Cumulative NamUs Missing Person Cases by State "
    "(Continental U.S., 2010–2024)",
    fontsize=24,
    fontweight='bold'
)
ax.axis('off')

plt.tight_layout()
plt.savefig(
    r'plots/demographics/[2010-2024]/[2010-2024]_mp_state_choropleth.png',
    dpi=1200,
    bbox_inches='tight'
)
plt.show()
