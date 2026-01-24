import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --------------------------------------------------
# Load data
# --------------------------------------------------

df_namus = pd.read_csv(
    r'export/mp_term.csv'
)

gdf_2024 = gpd.read_file(
    r'source/shape files/2024/counties/tl_2024_us_county.shp'
)

gdf_states_2024 = gpd.read_file(
    r'source/shape files/2024/states/tl_2024_us_state.shp'
)


df_namus['DisappearanceDate'] = pd.to_datetime(df_namus['DisappearanceDate'])

df_namus = df_namus[
    (df_namus['DisappearanceDate'] > pd.to_datetime('2009-12-31')) &
    (df_namus['DisappearanceDate'] < pd.to_datetime('2025-01-01'))
]

gdf_2024['GEOID'] = gdf_2024['GEOID'].astype(str)
gdf_2024['STATEFP'] = gdf_2024['STATEFP'].astype(str)
gdf_states_2024['GEOID'] = gdf_states_2024['GEOID'].astype(str)
gdf_states_2024['STATEFP'] = gdf_states_2024['STATEFP'].astype(str)

# --------------------------------------------------
# Keep only continental US (exclude AK, HI, PR)
# --------------------------------------------------

conus_states = {'02', '15', '72'}  # Alaska, Hawaii, Puerto Rico
gdf_2024 = gdf_2024[~gdf_2024['STATEFP'].isin(conus_states)].copy()
gdf_states_2024 = gdf_states_2024[~gdf_states_2024['STATEFP'].isin(conus_states)].copy()

# --------------------------------------------------
# Prepare case counts
# --------------------------------------------------

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

# --------------------------------------------------
# Log scaling (exclude zeros)
# --------------------------------------------------

vmin = gdf.loc[gdf['case_count'] > 0, 'case_count'].min()
vmax = gdf['case_count'].max()

norm = LogNorm(vmin=vmin, vmax=vmax)

# --------------------------------------------------
# Plot choropleth
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 8))

# Plot map
gdf.plot(
    column='case_count',
    ax=ax,
    cmap='viridis',
    linewidth=0.1,
    edgecolor='gray',
    norm=norm,
    legend=False  # We'll add a horizontal colorbar manually
)

# Zoom to continental US bounds (approximate)
ax.set_xlim([-125, -66])  # longitude
ax.set_ylim([24, 50])     # latitude

# Add horizontal colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm._A = []  # dummy array for ScalarMappable
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
####################################################################################################################


# --------------------------------------------------
# Prepare case counts (STATE-LEVEL)
# --------------------------------------------------

df_namus['FIPS'] = (
    df_namus['FIPS']
    .astype(str)
    .str.zfill(5)
)

# Extract state FIPS (first 2 digits)
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

# --------------------------------------------------
# Log scaling (exclude zeros)
# --------------------------------------------------

vmin = gdf.loc[gdf['case_count'] > 0, 'case_count'].min()
vmax = gdf['case_count'].max()

norm = LogNorm(vmin=vmin, vmax=vmax)

# --------------------------------------------------
# Plot choropleth
# --------------------------------------------------

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

# Zoom to continental US bounds
ax.set_xlim([-125, -66])
ax.set_ylim([24, 50])

# Horizontal colorbar
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
