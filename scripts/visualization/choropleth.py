import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --------------------------------------------------
# Load data
# --------------------------------------------------

df_namus = pd.read_csv(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\export\mp_term.csv'
)

gdf_2024 = gpd.read_file(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\source\shape files\2024\counties\tl_2024_us_county.shp'
)

gdf_2024['GEOID'] = gdf_2024['GEOID'].astype(str)
gdf_2024['STATEFP'] = gdf_2024['STATEFP'].astype(str)

# --------------------------------------------------
# Keep only continental US (exclude AK, HI, PR)
# --------------------------------------------------

conus_states = {'02', '15', '72'}  # Alaska, Hawaii, Puerto Rico
gdf_2024 = gdf_2024[~gdf_2024['STATEFP'].isin(conus_states)].copy()

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
cbar.set_label('$log_{10}$(Cumulative Missing Person Cases) (1969–2024)', fontsize=12)

ax.set_title(
    "$log_{10}$(Cumulative NamUs Missing Person Cases) by County (Continental U.S., 1969–2024)",
    fontsize=24,
    fontweight='bold'
)

ax.axis('off')

plt.tight_layout()
plt.savefig(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\plots\demographics\[1969-2024]_mp_county_choropleth.png',
    dpi=1200,
    bbox_inches='tight'
)
plt.show()