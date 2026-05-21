"""
U.S. county-year population panel construction.

Builds export/population.csv, a tidy (FIPS, Year, Population, name, state,
source) panel covering every U.S. county from 1969 through 2024. The panel
draws on three sources, stitched together in priority order:

    1. SEER age-adjusted historical county population estimates (1969-2022).
       This is the spine of the panel. We use SEER rather than the raw Census
       because it provides a consistent county-year format with stable FIPS
       semantics over the full window.
    2. Census Bureau county population estimates (2023, 2024). Used to extend
       the SEER panel forward to the most recent vintage.
    3. NHGIS historical county shapefiles (1900-2024) used as a *fallback*
       for unresolved FIPS codes. SEER occasionally carries legacy FIPS that
       no longer exist in the modern Census `cencounts` table; for those we
       walk backward through the shapefiles 2024 -> 1900 and pick up the
       county name from whichever vintage first matches.

The output also carries a `source` column noting where each row's county name
came from ("table" for the NBER cencounts table or "shapefile_YYYY" for the
NHGIS fallback) so downstream consumers can audit the provenance.

Run as:
    python scripts/us/data/cleaning/population_cleaning.py
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import csv


# ============================================================
# STEP 0: parse the SEER fixed-width text dump into a tidy CSV.
# ============================================================

def clean_and_export_population_data(input_file, output_csv_file):
    """Parse the SEER fixed-width population file into a (Year, FIPS, Population) CSV.

    SEER distributes population estimates as a fixed-width text file where:
      - columns 0:4  hold the year
      - columns 6:11 hold the 5-digit FIPS code
      - columns 18+  hold the population count (integer, right-padded)

    Lines shorter than 18 characters or whose population field isn't a pure
    digit string are silently skipped (typically header/footer lines that
    sneak past the SEER format guarantee).
    """
    cleaned_data = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) < 18:
                continue
            try:
                cleaned_data.append({
                    'Year': int(line[0:4]),
                    'FIPS': line[6:11],
                    'Population': int(line[18:]) if line[18:].isdigit() else None
                })
            except Exception:
                # Tolerate any malformed line silently. SEER's format is
                # stable in practice but we don't want a single bad line to
                # bring down the whole pipeline.
                continue

    if cleaned_data:
        with open(output_csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cleaned_data[0].keys())
            writer.writeheader()
            writer.writerows(cleaned_data)


# Parse the SEER input into export/us_pop_by_decade.csv (the filename is a
# historical artifact -- the file is actually year-resolved, not decadal).
clean_and_export_population_data(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\source\SEER Population Estimates\us_1969_2022.19ages.adjusted.txt',
    r'F:\dsl_CLIMA\projects\submittable\missing persons\export\us_pop_by_decade.csv'
)

# ============================================================
# STEP 1: Load the three population inputs we'll merge.
# ============================================================

# SEER, parsed in Step 0.
df_population = pd.read_csv(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\export\us_pop_by_decade.csv',
    dtype={'Year': int, 'FIPS': str}
)

# NBER's cencounts: a (fips, name) lookup table covering modern county codes.
# Used as the primary source for the human-readable county name.
df_cencount = pd.read_csv(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\source\NBER County Population Estimates\cencounts.csv',
    dtype=str
)

# Census Bureau's 2024 vintage county population estimates, which extends
# SEER's coverage through 2024. encoding='latin1' because the file has
# non-UTF-8 county names (Doña Ana etc.).
df_pop_est = pd.read_csv(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\source\2024 County Population Est\co-est2024-alldata.csv',
    dtype={'STATE': str, 'COUNTY': str},
    encoding='latin1'
)

# ============================================================
# STEP 2: Aggregate SEER and append the 2023-2024 rows from Census.
# SEER counts are broken out by age/sex/race bins; the aggregation here
# collapses those bins into a single per-(FIPS, Year) total.
# ============================================================

df_population = (
    df_population
    .groupby(['FIPS', 'Year'], as_index=False)
    .agg({'Population': 'sum'})
)

# Build a 5-digit FIPS from STATE+COUNTY in the Census file, then drop the
# state-aggregate rows (FIPS ending in '000' means "this is the state total").
df_pop_est['FIPS'] = df_pop_est['STATE'] + df_pop_est['COUNTY']
df_pop_est = df_pop_est[~df_pop_est['FIPS'].str.endswith('000')]

df_2023 = df_pop_est[['FIPS', 'POPESTIMATE2023']].rename(
    columns={'POPESTIMATE2023': 'Population'}
)
df_2023['Year'] = 2023

df_2024 = df_pop_est[['FIPS', 'POPESTIMATE2024']].rename(
    columns={'POPESTIMATE2024': 'Population'}
)
df_2024['Year'] = 2024

df_population = pd.concat([df_population, df_2023, df_2024], ignore_index=True)

# ============================================================
# STEP 3: Normalize FIPS.
# Strip any rows matching the pattern ^\d{2}9\d{2}$ -- those are SEER's
# state-pooled "registry" pseudo-FIPS (e.g. 06900 = California aggregate),
# not real counties. Leaving them in would double-count population.
# ============================================================

df_population['FIPS'] = df_population['FIPS'].astype(str).str.zfill(5)
df_population = df_population[
    ~df_population['FIPS'].str.match(r'^\d{2}9\d{2}$')
]

df_cencount['fips'] = df_cencount['fips'].astype(str).str.zfill(5)

# ============================================================
# STEP 4: Apply FIPS corrections to df_cencount.
#
# Why we re-map cencounts rather than SEER: SEER's panel uses the pre-1997
# Dade County FIPS (12025). cencounts uses the post-1997 Miami-Dade code
# (12086). We map cencounts to SEER's spelling so the merge in Step 5 finds
# a match. The SEER side keeps its historical FIPS continuity.
# ============================================================

fips_corrections = {
    '12025': '12086',  # Miami-Dade -> Dade
}

df_cencount['fips_corrected'] = df_cencount['fips'].replace(fips_corrections)

# ============================================================
# STEP 5: Authoritative name lookup (table-first).
# Left-join on cencounts to pick up the human-readable name for every FIPS
# that exists in the modern table. Rows that don't match here will be filled
# in by the shapefile fallback in Step 6.
# ============================================================

df_merged = df_population.merge(
    df_cencount[['fips_corrected', 'name']],
    left_on='FIPS',
    right_on='fips_corrected',
    how='left'
).drop(columns='fips_corrected')

# Provenance: 'table' means the name came from cencounts. NaN means the FIPS
# didn't match and we'll try the shapefile fallback next.
df_merged['source'] = np.where(df_merged['name'].notna(), 'table', None)

# ============================================================
# STEP 6: Shapefile fallback for FIPS the table can't resolve.
#
# Some SEER FIPS are legacy codes that have been retired from modern Census
# tables. For those we walk backward through NHGIS historical shapefiles
# (2024 -> 1900) and grab the county name from whichever vintage first
# matches. This way a retired FIPS gets the historically-correct name
# instead of being silently dropped.
# ============================================================

df_nan = df_merged[df_merged['name'].isna()].copy()
df_nan['name_filled'] = None
df_nan['source'] = None


def build_fips_map(gdf):
    """Build a {FIPS: county_name} dict from a NHGIS shapefile.

    NHGIS files use different column conventions across vintages; this helper
    tries the variants in order (GEOID, then STATEFP+COUNTYFP, then
    CNTY_FIPS) and pulls the name from whichever name-column exists
    (NAMELSAD, NAME, COUNTYNAME). Raises if neither side is recognizable.
    """
    if 'GEOID' in gdf.columns:
        fips_col = 'GEOID'
    elif 'STATEFP' in gdf.columns and 'COUNTYFP' in gdf.columns:
        gdf['FIPS'] = gdf['STATEFP'].astype(str).str.zfill(2) + gdf['COUNTYFP'].astype(str).str.zfill(3)
        fips_col = 'FIPS'
    elif 'CNTY_FIPS' in gdf.columns:
        gdf['FIPS'] = gdf['CNTY_FIPS'].astype(str).str.zfill(5)
        fips_col = 'FIPS'
    else:
        raise ValueError("No recognizable FIPS columns found.")

    for name_col in ['NAMELSAD', 'NAME', 'COUNTYNAME']:
        if name_col in gdf.columns:
            gdf[name_col] = gdf[name_col].astype(str).str.strip()
            return dict(zip(gdf[fips_col].astype(str).str.zfill(5), gdf[name_col]))

    raise ValueError("No recognizable county name column found.")


# Shapefile paths, walked from newest to oldest. The first match wins, so
# more-recent name-spellings take precedence when a FIPS appears in
# multiple vintages.
county_shape_files = {
    2024: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2024\counties\tl_2024_us_county.shp',
    2023: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2023\US_county_2023.shp',
    2022: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2022\US_county_2022.shp',
    2010: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2010\US_county_2010.shp',
    2000: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2000\US_county_2000.shp',
    1990: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1990\US_county_1990.shp',
    1980: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1980\US_county_1980.shp',
    1970: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1970\US_county_1970_conflated.shp',
    1960: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1960\US_county_1960_conflated.shp',
    1950: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1950\US_county_1950_conflated.shp',
    1940: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1940\US_county_1940_conflated.shp',
    1930: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1930\US_county_1930_conflated.shp',
    1920: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1920\US_county_1920_conflated.shp',
    1910: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1910\US_county_1910_conflated.shp',
    1900: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1900\US_county_1900_conflated.shp'
}
# Sub-county subdivisions (minor civil divisions / townships) — used for
# states like New England where the modern county isn't the right unit.
subdivision_shape_files = {
    2023: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2023\subdivisions\US_cty_sub_2023.shp',
    2022: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2022\subdivisons\US_cty_sub_2022.shp',
    2010: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2010\subdivisions\US_cty_sub_2010.shp',
    2000: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2000\subdivisions\US_cty_sub_2000.shp',
    1990: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1990\subdivisions\US_cty_sub_1990.shp',
    1980: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1980\subdivisions\US_mcd_1980.shp'
}

# Walk the county shapefiles first. Only rows whose name_filled is still
# NaN get updated each iteration -- a later (older) vintage can't overwrite
# a name we already resolved from a more-recent vintage.
for year, path in county_shape_files.items():
    gdf = gpd.read_file(path)
    fips_map = build_fips_map(gdf)

    mask = df_nan['name_filled'].isna()
    matches = df_nan.loc[mask, 'FIPS'].map(fips_map)

    df_nan.loc[mask, 'name_filled'] = matches
    df_nan.loc[mask & matches.notna(), 'source'] = f'shapefile_{year}'

# Then walk the subdivision shapefiles for whatever's still unresolved.
for year, path in subdivision_shape_files.items():
    gdf = gpd.read_file(path)
    fips_map = build_fips_map(gdf)

    mask = df_nan['name_filled'].isna()
    matches = df_nan.loc[mask, 'FIPS'].map(fips_map)

    df_nan.loc[mask, 'name_filled'] = matches
    df_nan.loc[mask & matches.notna(), 'source'] = f'shapefile_{year}'

# Push the shapefile-resolved names back into the main frame.
df_nan['name'] = df_nan['name_filled']
df_merged.update(df_nan[['FIPS', 'name', 'source']])

# ============================================================
# Derive a State column from FIPS prefix.
# We first try to extract a leading "XX " state-abbreviation prefix from the
# name (some shapefile vintages include this), but the authoritative source
# is the first two FIPS digits mapped via state_fips_to_abbr below.
# ============================================================

df_merged['State'] = df_merged['name'].str.extract(r'^([A-Z]{2})\s+')

us_state_abbrev = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts',
    'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana',
    'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
    'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
    'DC': 'District of Columbia'
}

# State FIPS (first two digits of any county FIPS) -> two-letter abbreviation.
state_fips_to_abbr = {
    '01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT','10':'DE',
    '11':'DC','12':'FL','13':'GA','15':'HI','16':'ID','17':'IL','18':'IN','19':'IA',
    '20':'KS','21':'KY','22':'LA','23':'ME','24':'MD','25':'MA','26':'MI','27':'MN',
    '28':'MS','29':'MO','30':'MT','31':'NE','32':'NV','33':'NH','34':'NJ','35':'NM',
    '36':'NY','37':'NC','38':'ND','39':'OH','40':'OK','41':'OR','42':'PA','44':'RI',
    '45':'SC','46':'SD','47':'TN','48':'TX','49':'UT','50':'VT','51':'VA','53':'WA',
    '54':'WV','55':'WI','56':'WY'
}

df_merged['state_abbr'] = df_merged['FIPS'].str[:2].map(state_fips_to_abbr)
df_merged['State'] = df_merged['state_abbr'].map(us_state_abbrev)
df_merged = df_merged.drop(columns=['state_abbr'])

# Strip the leading "XX " from county names if present.
df_merged['name'] = df_merged['name'].str.replace(r'^[A-Z]{2}\s+', '', regex=True)

# Manual name corrections to reconcile spelling differences between the SEER
# vintage and the BLS crosswalk that consumes this file in the next step.
df_merged.loc[(df_merged['name'] == 'Dade County') & (df_merged['State'] == 'Florida'), 'name'] = 'Miami-Dade County'
df_merged.loc[(df_merged['name'] == 'La Salle County') & (df_merged['State'] == 'Illinois'), 'name'] = 'Lasalle County'
df_merged.loc[(df_merged['name'] == 'DeBaca County') & (df_merged['State'] == 'New Mexico'), 'name'] = 'De Baca County'
df_merged.loc[(df_merged['name'] == 'St. John the Baptist Par.') & (df_merged['State'] == 'Louisiana'), 'name'] = 'St. John the Baptist Parish'
df_merged.loc[(df_merged['name'] == 'Dona Ana County') & (df_merged['State'] == 'New Mexico'), 'name'] = 'DOÑA ANA COUNTY'

# ============================================================
# STEP 7: write the final county-year population panel.
# ============================================================

df_merged.to_csv(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\export\population.csv',
    index=False
)

# print(df_merged['source'].value_counts(dropna=False))  # provenance breakdown
print(df_merged.isna().sum())
