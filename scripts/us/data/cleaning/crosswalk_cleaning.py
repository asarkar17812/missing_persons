"""
Crosswalk-driven merge of NamUs cases and the county-year population panel.

This is the final stage of the U.S. cleaning pipeline. It does three things:

    1. Loads the BLS QCEW County-MSA-CSA crosswalk in three historical
       vintages (Dec. 2003, Feb. 2013, Jul. 2023) and applies the vintage
       appropriate to each year so that historical CBSA boundaries are
       respected per row.
    2. Merges every NamUs case row to the population frame on
       (Year, County_norm, State_norm), pulling in the County / MSA / CSA /
       State populations the row needs for downstream scaling regressions.
    3. Emits two CSVs:
         - export/mp_term.csv   (one row per case, joined to population)
         - export/pop_term.csv  (the population panel with MSA/CSA/State
                                 aggregates attached)

A few details that aren't obvious from the code alone:

    - "Bad counties" (rows whose County field is MISSING / UNKNOWN /
      CENSORED) fall through a fallback path that tries to recover an MSA
      from the city -> MSA-title match. If the recovered MSA contains
      exactly one county, we backfill that county too. This is the only
      place in the pipeline where city information is used for spatial
      reconciliation rather than display.
    - CBSA Type and CSA Type get extracted from the title strings via a
      regex on the last whitespace-delimited word (e.g. "Boston-Cambridge-
      Newton, MA-NH Metropolitan Statistical Area" -> "Area"). They get
      cleaned up to {MSA, MicroSA, CSA} by downstream visualization code.

Run as:
    python scripts/us/data/cleaning/crosswalk_cleaning.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd


# ---------------------------------------------------------------------------
# Load the inputs produced by the upstream cleaning scripts.
# ---------------------------------------------------------------------------
df_population = pd.read_csv(
    r'/Users/ayushsarkar/missing_persons/missing_persons/export/population.csv',
    dtype={'FIPS': str}
)
df_namus = pd.read_csv(
    r'/Users/ayushsarkar/missing_persons/missing_persons/export/namus_cases.csv'
)
crosswalk_file = r'/Users/ayushsarkar/missing_persons/missing_persons/source/crosswalk/qcew-county-msa-csa-crosswalk.xlsx'

# Tokens we treat as unusable when checking the County field.
bad_values = {'MISSING', 'UNKNOWN', 'CENSORED'}


def clean_crosswalk(df_cw):
    """Pre-process a raw QCEW crosswalk sheet.

    Splits combined "County, ST" / "MSA, ST" titles into separate name and
    state-abbreviation columns, zero-pads the 5-digit County and MSA codes,
    and uppercases the name fields so downstream merges are case-insensitive.

    The QCEW workbook ships one sheet per vintage; this helper is called
    once per vintage.
    """
    df_cw[['County Name', 'State Full']] = df_cw['County Title'].str.upper().str.split(',', n=1, expand=True)
    df_cw['County Name'] = df_cw['County Name'].str.strip()
    df_cw['State Full'] = df_cw['State Full'].str.strip()

    msa_split = df_cw['MSA Title'].str.upper().str.split(',', n=1, expand=True)
    df_cw['MSA Name'] = msa_split[0].str.strip()
    df_cw['MSA State Abbr'] = msa_split[1].str.strip().str.slice(0, 2)

    df_cw['County Code'] = df_cw['County Code'].astype(str).str.zfill(5)
    df_cw['MSA Code'] = df_cw['MSA Code'].astype(str).str.zfill(5)

    return df_cw


def merge_pop_with_crosswalk(df_subset, cw, bad_values={'MISSING', 'UNKNOWN', 'CENSORED'}):
    """Attach CBSA/CSA codes to a population subframe.

    Joins by the 5-digit County FIPS (the crosswalk's `County Code`) so
    every county-year row inherits its parent MSA Code, CSA Code, and the
    associated title strings. The `bad_values` mask is currently unused
    here (population rows always have a usable county) but is left in for
    symmetry with `merge_cases_with_crosswalk` below.
    """
    df = df_subset.copy()

    # Normalize the merge keys to upper/stripped form.
    df['County_norm'] = df['County'].astype(str).str.upper().str.strip()
    df['State_norm'] = df['State'].astype(str).str.upper().str.strip()

    cw['County_norm'] = cw['County Title'].astype(str).str.upper().str.strip()
    cw['MSA_Title_norm'] = cw['MSA Title'].astype(str).str.upper().str.split(',', n=1).str[0].str.strip()
    cw['State_abbr'] = cw['MSA Title'].astype(str).str.extract(r',\s*([^\s]+)', expand=False)

    # Map QCEW's two-letter state abbreviations back to the full state names
    # the upstream population panel uses.
    us_state_abbrev = {
        'AL': 'Alabama','AK': 'Alaska','AZ': 'Arizona','AR': 'Arkansas','CA': 'California',
        'CO': 'Colorado','CT': 'Connecticut','DE': 'Delaware','FL': 'Florida','GA': 'Georgia',
        'HI': 'Hawaii','ID': 'Idaho','IL': 'Illinois','IN': 'Indiana','IA': 'Iowa','KS': 'Kansas',
        'KY': 'Kentucky','LA': 'Louisiana','ME': 'Maine','MD': 'Maryland','MA': 'Massachusetts',
        'MI': 'Michigan','MN': 'Minnesota','MS': 'Mississippi','MO': 'Missouri','MT': 'Montana',
        'NE': 'Nebraska','NV': 'Nevada','NH': 'New Hampshire','NJ': 'New Jersey','NM': 'New Mexico',
        'NY': 'New York','NC': 'North Carolina','ND': 'North Dakota','OH': 'Ohio','OK': 'Oklahoma',
        'OR': 'Oregon','PA': 'Pennsylvania','RI': 'Rhode Island','SC': 'South Carolina',
        'SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas','UT': 'Utah','VT': 'Vermont',
        'VA': 'Virginia','WA': 'Washington','WV': 'West Virginia','WI': 'Wisconsin','WY': 'Wyoming',
        'D.C.': 'District of Columbia'
    }
    cw['State_full'] = cw['State_abbr'].map(us_state_abbrev)

    # Population rows always have a usable County, but we apply the mask
    # anyway so the function signature matches the cases version.
    good_mask = df['County'].notna() & (~df['County'].isin(bad_values))
    df_good = df[good_mask].copy()

    df_good = df_good.merge(
        cw[['County Code','County Title','MSA Code','CSA Code','MSA Title','CSA Title']],
        left_on='FIPS',
        right_on='County Code',
        how='left'
    ).drop(columns=['County Code'], errors='ignore')

    return df_good


def merge_cases_with_crosswalk(df_subset, cw, bad_values={'MISSING', 'UNKNOWN', 'CENSORED'}):
    """Attach CBSA/CSA codes to a NamUs-cases subframe.

    Two-stage merge:
      - Rows with a usable County are joined on the 5-digit FIPS.
      - Rows with a "bad" County (MISSING / UNKNOWN / CENSORED) fall through
        to a city -> MSA-title match; if the recovered MSA contains exactly
        one county, the county and its FIPS are backfilled too.

    The recombine at the end concatenates the two paths back together.
    """
    df = df_subset.copy()

    df['County_norm'] = df['County'].astype(str).str.upper().str.strip()
    df['City_norm'] = df['City'].astype(str).str.upper().str.strip()
    df['State_norm'] = df['State'].astype(str).str.upper().str.strip()

    cw['County_norm'] = cw['County Title'].astype(str).str.upper().str.strip()
    cw['MSA_Title_norm'] = cw['MSA Title'].astype(str).str.upper().str.split(',', n=1).str[0].str.strip()
    cw['State_abbr'] = cw['MSA Title'].astype(str).str.extract(r',\s*([^\s]+)', expand=False)

    us_state_abbrev = {
        'AL': 'Alabama','AK': 'Alaska','AZ': 'Arizona','AR': 'Arkansas','CA': 'California',
        'CO': 'Colorado','CT': 'Connecticut','DE': 'Delaware','FL': 'Florida','GA': 'Georgia',
        'HI': 'Hawaii','ID': 'Idaho','IL': 'Illinois','IN': 'Indiana','IA': 'Iowa','KS': 'Kansas',
        'KY': 'Kentucky','LA': 'Louisiana','ME': 'Maine','MD': 'Maryland','MA': 'Massachusetts',
        'MI': 'Michigan','MN': 'Minnesota','MS': 'Mississippi','MO': 'Missouri','MT': 'Montana',
        'NE': 'Nebraska','NV': 'Nevada','NH': 'New Hampshire','NJ': 'New Jersey','NM': 'New Mexico',
        'NY': 'New York','NC': 'North Carolina','ND': 'North Dakota','OH': 'Ohio','OK': 'Oklahoma',
        'OR': 'Oregon','PA': 'Pennsylvania','RI': 'Rhode Island','SC': 'South Carolina',
        'SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas','UT': 'Utah','VT': 'Vermont',
        'VA': 'Virginia','WA': 'Washington','WV': 'West Virginia','WI': 'Wisconsin','WY': 'Wyoming',
        'D.C.': 'District of Columbia'
    }
    cw['State_full'] = cw['State_abbr'].map(us_state_abbrev)

    # Split good vs bad on the County field.
    good_mask = df['County'].notna() & (~df['County'].isin(bad_values))
    df_good = df[good_mask].copy()
    df_bad = df[~good_mask].copy()

    # Primary path: merge good rows by FIPS.
    df_good = df_good.merge(
        cw[['County Code','County Title','MSA Code','CSA Code','MSA Title','CSA Title']],
        left_on='FIPS',
        right_on='County Code',
        how='left'
    ).drop(columns=['County Code'], errors='ignore')

    # Fallback path: try to recover an MSA from the city + state match.
    df_bad = df_bad.merge(
        cw[['MSA_Title_norm','State_full','MSA Code','CSA Code','MSA Title','CSA Title']],
        left_on=['City_norm','State_norm'],
        right_on=['MSA_Title_norm','State_full'],
        how='left'
    ).drop(columns=['MSA_Title_norm','State_full'], errors='ignore')

    # For MSAs that map to exactly one county, backfill the County and FIPS
    # too -- otherwise we leave them blank because we can't pick one
    # arbitrarily.
    msa_single = (
        cw.groupby('MSA Code', as_index=False)
        .agg({'County Code':'nunique','County Title':'first'})
        .query('`County Code` == 1')
        .rename(columns={'County Title':'Single_County_Title'})
    )
    msa_code_fill = cw.groupby('MSA Code', as_index=False).agg({'County Code':'first'})
    msa_single = msa_single.drop(columns=['County Code']).merge(msa_code_fill, on='MSA Code')

    df_bad = df_bad.merge(msa_single, on='MSA Code', how='left')
    df_bad['FIPS'] = df_bad['FIPS'].fillna(df_bad['County Code'])
    df_bad['County'] = df_bad['County'].fillna(df_bad['Single_County_Title'])
    df_bad.drop(columns=['County Code','Single_County_Title'], inplace=True)

    # Recombine the two paths into a single frame.
    df_merged = pd.concat([df_good, df_bad], ignore_index=True)
    df_merged.drop(columns=['County_norm','City_norm','State_norm'], errors='ignore', inplace=True)

    return df_merged


def summarize_population_by_msa_all_years(df):
    """Sum county populations into MSA totals, per (Year, MSA Code)."""
    return (
        df.groupby(['Year', 'MSA Code'], as_index=False)
          .agg(MSA_pop=('Population', 'sum'))
          .sort_values(['Year', 'MSA Code'])
          .reset_index(drop=True)
    )


def summarize_population_by_csa_all_years(df):
    """Sum county populations into CSA totals, per (Year, CSA Code)."""
    return (
        df.groupby(['Year', 'CSA Code'], as_index=False)
          .agg(CSA_pop=('Population', 'sum'))
          .sort_values(['Year', 'CSA Code'])
          .reset_index(drop=True)
    )


def summarize_population_by_state_all_years(df):
    """Sum county populations into state totals, per (Year, State)."""
    return (
        df.groupby(['Year', 'State'], as_index=False)
          .agg(State_pop=('Population', 'sum'))
          .sort_values(['Year', 'State'])
          .reset_index(drop=True)
    )


def simplify_titles(df):
    """Extract CBSA Type / CSA Type and strip the trailing state suffix.

    The MSA/CSA Title fields ship as e.g. "Boston-Cambridge-Newton, MA-NH
    Metropolitan Statistical Area". We pull off the last whitespace word
    (the *Type*) into its own column and shorten the title to the
    leading place-name part.
    """
    if 'MSA Title' in df.columns:
        df['CBSA Type'] = df['MSA Title'].astype(str).str.extract(r'(\w+)$')[0].replace('nan', np.nan)
        df['MSA Title'] = df['MSA Title'].astype(str).str.split(',', n=1).str[0].str.strip()
    if 'CSA Title' in df.columns:
        df['CSA Type'] = df['CSA Title'].astype(str).str.extract(r'(\w+)$')[0].replace('nan', np.nan)
        df['CSA Title'] = df['CSA Title'].astype(str).str.split(',', n=1).str[0].str.strip()
    return df


# ---------------------------------------------------------------------------
# Pre-merge normalization: upper/stripped form on the join keys for both
# frames so the merge is case-insensitive.
# ---------------------------------------------------------------------------
df_namus['County_norm'] = df_namus['County'].str.upper().str.strip()
df_namus['State_norm'] = df_namus['State'].str.upper().str.strip()
df_population['County_norm'] = df_population['name'].str.upper().str.strip()
df_population['State_norm'] = df_population['State'].str.upper().str.strip()
df_namus['Year'] = df_namus['Year'].astype(int)
df_population['Year'] = df_population['Year'].astype(int)

# Deduplicate the population frame: occasionally SEER + Census produce two
# rows for the same (Year, State, County) under slightly different FIPS
# spellings. We keep one to avoid row-multiplication in the merge.
df_population = df_population.drop_duplicates(subset=['Year', 'State_norm', 'County_norm'])

# ---------------------------------------------------------------------------
# Load and clean all three crosswalk vintages.
# ---------------------------------------------------------------------------
cw_2003 = clean_crosswalk(pd.read_excel(crosswalk_file, sheet_name='Dec. 2003 Crosswalk', dtype=str))
cw_2013 = clean_crosswalk(pd.read_excel(crosswalk_file, sheet_name='Feb. 2013 Crosswalk', dtype=str))
cw_2023 = clean_crosswalk(pd.read_excel(crosswalk_file, sheet_name='Jul. 2023 Crosswalk', dtype=str))

# ---------------------------------------------------------------------------
# Split the population frame into three slabs and apply the crosswalk
# vintage appropriate to each. This is what makes the per-year CBSA / CSA
# aggregates historically faithful.
# ---------------------------------------------------------------------------
df_population['County'] = df_population['name'].copy()
df_pop_2003 = df_population[df_population['Year'] <= 2003]
df_pop_2013 = df_population[(df_population['Year'] > 2003) & (df_population['Year'] < 2013)]
df_pop_2023 = df_population[df_population['Year'] >= 2013]

df_pop_final = pd.concat([
    merge_pop_with_crosswalk(df_pop_2003, cw_2003),
    merge_pop_with_crosswalk(df_pop_2013, cw_2013),
    merge_pop_with_crosswalk(df_pop_2023, cw_2023)
], ignore_index=True)

# Build MSA / CSA / State per-year totals and merge them back on.
df_cbsa = summarize_population_by_msa_all_years(df_pop_final)
df_csa = summarize_population_by_csa_all_years(df_pop_final)
df_state = summarize_population_by_state_all_years(df_pop_final)

df_pop_final = (
    df_pop_final
    .merge(df_cbsa, on=['Year', 'MSA Code'], how='left')
    .merge(df_csa, on=['Year', 'CSA Code'], how='left')
    .merge(df_state, on=['Year', 'State'], how='left')
    .rename(columns={'Population': 'County_pop'})
).copy()

df_pop_final = simplify_titles(df_pop_final)

# ---------------------------------------------------------------------------
# Merge the population frame into the NamUs cases frame.
# Every case row gets County_pop, MSA_pop, CSA_pop, and State_pop for its
# disappearance year, plus the CBSA / CSA Code / Title / Type metadata.
# ---------------------------------------------------------------------------
df_namus = df_namus.merge(
    df_pop_final[['FIPS', 'Year', 'County_pop', 'name', 'source', 'State', 'MSA Code', 'CSA Code', 'MSA Title', 'CSA Title', 'MSA_pop', 'CSA_pop', 'State_pop', 'CBSA Type', 'CSA Type', 'County_norm', 'State_norm']],
    on=['Year', 'County_norm', 'State_norm'],
    how='left'
).drop_duplicates()

# Keep only the columns downstream visualizations need.
df_namus = df_namus[['CaseID','CurrentMinAge','CurrentMaxAge','Sex','Ethnicity','DisappearanceDate','City','State_x','County','Year','FIPS','County_pop','MSA Code','CSA Code','MSA Title','CSA Title','MSA_pop','CSA_pop', 'State_pop', 'CBSA Type','CSA Type']]
df_namus = df_namus.rename(columns={'State_x': 'State'})

# Drop rows that came out of the merge without a FIPS -- those are cases we
# couldn't reconcile to any county-year row, so they can't be used in a
# scaling regression.
df_namus = df_namus[df_namus['FIPS'].notna()].copy()

# ---------------------------------------------------------------------------
# Final exports.
# ---------------------------------------------------------------------------
df_namus.to_csv(r'/Users/ayushsarkar/missing_persons/missing_persons/export/mp_term.csv', index=False)

df_pop_final = df_pop_final[['FIPS', 'Year', 'County_pop', 'name', 'source', 'State', 'MSA Code', 'CSA Code', 'MSA Title', 'CSA Title', 'MSA_pop', 'CSA_pop', 'State_pop', 'CBSA Type', 'CSA Type']]
df_pop_final.to_csv(r'/Users/ayushsarkar/missing_persons/missing_persons/export/pop_term.csv', index=False)

print("Final row count:", len(df_namus))
print(df_namus.isna().sum())
