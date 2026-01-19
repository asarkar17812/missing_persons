import pandas as pd
import numpy as np
import geopandas as gpd

# --- Load files ---
df_population = pd.read_csv(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\export\population.csv',
    dtype={'FIPS': str}
)
df_namus = pd.read_csv(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\export\namus_cases.csv'
)
crosswalk_file = r'F:\dsl_CLIMA\projects\Missing Persons Project\working_dfs\qcew-county-msa-csa-crosswalk.xlsx'

# --- Helper functions ---
bad_values = {'MISSING', 'UNKNOWN', 'CENSORED'}
def clean_crosswalk(df_cw):
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
    df = df_subset.copy()

    # --- Normalize columns ---
    df['County_norm'] = df['County'].astype(str).str.upper().str.strip()
    df['State_norm'] = df['State'].astype(str).str.upper().str.strip()

    cw['County_norm'] = cw['County Title'].astype(str).str.upper().str.strip()
    cw['MSA_Title_norm'] = cw['MSA Title'].astype(str).str.upper().str.split(',', n=1).str[0].str.strip()
    cw['State_abbr'] = cw['MSA Title'].astype(str).str.extract(r',\s*([^\s]+)', expand=False)
    
    # Map abbreviation to full state name if needed
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

    # --- Split good vs bad counties ---
    good_mask = df['County'].notna() & (~df['County'].isin(bad_values))
    df_good = df[good_mask].copy()

    # --- Merge good counties by FIPS ---
    df_good = df_good.merge(
        cw[['County Code','County Title','MSA Code','CSA Code','MSA Title','CSA Title']],
        left_on='FIPS',
        right_on='County Code',
        how='left'
    ).drop(columns=['County Code'], errors='ignore')

    return df_good

def merge_cases_with_crosswalk(df_subset, cw, bad_values={'MISSING', 'UNKNOWN', 'CENSORED'}):
    df = df_subset.copy()

    # --- Normalize columns ---
    df['County_norm'] = df['County'].astype(str).str.upper().str.strip()
    df['City_norm'] = df['City'].astype(str).str.upper().str.strip() 
    df['State_norm'] = df['State'].astype(str).str.upper().str.strip()

    cw['County_norm'] = cw['County Title'].astype(str).str.upper().str.strip()
    cw['MSA_Title_norm'] = cw['MSA Title'].astype(str).str.upper().str.split(',', n=1).str[0].str.strip()
    cw['State_abbr'] = cw['MSA Title'].astype(str).str.extract(r',\s*([^\s]+)', expand=False)
    
    # Map abbreviation to full state name if needed
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

    # --- Split good vs bad counties ---
    good_mask = df['County'].notna() & (~df['County'].isin(bad_values))
    df_good = df[good_mask].copy()
    df_bad = df[~good_mask].copy()

    # --- Merge good counties by FIPS ---
    df_good = df_good.merge(
        cw[['County Code','County Title','MSA Code','CSA Code','MSA Title','CSA Title']],
        left_on='FIPS',
        right_on='County Code',
        how='left'
    ).drop(columns=['County Code'], errors='ignore')

    # --- Merge bad counties by City → MSA ---
    df_bad = df_bad.merge(
        cw[['MSA_Title_norm','State_full','MSA Code','CSA Code','MSA Title','CSA Title']],
        left_on=['City_norm','State_norm'],
        right_on=['MSA_Title_norm','State_full'],
        how='left'
    ).drop(columns=['MSA_Title_norm','State_full'], errors='ignore')

    # --- Fill FIPS and County for MSAs with a single county ---
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

    # --- Recombine ---
    df_merged = pd.concat([df_good, df_bad], ignore_index=True)
    df_merged.drop(columns=['County_norm','City_norm','State_norm'], errors='ignore', inplace=True)

    return df_merged


def summarize_population_by_msa_all_years(df):
    return (
        df.groupby(['Year', 'MSA Code'], as_index=False)
          .agg(MSA_pop=('Population', 'sum'))
          .sort_values(['Year', 'MSA Code'])
          .reset_index(drop=True)
    )

def summarize_population_by_csa_all_years(df):
    return (
        df.groupby(['Year', 'CSA Code'], as_index=False)
          .agg(CSA_pop=('Population', 'sum'))
          .sort_values(['Year', 'CSA Code'])
          .reset_index(drop=True)
    )

def simplify_titles(df):
    if 'MSA Title' in df.columns:
        df['CBSA Type'] = df['MSA Title'].astype(str).str.extract(r'(\w+)$')[0].replace('nan', np.nan)
        df['MSA Title'] = df['MSA Title'].astype(str).str.split(',', n=1).str[0].str.strip()
    if 'CSA Title' in df.columns:
        df['CSA Type'] = df['CSA Title'].astype(str).str.extract(r'(\w+)$')[0].replace('nan', np.nan)
        df['CSA Title'] = df['CSA Title'].astype(str).str.split(',', n=1).str[0].str.strip()
    return df

# --- Normalize for merging ---
df_namus['County_norm'] = df_namus['County'].str.upper().str.strip()
df_namus['State_norm'] = df_namus['State'].str.upper().str.strip()
df_population['County_norm'] = df_population['name'].str.upper().str.strip()
df_population['State_norm'] = df_population['State'].str.upper().str.strip()
df_namus['Year'] = df_namus['Year'].astype(int)
df_population['Year'] = df_population['Year'].astype(int)

# --- Deduplicate population to avoid row multiplication ---
df_population = df_population.drop_duplicates(subset=['Year', 'State_norm', 'County_norm'])

# --- Load and clean crosswalks ---
cw_2003 = clean_crosswalk(pd.read_excel(crosswalk_file, sheet_name='Dec. 2003 Crosswalk', dtype=str))
cw_2013 = clean_crosswalk(pd.read_excel(crosswalk_file, sheet_name='Feb. 2013 Crosswalk', dtype=str))
cw_2023 = clean_crosswalk(pd.read_excel(crosswalk_file, sheet_name='Jul. 2023 Crosswalk', dtype=str))

# --- Split Population by year ---
df_population['County'] = df_population['name'].copy()
df_pop_2003 = df_population[df_population['Year'] <= 2003]
df_pop_2013 = df_population[(df_population['Year'] > 2003) & (df_population['Year'] < 2013)]
df_pop_2023 = df_population[df_population['Year'] >= 2013]

df_pop_final = pd.concat([
    merge_pop_with_crosswalk(df_pop_2003, cw_2003),
    merge_pop_with_crosswalk(df_pop_2013, cw_2013),
    merge_pop_with_crosswalk(df_pop_2023, cw_2023)
], ignore_index=True)

df_cbsa = summarize_population_by_msa_all_years(df_pop_final)
df_csa = summarize_population_by_csa_all_years(df_pop_final)

# --- Summarize populations ---
df_pop_final = (
    df_pop_final
    .merge(df_cbsa, on=['Year', 'MSA Code'], how='left')
    .merge(df_csa, on=['Year', 'CSA Code'], how='left')
    .rename(columns={'Population': 'County_pop'})
).copy()

df_pop_final = simplify_titles(df_pop_final)

# --- Merge population ---
df_namus = df_namus.merge(
    df_pop_final[['FIPS', 'Year', 'County_pop', 'name', 'source', 'State', 'MSA Code', 'CSA Code', 'MSA Title', 'CSA Title', 'MSA_pop', 'CSA_pop', 'CBSA Type', 'CSA Type', 'County_norm', 'State_norm']],
    on=['Year', 'County_norm', 'State_norm'],
    how='left'
).drop_duplicates()

df_namus = df_namus[['CaseID','CurrentMinAge','CurrentMaxAge','Sex','Ethnicity','DisappearanceDate','City','State_x','County','Year','FIPS','County_pop','MSA Code','CSA Code','MSA Title','CSA Title','MSA_pop','CSA_pop','CBSA Type','CSA Type']]
df_namus = df_namus.rename(columns={'State_x': 'State'})
# --- Filter years and drop territories ---
# df_namus = df_namus[(df_namus['Year'] > 1999) & (df_namus['Year'] < 2025)]

# # --- Drop rows without FIPS after merge ---
df_namus = df_namus[df_namus['FIPS'].notna()].copy()

# --- Export ---
df_namus.to_csv(r'F:\dsl_CLIMA\projects\submittable\missing persons\export\mp_term.csv', index=False)

df_pop_final = df_pop_final[['FIPS', 'Year', 'County_pop', 'name', 'source', 'State', 'MSA Code', 'CSA Code', 'MSA Title', 'CSA Title', 'MSA_pop', 'CSA_pop', 'CBSA Type', 'CSA Type']]
df_pop_final.to_csv(r'F:\dsl_CLIMA\projects\submittable\missing persons\export\pop_term.csv', index=False)

print("Final row count:", len(df_namus))
print(df_namus.isna().sum())