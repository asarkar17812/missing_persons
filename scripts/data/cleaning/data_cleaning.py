import pandas as pd 
import geopandas as gpd
import numpy as np 
import json as json
import csv

### ----- SEER Historical County Population Estimates Processing -----
def clean_and_export_population_data(input_file, output_csv_file):
    cleaned_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) < 18:
                continue  # Skip malformed lines
            
            try:
                record = {
                    'Year': line[0:4],
                    'State': line[4:6],
                    'FIPS': line[6:11],
                    'Registry': line[11:13],
                    'Origin': line[13:15],
                    'Sex': line[15:16],
                    'Age': line[16:18],
                    'Population': int(line[18:]) if line[18:].isdigit() else None
                }
                cleaned_data.append(record)
            except Exception:
                continue  # Skip problematic lines silently

    # Export to CSV
    if cleaned_data:
        with open(output_csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=cleaned_data[0].keys())
            writer.writeheader()
            writer.writerows(cleaned_data)

clean_and_export_population_data('F:\\dsl_CLIMA\\projects\\submittable\\missing persons\\source\\SEER Population Estimates\\us_1969_2022.19ages.adjusted.txt', 
                                 'F:\\dsl_CLIMA\\projects\\submittable\\missing persons\\export\\us_pop_by_decade.csv')

df_population = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\missing persons\\export\\us_pop_by_decade.csv', dtype={'FIPS':str})
df_population['Year'] = df_population['Year'].astype(int)
df_cencount = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\missing persons\\source\\NBER County Population Estimates\\cencounts.csv')
df_pop_est = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\missing persons\\source\\2024 County Population Est\\co-est2024-alldata.csv', 
                         dtype={'STATE':str,"COUNTY":str, 'STNAME':str,'CTYNAME':str, "POPESTIMATE2024":'Int64', 'POPESTIMATE2023':'Int64'},
                         encoding='latin1')

df_population = df_population.groupby(["FIPS", "Year"], as_index=False).agg({"Population": "sum"}).copy()

df_pop_est['FIPS'] = df_pop_est['STATE'] + df_pop_est['COUNTY']
df_pop_extract = df_pop_est[['FIPS', 'POPESTIMATE2023', 'POPESTIMATE2024', 'STNAME', 'CTYNAME']].copy()
df_pop_extract['POPESTIMATE2023'] = pd.to_numeric(df_pop_extract['POPESTIMATE2023'], errors='coerce')
df_pop_extract['POPESTIMATE2024'] = pd.to_numeric(df_pop_extract['POPESTIMATE2024'], errors='coerce')

# Create rows for 2023
df_2023 = df_pop_extract[['FIPS', 'POPESTIMATE2023']].copy()
df_2023['Year'] = 2023
df_2023['Population'] = df_2023['POPESTIMATE2023']
df_2023 = df_2023.drop(columns=['POPESTIMATE2023'])

# Create rows for 2024
df_2024 = df_pop_extract[['FIPS', 'POPESTIMATE2024']].copy()
df_2024['Year'] = 2024
df_2024['Population'] = df_2024['POPESTIMATE2024']
df_2024 = df_2024.drop(columns=['POPESTIMATE2024'])

# Combine new population rows
df_new_pop = pd.concat([df_2023, df_2024], ignore_index=True)

# Fill missing columns with NaNs or appropriate values so df_new_pop matches df_primary columns
for col in df_population.columns:
    if col not in df_new_pop.columns:
        df_new_pop[col] = pd.NA

# Reorder columns to match df_primary
df_new_pop = df_new_pop[df_population.columns]

# Remove existing 2023/2024 rows
# df_population = df_population[~df_population['Year'].isin([2023, 2024])]

# Append new rows
df_population_updated = pd.concat([df_population, df_new_pop], ignore_index=True)
df_population_updated = df_population_updated.sort_values(by=['FIPS', 'Year']).reset_index(drop=True)

# FIPS code formatting and corrections
df_population_updated['FIPS'] = df_population_updated['FIPS'].astype(str).str.zfill(5)
df_cencount['fips'] = df_cencount['fips'].astype(str).str.zfill(5)

# Drop hyperspecific FIPS codes (xx9xx pattern)
df_population_updated = df_population_updated[~df_population_updated['FIPS'].str.match(r'^\d{2}9\d{2}$')]

fips_corrections = {
    '12086': '12025',  # Miami-Dade → Dade
    '46102': '46113',  # Oglala Lakota → Shannon
    '11999': '11001',  # DC legacy → DC standard
}
df_population_updated['FIPS_merge'] = df_population_updated['FIPS'].replace(fips_corrections)

# Merge from table (original census mapping)
df_merged = df_population_updated.merge(
    df_cencount[['fips', 'name']],
    left_on='FIPS_merge',
    right_on='fips',
    how='left'
).drop(columns=['fips'])
df_merged['source'] = df_merged['name'].apply(lambda x: 'table' if pd.notna(x) else None)

# Identify missing names for shapefile matching
df_nan = df_merged[df_merged['name'].isna()].copy()
df_nan['name_filled'] = None
df_nan['source'] = None

# === Step 2: Load shapefiles ===
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
subdivision_shape_files = {
    2023: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2023\subdivisions\US_cty_sub_2023.shp',
    2022: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2022\subdivisons\US_cty_sub_2022.shp',
    2010: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2010\subdivisions\US_cty_sub_2010.shp',
    2000: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\2000\subdivisions\US_cty_sub_2000.shp',
    1990: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1990\subdivisions\US_cty_sub_1990.shp',
    1980: r'F:\dsl_CLIMA\projects\Missing Persons Project\shape files\1980\subdivisions\US_mcd_1980.shp'
}

def build_fips_map(gdf):
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

# Match using shapefiles
for year, path in county_shape_files.items():
    print(f"🔍 Matching using shapefile {year}")
    try:
        gdf = gpd.read_file(path)
        fips_map = build_fips_map(gdf)
        unmatched = df_nan['name_filled'].isna()
        matches = df_nan.loc[unmatched, 'FIPS'].map(fips_map)
        df_nan.loc[unmatched, 'name_filled'] = matches
        df_nan.loc[unmatched & matches.notna(), 'source'] = f"shapefile_{year}"
        print(f"   ✅ Filled: {(unmatched & matches.notna()).sum()}")
        if df_nan['name_filled'].isna().sum() == 0:
            break
    except Exception as e:
        print(f"⚠️ Skipping {year}: {e}")

for year, path in subdivision_shape_files.items():
    print(f"🔍 Matching using SUBDIVISION shapefile {year}")
    try:
        gdf = gpd.read_file(path)
        fips_map = build_fips_map(gdf)
        unmatched = df_nan['name_filled'].isna()
        matches = df_nan.loc[unmatched, 'FIPS'].map(fips_map)
        df_nan.loc[unmatched, 'name_filled'] = df_nan['name_filled'].combine_first(matches)
        df_nan.loc[unmatched & matches.notna(), 'source'] = f"subdivision_shapefile_{year}"
        print(f"   ✅ Filled from subdivision: {(unmatched & matches.notna()).sum()}")
        if df_nan['name_filled'].isna().sum() == 0:
            break
    except Exception as e:
        print(f"⚠️ Skipping subdivision shapefile {year}: {e}")

# Update main df with matched names
df_nan['name'] = df_nan['name_filled']
df_merged.update(df_nan[['FIPS', 'name']])

# Manually assign known Alaska FIPS codes
manual_fips_names = {
    '02010': 'Haines Borough',
    '02232': 'Skagway-Hoonah-Angoon Census Area, Alaska'
}
for fips, name in manual_fips_names.items():
    df_merged.loc[df_merged['FIPS'] == fips, 'name'] = name
    df_merged.loc[df_merged['FIPS'] == fips, 'source'] = 'manual_alaska_fix'

# === Step 3: Expand state and county names ===
# Extract state FIPS code: first two digits of the 5-digit FIPS code
df_merged['state_fips'] = df_merged['FIPS'].str[:2]

# Mapping from state FIPS to state abbreviation and full name
state_fips_to_abbr = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE',
    '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
    '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
    '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM',
    '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
    '54': 'WV', '55': 'WI', '56': 'WY'
}

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

# Map state_fips to abbreviation and name
df_merged['state_abbr'] = df_merged['state_fips'].map(state_fips_to_abbr)
df_merged['state_name'] = df_merged['state_abbr'].map(us_state_abbrev)

# Extract county name from 'name' by removing state abbreviation at the start if present
df_merged['county_name'] = df_merged['name'].str.replace(r'^[A-Z]{2}\s+', '', regex=True).str.strip()

# Check missing after all matches
missing_final = df_merged[df_merged['name'].isna()]
print(f"🔢 Still missing after shapefile match: {len(missing_final)}")
print("Missing FIPS:", missing_final['FIPS'].unique())

# Final cleanup and export
df_merged = df_merged.drop(columns=['FIPS_merge', 'state_fips', 'name'])
df_merged.to_csv('F:\\dsl_CLIMA\\projects\\submittable\\missing persons\\export\\population.csv', index=False)


### ----- Crosswalk Processing ----- ###
# --- Load crosswalk Excel
df_crosswalk = pd.ExcelFile('F:\\dsl_CLIMA\\projects\\Missing Persons Project\\working_dfs\\qcew-county-msa-csa-crosswalk.xlsx')

# --- Load each crosswalk sheet
cw_2003 = pd.read_excel(df_crosswalk, sheet_name='Dec. 2003 Crosswalk', dtype=str)
cw_2013 = pd.read_excel(df_crosswalk, sheet_name='Feb. 2013 Crosswalk', dtype=str)
cw_2023 = pd.read_excel(df_crosswalk, sheet_name='Jul. 2023 Crosswalk', dtype=str)

# --- Clean and standardize crosswalks
def prepare_crosswalk(df_cw):
    df_cw['County Title'] = df_cw['County Title'].str.strip()
    return df_cw[['County Title', 'MSA Title', 'CSA Title', 'County Code', 'MSA Code', 'CSA Code']]

cw_2003 = prepare_crosswalk(cw_2003)
cw_2013 = prepare_crosswalk(cw_2013)
cw_2023 = prepare_crosswalk(cw_2023)

# --- Function to clean string columns
def clean_string_column(df, column):
    df[column] = df[column].astype(str).str.strip()
    df[column] = df[column].str.replace(r'\s+', ' ', regex=True)
    return df

# --- Clean county_title
for df in [cw_2003, cw_2013, cw_2023]:
    df = clean_string_column(df, 'County Title')

# --- Compare MSA Code changes between crosswalks
# First, rename MSA Codes to make them distinguishable
cw_2003_renamed = cw_2003.rename(columns={'MSA Code': 'MSA_Code_2003', 'MSA Title':'MSA_Title_2003', 'CSA Code':'CSA_Code_2003', 'CSA Title':'CSA_Title_2003' })
cw_2003_renamed['County_Code_2003'] = cw_2003['County Code']

cw_2013_renamed = cw_2013.rename(columns={'MSA Code': 'MSA_Code_2013', 'MSA Title': 'MSA_Title_2013', 'CSA Code':'CSA_Code_2013', 'CSA Title':'CSA_Title_2013'})
cw_2013_renamed['County_Code_2013'] = cw_2013['County Code']

cw_2023_renamed = cw_2023.rename(columns={'MSA Code': 'MSA_Code_2023', 'MSA Title': 'MSA_Title_2023', 'CSA Code':'CSA_Code_2023', 'CSA Title':'CSA_Title_2023'})
cw_2023_renamed['County_Code_2023'] = cw_2023['County Code']

# --- Merge 2003 with 2013, then 2013 with 2023
merge_03_13 = pd.merge(
    (cw_2003_renamed.copy())[['County Code', 'MSA_Code_2003', 'MSA_Title_2003', 'CSA_Code_2003', 'CSA_Title_2003']],
    (cw_2013_renamed.copy())[['County Code', 'MSA_Code_2013', 'MSA_Title_2013', 'CSA_Code_2013', 'CSA_Title_2013']],
    on='County Code',
    how='inner'
)

merge_13_23 = pd.merge(
    (cw_2013_renamed.copy())[['County Code',  'MSA_Code_2013', 'MSA_Title_2013', 'CSA_Code_2013', 'CSA_Title_2013']],
    (cw_2023_renamed.copy())[['County Code', 'MSA_Code_2023', 'MSA_Title_2023', 'CSA_Code_2023', 'CSA_Title_2023' ]],
    on='County Code',
    how='inner'
)

merge_03_13['MSA_Code_Change_03_13'] = merge_03_13['MSA_Code_2003'] != merge_03_13['MSA_Code_2013']
merge_13_23['MSA_Code_Change_13_23'] = merge_13_23['MSA_Code_2013'] != merge_13_23['MSA_Code_2023']

merge_03_13['CSA_Code_Change_03_13'] = merge_03_13['CSA_Code_2003'] != merge_03_13['CSA_Code_2013']
merge_13_23['CSA_Code_Change_13_23'] = merge_13_23['CSA_Code_2013'] != merge_13_23['CSA_Code_2023']

# --- Optional: merge all three together
all_merge = merge_03_13.merge(merge_13_23, on=['County Code'], how='outer')
all_merge['FIPS'] = all_merge['County Code'].astype(str)

# # --- Now, merge this info into the population dataframe
df_pop_final = df_merged.merge(all_merge, on=['FIPS'], how='left')

def summarize_population_by_msa_all_years(df):
    # Drop irrelevant columns if they exist
    df = df.drop(columns=['county_name'], errors='ignore')

    # Group by Year and MSA_Title
    df_grouped = df.groupby(['Year', 'MSA Code'], as_index=False)['MSA_pop'].sum()
    df_grouped = df_grouped.sort_values(by=['Year', 'MSA Code']).reset_index(drop=True)

    return df_grouped

def summarize_population_by_csa_all_years(df):
    # Group by Year and MSA_Title
    df_grouped = df.groupby(['Year', 'CSA Code'], as_index=False)['CSA_pop'].sum()
    df_grouped = df_grouped.sort_values(by=['Year', 'CSA Code']).reset_index(drop=True)

    return df_grouped

def assign_msa_code(row):
    year = row['Year']
    if year < 2013:
        return row['MSA_Code_2003']
    elif (year < 2023):
        if (row['MSA_Code_Change_03_13']==True):
            return row['MSA_Code_2013_x']
        else:
            return row['MSA_Code_2003']
    elif year > 2022:
        if row['MSA_Code_Change_13_23']==True:
            return row['MSA_Code_2023']
        elif row['MSA_Code_Change_03_13']==True:
            return row['MSA_Code_2013_x']
        else:
            return row['MSA_Code_2003'] 

def assign_msa_title(row):
    year = row['Year']
    if year < 2013:
        return row['MSA_Title_2003']
    elif (year < 2023):
        if (row['MSA_Code_Change_03_13']==True):
            return row['MSA_Title_2013_x']
        else:
            return row['MSA_Title_2003']
    elif year > 2022:
        if row['MSA_Code_Change_13_23']==True:
            return row['MSA_Title_2023']
        elif row['MSA_Code_Change_03_13']==True:
            return row['MSA_Title_2013_x']
        else:
            return row['MSA_Title_2003'] 

def assign_csa_code(row):
    year = row['Year']
    if year < 2013:
        return row['CSA_Code_2003']
    elif (year < 2023):
        if (row['CSA_Code_Change_03_13']==True):
            return row['CSA_Code_2013_x']
        else:
            return row['CSA_Code_2003']
    elif year > 2022:
        if row['CSA_Code_Change_13_23']==True:
            return row['CSA_Code_2023']
        elif row['CSA_Code_Change_03_13']==True:
            return row['CSA_Code_2013_x']
        else:
            return row['CSA_Code_2003'] 

def assign_csa_title(row):
    year = row['Year']
    if year < 2013:
        return row['CSA_Title_2003']
    elif (year < 2023):
        if (row['CSA_Code_Change_03_13']==True):
            return row['CSA_Title_2013_x']
        else:
            return row['CSA_Title_2003']
    elif year > 2022:
        if row['CSA_Code_Change_13_23']==True:
            return row['CSA_Title_2023']
        elif row['CSA_Code_Change_03_13']==True:
            return row['CSA_Title_2013_x']
        else:
            return row['CSA_Title_2003'] 

def clean_crosswalk(df_cw):
    # Split 'County Title' into 'County Name' and 'State Full Name' (both uppercase)
    df_cw[['County Name', 'State Full']] = df_cw['County Title'].str.upper().str.split(',', n=1, expand=True)
    df_cw['County Name'] = df_cw['County Name'].str.strip()
    df_cw['State Full'] = df_cw['State Full'].str.strip()

    # Extract MSA Name and MSA State Abbreviation from 'MSA Title'
    # Example: "Montgomery, AL MSA"
    # Split by comma -> ["Montgomery", " AL MSA"]
    # Then extract state abbreviation (2 letters) from second part
    msa_split = df_cw['MSA Title'].str.upper().str.split(',', n=1, expand=True)
    df_cw['MSA Name'] = msa_split[0].str.strip()
    df_cw['MSA State Abbr'] = msa_split[1].str.strip().str.slice(0, 2)  # first two chars after comma

    # Pad codes to 5 characters as string
    df_cw['County Code'] = df_cw['County Code'].astype(str).str.zfill(5)
    df_cw['MSA Code'] = df_cw['MSA Code'].astype(str).str.zfill(5)

    return df_cw

def merge_with_crosswalk(df_subset, cw):
    # Good county rows: merge on County Name + State Full Name
    df_good = df_subset[~df_subset['BadCounty']].merge(
        cw[['County Name', 'State Full', 'County Code', 'MSA Code', 'CSA Code']],
        left_on=['County', 'State'],
        right_on=['County Name', 'State Full'],
        how='left'
    )

    # Bad county rows: merge on City + MSA Name + MSA State Abbr
    df_bad = df_subset[df_subset['BadCounty']].copy()

    df_bad = df_bad.merge(
        cw[['MSA Name', 'MSA State Abbr', 'County Code', 'MSA Code', 'CSA Code']],
        left_on=['City', 'State'],
        right_on=['MSA Name', 'MSA State Abbr'],
        how='left'
    )

    # Combine good and bad rows
    df_combined = pd.concat([df_good, df_bad], ignore_index=True, sort=False)

    # Drop helper columns after merge
    df_combined.drop(columns=['County Name', 'State Full', 'MSA Name', 'MSA State Abbr'], inplace=True, errors='ignore')

    return df_combined

# Load and clean crosswalk sheets
crosswalk_file = 'F:\dsl_CLIMA\projects\Missing Persons Project\working_dfs\qcew-county-msa-csa-crosswalk.xlsx'
cw_2003 = clean_crosswalk(pd.read_excel(crosswalk_file, sheet_name='Dec. 2003 Crosswalk'))
cw_2013 = clean_crosswalk(pd.read_excel(crosswalk_file, sheet_name='Feb. 2013 Crosswalk'))
cw_2023 = clean_crosswalk(pd.read_excel(crosswalk_file, sheet_name='Jul. 2023 Crosswalk'))

# Date thresholds
date_2003 = pd.Timestamp('2003-12-01')
date_2013 = pd.Timestamp('2013-02-01')

# Split by date ranges and merge each part
df_2003 = df_namus[df_namus['DisappearanceDate'] <= date_2003]
df_2013 = df_namus[(df_namus['DisappearanceDate'] > date_2003) & (df_namus['DisappearanceDate'] < date_2013)]
df_2023 = df_namus[df_namus['DisappearanceDate'] >= date_2013]

df_2003_merged = merge_with_crosswalk(df_2003, cw_2003)
df_2013_merged = merge_with_crosswalk(df_2013, cw_2013)
df_2023_merged = merge_with_crosswalk(df_2023, cw_2023)

# Concatenate all final parts
df_final = pd.concat([df_2003_merged, df_2013_merged, df_2023_merged], ignore_index=True)

# Save final output CSV
# df_final.to_csv('abdulahi.csv', index=False)

df_final['MSA Code'] = df_final.apply(assign_msa_code, axis=1)
df_final['MSA Title'] = df_final.apply(assign_msa_title, axis=1)
df_final['CSA Code'] = df_final.apply(assign_csa_code, axis=1)
df_final['CSA Title'] = df_final.apply(assign_csa_title, axis=1)

df_cbsa = summarize_population_by_msa_all_years(df_final)
df_csa = summarize_population_by_csa_all_years(df_final)
df_cbsa = df_cbsa.sort_values(by=['MSA Code', 'Year']).reset_index(drop=True)

df_mp = df_final.merge(df_cbsa, on=['Year', 'MSA Code'], how='left')
df_mp = df_mp.merge(df_csa, on=['Year', 'CSA Code'], how='left')

df_mp = df_mp.drop(columns=['MSA_Code_2023', 'MSA_Code_2013_y', 'MSA_Code_2013_x', 'MSA_Code_2003', 'MSA_Code_Change_13_23', 'MSA_Code_Change_03_13', 'MSA_Title_2023', 'MSA_Title_2013_x', 'MSA_Title_2013_y', 'MSA_Title_2003','CSA_Code_2023', 'CSA_Code_2013_y', 'CSA_Code_2013_x', 'CSA_Code_2003', 'CSA_Code_Change_13_23', 'CSA_Code_Change_03_13', 'CSA_Title_2023', 'CSA_Title_2013_x', 'CSA_Title_2013_y', 'CSA_Title_2003', 'MSA_pop_x', 'CSA_pop_x'])
df_mp = df_mp.rename(columns={'Total_Population':'County_pop', 'MSA_pop_y':'MSA_pop', 'CSA_pop_y':'CSA_pop'})

def simplify_titles(df, msa_col='MSA Title', csa_col='CSA Title'):
    """
    Simplify MSA_Title and CSA_Title by keeping only the part before the first comma.
    Also strips whitespace.
    Extract the last word of the original MSA_Title into a new column 'CBSA Type',
    which would typically be 'MSA', 'MicroSA', or NaN.
    """
    if msa_col in df.columns:
        # Extract last word from original MSA_Title (handle NaNs)
        df['CBSA Type'] = df[msa_col].astype(str).str.extract(r'(\w+)$')[0]
        # Replace 'nan' string with actual np.nan
        df['CBSA Type'] = df['CBSA Type'].replace('nan', np.nan)

    if csa_col in df.columns:
        df['CSA Type'] = df[csa_col].astype(str).str.extract(r'(\w+)$')[0]
        # Replace 'nan' string with actual np.nan
        df['CSA Type'] = df['CSA Type'].replace('nan', np.nan)

    for col in [msa_col, csa_col]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.split(',', n=1).str[0].str.strip()

    return df

df_mp = simplify_titles(df_mp)
df_mp['FIPS'] = df_mp['FIPS'].astype(str)

df_mp = df_mp.fillna(np.nan)
df_mp.to_csv('F:\dsl_CLIMA\projects\submittable\missing persons\export\\mp_term.csv')