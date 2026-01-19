import pandas as pd
import numpy as np
import csv
import json

# ===============================
# Helper: tokenize missing values
# ===============================
def tokenize(value):
    """Convert empty/missing/unknown/redacted values into consistent tokens."""
    if value is None:
        return "MISSING"
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in ["", "na", "n/a", "null", "not available"]:
            return "CENSORED"
        if stripped in ["unknown", "unk"]:
            return "UNKNOWN"
    return value


# ===============================
# Load raw NamUs JSON
# ===============================
with open(
    r'F:\dsl_CLIMA\projects\Missing Persons Project\output\namus-20250717.json',
    'r',
    encoding='utf-8'
) as f:
    data = json.load(f)

main_data = []

for entry in data:
    subject = entry.get("subjectIdentification", {})
    desc = entry.get("subjectDescription", {})
    physical = entry.get("physicalDescription", {})
    sighting = entry.get("sighting", {})
    agency = entry.get("primaryInvestigatingAgency", {})

    row = {
        "CaseID": tokenize(entry.get("idFormatted")),
        "CurrentMinAge": tokenize(subject.get("currentMinAge")),
        "CurrentMaxAge": tokenize(subject.get("currentMaxAge")),
        "Sex": tokenize(desc.get("sex", {}).get("name") if desc.get("sex") else None),
        "Ethnicity": tokenize(desc.get("primaryEthnicity", {}).get("name") if desc.get("primaryEthnicity") else None),
        "DisappearanceDate": tokenize(sighting.get("date")),
        "City": tokenize(sighting.get("address", {}).get("city") if sighting.get("address") else None),
        "State": tokenize(
            sighting.get("address", {})
            .get("state", {})
            .get("name") if sighting.get("address") else None
        ),
        "County": tokenize(
            sighting.get("address", {})
            .get("county", {})
            .get("name") if sighting.get("address") else None
        ),
        "InvestigatingAgency": tokenize(agency.get("name")),
    }

    main_data.append(row)


# ===============================
# Write intermediate CSV
# ===============================
output_csv = r'F:\dsl_CLIMA\projects\submittable\missing persons\export\cleaned_missing_persons.csv'

with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=main_data[0].keys())
    writer.writeheader()
    writer.writerows(main_data)


# ===============================
# Reload as DataFrame
# ===============================
df_namus = pd.read_csv(
    output_csv,
    parse_dates=['DisappearanceDate'],
    date_parser=lambda x: pd.to_datetime(x, errors='coerce')
)

df_namus = df_namus[
    ["CaseID", "CurrentMinAge", "CurrentMaxAge", "Sex", "Ethnicity",
     "DisappearanceDate", "City", "State", "County"]
].copy()


# ===============================
# Year handling (cap pre-1969)
# ===============================
df_namus['Year'] = df_namus['DisappearanceDate'].dt.year
df_namus.loc[df_namus['Year'] < 1969, 'Year'] = 1969
df_namus.loc[df_namus['Year'] > 2024, 'Year'] = 2024
df_namus['Year'] = df_namus['Year'].astype(int)


# ===============================
# Connecticut post-2022 handling
# ===============================
connecticut_cities_to_county = {
    'EAST HARTFORD': 'CAPITOL PLANNING REGION',
    'MERIDEN': 'SOUTH CENTRAL CONNECTICUT PLANNING REGION',
    'NEW BRITAIN': 'CAPITOL PLANNING REGION',
    'TORRINGTON': 'NORTHWEST HILLS PLANNING REGION',
    'WEST HARTFORD': 'CAPITOL PLANNING REGION',
    'GLASTONBURY': 'CAPITOL PLANNING REGION',
    'DERBY': 'NAUGATUCK VALLEY PLANNING REGION',
    'LISBON': 'SOUTHEASTERN CONNECTICUT PLANNING REGION',
    'AVON': 'CAPITOL PLANNING REGION',
    'GUILFORD': 'SOUTH CENTRAL CONNECTICUT PLANNING REGION',
    'HAMDEN': 'SOUTH CENTRAL CONNECTICUT PLANNING REGION',
    'GROTON': 'SOUTHEASTERN CONNECTICUT PLANNING REGION',
    'BRIDGEPORT': 'GREATER BRIDGEPORT PLANNING REGION',
    'NEW HAVEN': 'SOUTH CENTRAL CONNECTICUT PLANNING REGION',
    'HARTFORD': 'CAPITOL PLANNING REGION',
    'LEDYARD': 'SOUTHEASTERN CONNECTICUT PLANNING REGION',
    'DANBURY': 'SOUTHEASTERN CONNECTICUT PLANNING REGION'
}

# Normalize early
df_namus['State'] = df_namus['State'].astype(str).str.strip().str.upper()
df_namus['County'] = df_namus['County'].astype(str).str.strip().str.upper()
df_namus['City'] = df_namus['City'].astype(str).str.strip().str.upper()

ct_mask = (df_namus['State'] == 'CONNECTICUT') & (df_namus['Year'] > 2022)
mapped_ct = df_namus.loc[ct_mask, 'City'].map(connecticut_cities_to_county)
df_namus.loc[ct_mask, 'County'] = mapped_ct.combine_first(df_namus.loc[ct_mask, 'County'])


# ===============================
# Drop territories
# 
# ===============================
dropped_states = {
    'PUERTO RICO',
    'VIRGIN ISLANDS',
    'GUAM',
    'NORTHERN MARIANA ISLANDS'
}
df_namus = df_namus[~df_namus['State'].isin(dropped_states)]

# ===============================
# Bad county flag
# ===============================
bad_values = {'MISSING', 'UNKNOWN', 'CENSORED'}

df_namus['County'] = (
    df_namus['County']
        .astype(str)
        .str.strip()
        .replace({'NAN': np.nan})
)

df_namus = df_namus[
    df_namus['County'].notna() &
    (~df_namus['County'].isin(bad_values))
].copy()


# ===============================
# Export final NamUs cases
# Total Cases: 25532
# ===============================
df_namus.to_csv( r'F:\dsl_CLIMA\projects\submittable\missing persons\export\namus_cases.csv', index=False)

print("Final row count:", len(df_namus))
print(df_namus)