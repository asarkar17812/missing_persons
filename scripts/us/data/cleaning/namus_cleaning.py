"""
NamUs cleaning pipeline.

Reads the raw NamUs JSON dump (from scripts/us/data/scraper/namus.py) and
produces two CSVs:

    export/cleaned_missing_persons.csv
        One row per case, with the eight columns we actually use downstream.
        Missing / empty / unknown fields are tokenized as MISSING / CENSORED
        / UNKNOWN so that downstream code can tell them apart.

    export/namus_cases.csv
        The above, post-filtering: years clamped to [1969, 2024], territories
        (PR, VI, GU, MP) dropped, Connecticut post-2022 city -> planning-region
        remap applied, and rows whose County field can't be resolved dropped.
        Final row count: 25,532 (as of the 07/17/2025 snapshot).

The two CSVs exist as separate artifacts because some downstream demographic
analyses want the un-filtered version (so they can count CENSORED records
honestly), while the scaling regression wants only rows with a usable County.

Run as:
    python scripts/us/data/cleaning/namus_cleaning.py
"""

import pandas as pd
import numpy as np
import csv
import json


def tokenize(value):
    """Normalize a field to one of three explicit tokens or return as-is.

    NamUs fields come in three "bad" flavors that we want to distinguish:
      - `None` from the JSON  -> "MISSING"   (the field was never populated)
      - Empty / "N/A"-like     -> "CENSORED" (the field was explicitly blanked)
      - "Unknown" / "Unk"      -> "UNKNOWN"  (the source labeled it unknown)
    Anything else (including normal strings, numbers, etc.) passes through
    unchanged.

    Keeping these distinct lets us reason later about *why* a column is
    sparse — random data loss vs. deliberate redaction vs. agency-reported
    uncertainty are three different problems.
    """
    if value is None:
        return "MISSING"
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in ["", "na", "n/a", "null", "not available"]:
            return "CENSORED"
        if stripped in ["unknown", "unk"]:
            return "UNKNOWN"
    return value


# ---------------------------------------------------------------------------
# Step 1: load the raw NamUs JSON and flatten the nested case structure.
# ---------------------------------------------------------------------------

with open(
    r'F:\dsl_CLIMA\projects\Missing Persons Project\output\namus-20250717.json',
    'r',
    encoding='utf-8'
) as f:
    data = json.load(f)

main_data = []

# Each NamUs case is a deeply-nested object; pull out only the fields we use.
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


# ---------------------------------------------------------------------------
# Step 2: write the intermediate, un-filtered CSV.
# ---------------------------------------------------------------------------

output_csv = r'F:\dsl_CLIMA\projects\submittable\missing persons\export\cleaned_missing_persons.csv'

with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=main_data[0].keys())
    writer.writeheader()
    writer.writerows(main_data)


# ---------------------------------------------------------------------------
# Step 3: reload as a DataFrame for the filtering / normalization steps.
# We parse DisappearanceDate as a datetime up front because we need to clamp
# the year next.
# ---------------------------------------------------------------------------

df_namus = pd.read_csv(
    output_csv,
    parse_dates=['DisappearanceDate'],
    date_parser=lambda x: pd.to_datetime(x, errors='coerce')
)

df_namus = df_namus[
    ["CaseID", "CurrentMinAge", "CurrentMaxAge", "Sex", "Ethnicity",
     "DisappearanceDate", "City", "State", "County"]
].copy()


# ---------------------------------------------------------------------------
# Step 4: clamp the year to [1969, 2024].
#   - Our SEER population panel only starts in 1969, so anything earlier
#     gets clipped to 1969 (rather than dropped).
#   - 2025 entries are in-progress and get clipped to 2024.
# ---------------------------------------------------------------------------

df_namus['Year'] = df_namus['DisappearanceDate'].dt.year
df_namus.loc[df_namus['Year'] < 1969, 'Year'] = 1969
df_namus.loc[df_namus['Year'] > 2024, 'Year'] = 2024
df_namus['Year'] = df_namus['Year'].astype(int)


# ---------------------------------------------------------------------------
# Step 5: Connecticut post-2022 handling.
#
# Background: Connecticut spent more than three centuries divided into eight
# counties (Fairfield, Hartford, Litchfield, Middlesex, New Haven, New
# London, Tolland, Windham). In 2019 the state petitioned the Census
# Bureau to retire those counties as a statistical unit; the Office of
# Management and Budget approved the change in June 2022, and as of
# vintage 2023 every federal release (Census, BLS, BEA, SEER going
# forward) replaces the eight counties with nine "Planning Regions"
# (Capitol, Greater Bridgeport, Lower Connecticut River Valley, Naugatuck
# Valley, Northeast, Northwest Hills, South Central Connecticut,
# Southeast, Western Connecticut). The counties still exist as legal
# entities -- towns are still incorporated under them -- but they no
# longer carry any federal statistical weight.
#
# The problem for NamUs: case records from 2023 onward still write the
# city ("HARTFORD", "BRIDGEPORT", etc.) but the county field is often
# left blank, or worse, still names the legacy county that no longer
# exists in the population panel. If we leave those rows alone they
# either drop out at the merge (no FIPS match) or get assigned to a
# county that doesn't appear in the 2023 crosswalk vintage.
#
# The fix: a manual city -> planning-region lookup. The dictionary below
# enumerates every Connecticut city that appears in the 07/17/2025 NamUs
# snapshot with a disappearance year > 2022, mapped to the planning
# region the city now belongs to under the Jul. 2023 BLS crosswalk. The
# mapping is hand-built rather than algorithmic because the underlying
# town -> planning-region table has edge cases (a few towns straddle
# planning-region boundaries, others were re-assigned between draft and
# final OMB delineations) and because the list is small enough -- ~17
# affected cities -- that an explicit table is easier to audit than a
# fuzzy match.
# ---------------------------------------------------------------------------

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

# Normalize string columns to upper-case stripped form before the merge.
df_namus['State'] = df_namus['State'].astype(str).str.strip().str.upper()
df_namus['County'] = df_namus['County'].astype(str).str.strip().str.upper()
df_namus['City'] = df_namus['City'].astype(str).str.strip().str.upper()

ct_mask = (df_namus['State'] == 'CONNECTICUT') & (df_namus['Year'] > 2022)
mapped_ct = df_namus.loc[ct_mask, 'City'].map(connecticut_cities_to_county)
# `combine_first` preserves the existing County value if the city isn't in
# our manual map — we only overwrite when we have a known mapping.
df_namus.loc[ct_mask, 'County'] = mapped_ct.combine_first(df_namus.loc[ct_mask, 'County'])


# ---------------------------------------------------------------------------
# Step 6: drop U.S. territories.
# SEER doesn't cover PR/VI/GU/MP, so cases from those territories can't
# be joined to a population panel and have to be dropped.
# ---------------------------------------------------------------------------

dropped_states = {
    'PUERTO RICO',
    'VIRGIN ISLANDS',
    'GUAM',
    'NORTHERN MARIANA ISLANDS'
}
df_namus = df_namus[~df_namus['State'].isin(dropped_states)]


# ---------------------------------------------------------------------------
# Step 7: drop rows whose County field can't be resolved.
# Anything tokenized as MISSING/UNKNOWN/CENSORED can't be merged to a FIPS
# code in the crosswalk step, so we drop it. We also coerce a literal
# "NAN" string (which pandas sometimes leaves around) back to actual NaN.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Step 8: write the filtered CSV.
# Expected row count: 25,532 (07/17/2025 snapshot).
# ---------------------------------------------------------------------------

df_namus.to_csv(r'F:\dsl_CLIMA\projects\submittable\missing persons\export\namus_cases.csv', index=False)

print("Final row count:", len(df_namus))
print(df_namus)
