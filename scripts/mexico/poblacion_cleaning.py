"""
Mexican state-level population panel construction.

Parses the INEGI Población projection file (a CSV with embedded header
lines, footer text, Latin-1 encoding, and broken accent characters) into a
tidy state x age-group panel.

Output: export/poblacion.csv

Why this file is so fiddly: INEGI's exported CSV is a human-readable
spreadsheet dump, not a machine-readable one. The first ~10 lines are
multi-row column headers, the last ~3 lines are a "FUENTE: ..." source
attribution footer, and the file is encoded in Latin-1 with the Spanish n-tilde
mangled to a U+FFFD replacement character ("?"). We scan the file with
plain text parsing, locate the real header row by string-matching, splice
out only the data section, and feed that into pandas.

Run as:
    python scripts/mexico/poblacion_cleaning.py
"""

import pandas as pd
from io import StringIO


# ---------------------------------------------------------------------------
# Read the raw text. The file is Latin-1 because INEGI's exporter doesn't
# emit UTF-8.
# ---------------------------------------------------------------------------
with open("/Users/ayushsarkar/missing_persons/missing_persons/source/mexico_missing_persons/INEGI_exporta_27_1_2026_11_15_40.csv", encoding="latin1") as f:
    lines = f.readlines()

# ---------------------------------------------------------------------------
# Locate the real header row.
# INEGI's exporter puts a section banner above the actual column headers;
# the column-header row is the one that starts with ", , Total" (i.e. two
# blank rank/index columns followed by the "Total" column header).
# ---------------------------------------------------------------------------
header_idx = next(
    i for i, l in enumerate(lines)
    if l.strip().startswith(", , Total")
)

# ---------------------------------------------------------------------------
# Locate where the data starts. The first data row is the national total,
# which begins with ", Total" (one blank rank column, then the label
# "Total").
# ---------------------------------------------------------------------------
data_start = next(
    i for i, l in enumerate(lines)
    if l.lstrip().startswith(", Total")
)

# ---------------------------------------------------------------------------
# Locate the footer. INEGI prepends "FUENTE: ..." to its source attribution
# footer; anything from that line onward is metadata, not data.
# ---------------------------------------------------------------------------
data_end = next(
    i for i, l in enumerate(lines)
    if l.startswith("FUENTE")
)

# ---------------------------------------------------------------------------
# Rebuild a valid CSV by splicing the header row directly onto the data
# slice. The header_idx + data_start::data_end ordering matters -- the
# original file has multiple header lines we want to skip, and the data
# region is a contiguous block.
# ---------------------------------------------------------------------------
csv_text = (
    lines[header_idx] +
    "".join(lines[data_start:data_end])
)

df = pd.read_csv(
    StringIO(csv_text),
    sep=",",
    quotechar='"',
    engine="python"
)

# ---------------------------------------------------------------------------
# Rename the first two columns to entidad_id / entidad. INEGI's header for
# these is empty (they're the rank and label columns), so pandas gives
# them auto-generated names like "Unnamed: 0" -- we fix that here.
# ---------------------------------------------------------------------------
df = df.rename(columns={
    df.columns[0]: "entidad_id",
    df.columns[1]: "entidad"
})

# ---------------------------------------------------------------------------
# Clean up the column headers. INEGI's Latin-1 export mangles the n-tilde
# ('ñ') to U+FFFD ('?'), so "años" comes out as "a?os". Fix the substitution
# explicitly so downstream code can rely on the standard spelling.
# ---------------------------------------------------------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.replace("?", "ó", regex=False)
    .str.replace("a?os", "años", regex=False)
)

# ---------------------------------------------------------------------------
# Clean the numeric cells. INEGI ships population counts with thousands
# separators ("1,234,567"); strip them and coerce to a nullable integer
# dtype so we can keep NaN sentinels for missing data.
# ---------------------------------------------------------------------------
for col in df.columns[2:]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype("Int64")
    )

df.to_csv('missing_persons/export/poblacion.csv', index=False)
