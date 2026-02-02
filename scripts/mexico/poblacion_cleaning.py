import pandas as pd 
from io import StringIO

# --- read raw text ---
with open("/Users/ayushsarkar/missing_persons/missing_persons/source/mexico_missing_persons/INEGI_exporta_27_1_2026_11_15_40.csv", encoding="latin1") as f:
    lines = f.readlines()

# --- find header row (age groups) ---
header_idx = next(
    i for i, l in enumerate(lines)
    if l.strip().startswith(", , Total")
)

# --- find data start (national total row) ---
data_start = next(
    i for i, l in enumerate(lines)
    if l.lstrip().startswith(", Total")
)

# --- find footer ---
data_end = next(
    i for i, l in enumerate(lines)
    if l.startswith("FUENTE")
)

# --- rebuild a valid CSV ---
csv_text = (
    lines[header_idx] +          # real headers
    "".join(lines[data_start:data_end])
)

df = pd.read_csv(
    StringIO(csv_text),
    sep=",",
    quotechar='"',
    engine="python"
)

# --- fix first columns ---
df = df.rename(columns={
    df.columns[0]: "entidad_id",
    df.columns[1]: "entidad"
})

# --- clean column names ---
df.columns = (
    df.columns
    .str.strip()
    .str.replace("�", "ó", regex=False)
    .str.replace("a�os", "años", regex=False)
)

# --- clean numeric values ---
for col in df.columns[2:]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype("Int64")
    )

df.to_csv('missing_persons/export/poblacion.csv', index=False)
