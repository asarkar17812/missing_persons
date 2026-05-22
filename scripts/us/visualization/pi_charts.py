"""
Sex distribution pie chart for NamUs cases (2010-2024).

Builds a pie chart of the cumulative sex breakdown of NamUs missing-persons
cases over the 2010-2024 window, conditional on a usable Sex field.

Two caveats worth keeping in mind when interpreting the output:
    1. NamUs records sex as reported by the agency filing the case, so the
       field is binary in the data even though the underlying population
       obviously isn't.
    2. Rows whose Sex tokenized to MISSING / UNKNOWN / CENSORED are dropped
       *before* computing percentages, so the chart shows the conditional
       distribution among usable records, not the raw distribution.

Output: plots/demographics/[2010-2024]/[2010-2024]_mp_sex_distribution.png

Run as:
    python scripts/us/visualization/pi_charts.py
"""

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Load the merged case+population frame and restrict to 2010-2024.
# ---------------------------------------------------------------------------
df_namus = pd.read_csv(
    r'export/mp_term.csv'
)

df_namus['DisappearanceDate'] = pd.to_datetime(df_namus['DisappearanceDate'])

df_namus = df_namus[
    (df_namus['DisappearanceDate'] > pd.to_datetime('2009-12-31')) &
    (df_namus['DisappearanceDate'] < pd.to_datetime('2025-01-01'))
]

# Normalize Sex/Ethnicity formatting and drop rows missing either field.
df_namus['Sex'] = df_namus['Sex'].astype(str).str.strip().str.capitalize()
df_namus['Ethnicity'] = df_namus['Ethnicity'].astype(str).str.strip()

df_namus = df_namus.dropna(subset=['Sex', 'Ethnicity'])

# ---------------------------------------------------------------------------
# Pie chart of the sex distribution.
# ---------------------------------------------------------------------------
sex_counts = df_namus['Sex'].value_counts()
n_sex = sex_counts.sum()

fig, ax = plt.subplots(figsize=(8, 6))

ax.pie(
    sex_counts.values,
    labels=sex_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    counterclock=False,
    wedgeprops={'edgecolor': 'white'}
)

ax.set_title(
    f"Sex Distribution of Cumulative NamUs Missing Persons [2010-2024] Cases\n(N = {n_sex:,} cases)",
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()
plt.savefig(
    r'plots/demographics/[2010-2024]/[2010-2024]_mp_sex_distribution.png',
    dpi=1200,
    bbox_inches='tight'
)
plt.show()
