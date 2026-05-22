"""
Ethnicity distribution bar chart for NamUs cases (2010-2024).

Builds a horizontal bar chart of the cumulative ethnicity breakdown of NamUs
missing-persons cases over the 2010-2024 window. Categories that hold less
than 3% of total cases get rolled up into an "Other" bucket so the long tail
of rarely-reported labels doesn't dominate the y-axis. Each bar is annotated
with the percent share and the raw count.

Output: plots/demographics/[2010-2024]/[2010-2024]_mp_ethnicity_bar.png

Run as:
    python scripts/us/visualization/bar_charts.py
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

# Normalize Sex/Ethnicity formatting so value_counts collapses near-duplicates.
df_namus['Sex'] = df_namus['Sex'].astype(str).str.strip().str.capitalize()
df_namus['Ethnicity'] = df_namus['Ethnicity'].astype(str).str.strip()

# Drop rows missing either field -- we report the conditional distribution
# over records with usable demographics.
df_namus = df_namus.dropna(subset=['Sex', 'Ethnicity'])


# ==================================================
# Ethnicity distribution bar chart
# ==================================================

eth_counts = df_namus['Ethnicity'].value_counts()
n_eth = eth_counts.sum()

# Group small categories (< 3% of total) into "Other" so the long tail of
# rarely-reported labels doesn't crowd the y-axis.
threshold = 0.03
eth_percent = eth_counts / n_eth
small = eth_percent < threshold

eth_counts_grouped = eth_counts.copy()
if small.any():
    eth_counts_grouped = eth_counts_grouped[~small]
    eth_counts_grouped['Other'] = eth_counts[small].sum()

# Convert to percentages and sort ascending for plotting (largest bar on top).
eth_percent_grouped = eth_counts_grouped / n_eth * 100
eth_percent_grouped = eth_percent_grouped.sort_values()

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    eth_percent_grouped.index,
    eth_percent_grouped.values,
    color='slategray',
    edgecolor='black',
    alpha=0.85
)

# Annotate each bar with its percentage and the underlying count.
for bar, label in zip(bars, eth_percent_grouped.index):
    pct = eth_percent_grouped[label]
    count = eth_counts_grouped[label]
    ax.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{pct:.1f}% (N={count:,})",
        va='center',
        fontsize=10
    )

ax.set_xlabel("Percent of Total (%)")
ax.set_title(
    f"Ethnicity Distribution of Cumulative NamUs Missing Persons [2010-2024] Cases\n(N = {n_eth:,} cases)",
    fontsize=14,
    fontweight='bold'
)

# Give the annotation text a bit of headroom past the longest bar.
ax.set_xlim(0, eth_percent_grouped.max() * 1.25)

plt.tight_layout()
plt.savefig(
    r'plots/demographics/[2010-2024]/[2010-2024]_mp_ethnicity_bar.png',
    dpi=1200,
    bbox_inches='tight'
)
plt.show()
