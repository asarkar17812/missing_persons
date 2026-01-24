import pandas as pd
import matplotlib.pyplot as plt

df_namus = pd.read_csv(
    r'export/mp_term.csv'
)

df_namus['DisappearanceDate'] = pd.to_datetime(df_namus['DisappearanceDate'])

df_namus = df_namus[
    (df_namus['DisappearanceDate'] > pd.to_datetime('2009-12-31')) &
    (df_namus['DisappearanceDate'] < pd.to_datetime('2025-01-01'))
]

df_namus['Sex'] = df_namus['Sex'].astype(str).str.strip().str.capitalize()
df_namus['Ethnicity'] = df_namus['Ethnicity'].astype(str).str.strip()

df_namus = df_namus.dropna(subset=['Sex', 'Ethnicity'])


# ==================================================
# BAR CHART: Ethnicity Distribution
# ==================================================

eth_counts = df_namus['Ethnicity'].value_counts()
n_eth = eth_counts.sum()

# Group small categories
threshold = 0.03  # 3%
eth_percent = eth_counts / n_eth
small = eth_percent < threshold

eth_counts_grouped = eth_counts.copy()
if small.any():
    eth_counts_grouped = eth_counts_grouped[~small]
    eth_counts_grouped['Other'] = eth_counts[small].sum()

# Convert to percentages
eth_percent_grouped = eth_counts_grouped / n_eth * 100

# Sort for nicer plotting
eth_percent_grouped = eth_percent_grouped.sort_values()

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    eth_percent_grouped.index,
    eth_percent_grouped.values,
    color='slategray',
    edgecolor='black',
    alpha=0.85
)

# Label bars with % and count
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

ax.set_xlim(0, eth_percent_grouped.max() * 1.25)

plt.tight_layout()
plt.savefig(
    r'plots/demographics/[2010-2024]/[2010-2024]_mp_ethnicity_bar.png',
    dpi=1200,
    bbox_inches='tight'
)
plt.show()