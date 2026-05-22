"""
CBSA-type distribution bar chart for NamUs cases (1969-2024).

Counts NamUs cases by CBSA type (MSA / MicroSA / None for counties not in
any CBSA) and renders the breakdown as a vertical bar chart. The point is
to sanity-check the case-mass distribution before partitioning the data
into MSA-only and MicroSA-only scaling regressions -- if MicroSAs only
carry ~10% of cases, a noisy MicroSA fit is partly a small-sample story
rather than a real structural difference in how those areas scale.

Output: plots/type_distribution/[1969-2024]mp_type_distribution(cbsa).png

Run as:
    python scripts/us/visualization/cbsaType_distribution.py
"""

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Load the merged case+population frame and clip to the cumulative window.
# ---------------------------------------------------------------------------
df_primary = pd.read_csv(r'F:\dsl_CLIMA\projects\submittable\missing persons\export\mp_term.csv')
df_primary['DisappearanceDate'] = pd.to_datetime(df_primary['DisappearanceDate'])

df_primary = df_primary[
    (df_primary['DisappearanceDate'] > pd.to_datetime('1968-12-31')) &
    (df_primary['DisappearanceDate'] < pd.to_datetime('2025-01-01'))
]


def plot_cbsa_type_distribution(df):
    """Render the CBSA-type bar chart for one case frame.

    Counts the `CBSA Type` field (with NaN treated as a "None" category for
    counties that don't belong to any CBSA), annotates each bar with the
    count and percent share, and writes the figure to PNG.

    Bars are sorted ascending by count, and the in-bar text switches to
    coral on the tiny bars (where black-on-tab20 would have low contrast).
    """
    if 'CBSA Type' not in df.columns:
        raise ValueError("'CBSA Type' column not found in the DataFrame.")

    # Treat NaN as its own category ("None" = no CBSA assignment).
    counts = df['CBSA Type'].value_counts(dropna=False)
    counts.index = counts.index.fillna('None')
    total = counts.sum()

    counts_df = counts.sort_values(ascending=True).to_frame(name='count')
    counts_df['percentage'] = counts_df['count'] / total * 100

    # tab20 gives us 20 distinct colors -- more than enough since CBSA Type
    # has at most a handful of categories.
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(counts_df))]

    plt.figure(figsize=(18, 8))
    bars = plt.bar(
        counts_df.index,
        counts_df['count'],
        color=colors,
        edgecolor='black'
    )

    # In-bar annotation with count and percent. Switch to coral text when
    # the bar is too small for black text to be legible.
    for bar, count, perc in zip(bars, counts_df['count'], counts_df['percentage']):
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f'{count:,}\n({perc:.1f}%)',
                ha='center',
                va='center',
                fontsize=16,
                color='black' if height > total * 0.05 else 'coral',
                fontweight='bold'
            )

    # Total-cases banner in the upper-left corner of the plot.
    plt.text(
        0.05, 0.95,
        f"Total Cases: {total:,}",
        transform=plt.gca().transAxes,
        fontsize=22,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black')
    )

    plt.title(
        f"Distribution of NamUS Missing Persons Cases by CBSA Type (1969–2024)",
        fontsize=28
    )
    plt.xlabel("CBSA Type", fontsize=28)
    plt.ylabel("Number of Cases", fontsize=28)
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(r"F:\dsl_CLIMA\projects\submittable\missing persons\plots\type_distribution\[1969-2024]mp_type_distribution(cbsa).png", dpi=1200, bbox_inches='tight')
    plt.show()


plot_cbsa_type_distribution(df_primary)
