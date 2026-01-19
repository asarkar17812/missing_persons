import pandas as pd  
import matplotlib.pyplot as plt

df_primary = pd.read_csv(r'F:\dsl_CLIMA\projects\submittable\missing persons\export\mp_term.csv')
df_primary['DisappearanceDate'] = pd.to_datetime(df_primary['DisappearanceDate'])

df_primary = df_primary[
    (df_primary['DisappearanceDate'] > pd.to_datetime('1968-12-31')) &
    (df_primary['DisappearanceDate'] < pd.to_datetime('2025-01-01'))
]

def plot_cbsa_type_distribution(df):

    if 'CBSA Type' not in df.columns:
        raise ValueError("'CBSA Type' column not found in the DataFrame.")
    
    # Count values, include NaNs under 'None'
    counts = df['CBSA Type'].value_counts(dropna=False)
    counts.index = counts.index.fillna('None')
    total = counts.sum()

    # Convert counts to DataFrame for sorting and indexing
    counts_df = counts.sort_values(ascending=True).to_frame(name='count')
    counts_df['percentage'] = counts_df['count'] / total * 100

    # Use a color map for distinct colors
    cmap = plt.get_cmap('tab20')  # Choose from: 'Set3', 'tab20', 'viridis', etc.
    colors = [cmap(i) for i in range(len(counts_df))]

    # Plot
    plt.figure(figsize=(18, 8))
    bars = plt.bar(
        counts_df.index,
        counts_df['count'],
        color=colors,
        edgecolor='black'
    )

    # Annotate each bar with count and percentage
    for bar, count, perc in zip(bars, counts_df['count'], counts_df['percentage']):
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,  # vertically centered inside the bar
                f'{count:,}\n({perc:.1f}%)',
                ha='center',
                va='center',
                fontsize=16,
                color='black' if height > total * 0.05 else 'coral',  # ensure contrast
                fontweight='bold'
            )


    plt.text(
        0.05, 0.95,
        f"Total Cases: {total:,}",
        transform=plt.gca().transAxes,
        fontsize=22,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black')
    )
    # Labels and formatting
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