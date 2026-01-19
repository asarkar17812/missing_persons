import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

df_namus = pd.read_csv(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\export\mp_term.csv'
)

df_namus['DisappearanceDate'] = pd.to_datetime(df_namus['DisappearanceDate'])

df_namus = df_namus[
    (df_namus['DisappearanceDate'] > pd.to_datetime('1968-12-31')) &
    (df_namus['DisappearanceDate'] < pd.to_datetime('2025-01-01'))
]

####################################################
# Population Pyramid of Missing Persons
####################################################

# Keep only rows with valid age + sex
df_plot = df_namus.dropna(subset=['CurrentMinAge', 'CurrentMaxAge', 'Sex']).copy()

# Optional: standardize sex labels just in case
df_plot['Sex'] = df_plot['Sex'].str.capitalize()

# Number of included cases (IMPORTANT: from original rows)
n_cases = df_plot.shape[0]

age_bins = [
    (0, 4), (5, 9), (10, 14), (15, 17), (18, 19), (20, 20), (21, 21),
    (22, 24), (25, 29), (30, 34), (35, 39), (40, 44), (45, 49),
    (50, 54), (55, 59), (60, 61), (62, 64), (65, 66), (67, 69),
    (70, 74), (75, 79), (80, 84), (85, 120)
]

age_labels = [
    'Under 5', '5-9', '10-14', '15-17', '18-19', '20', '21', '22-24',
    '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
    '60-61', '62-64', '65-66', '67-69', '70-74', '75-79', '80-84', '85+'
]

def expand_to_age_bins(df):
    records = []

    for _, row in df.iterrows():
        for (low, high), label in zip(age_bins, age_labels):
            if row['CurrentMaxAge'] >= low and row['CurrentMinAge'] <= high:
                records.append({
                    'Sex': row['Sex'],
                    'AgeBin': label
                })

    return pd.DataFrame(records)

expanded = expand_to_age_bins(df_plot)

counts = (
    expanded
    .groupby(['AgeBin', 'Sex'])
    .size()
    .unstack(fill_value=0)
    .reindex(age_labels)
)

# Ensure columns exist even if one sex is missing
counts = counts.reindex(columns=['Male', 'Female'], fill_value=0)

total = counts.sum().sum()
male_percent = counts['Male'] / total * 100
female_percent = counts['Female'] / total * 100

y = np.arange(len(age_labels))

fig, ax = plt.subplots(figsize=(10, 8))

ax.barh(y, -male_percent.values, color='steelblue', label='Male')
ax.barh(y,  female_percent.values, color='lightcoral', label='Female')

ax.set_yticks(y)
ax.set_yticklabels(age_labels)
ax.axvline(0, color='black', linewidth=1)

# Make x-axis labels show positive percentages
ax.xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, _: f"{abs(x):.0f}%")
)

# Symmetric x-limits
max_val = max(male_percent.max(), female_percent.max())
ax.set_xlim(-max_val * 1.1, max_val * 1.1)

ax.set_xlabel("Percent of Total (%)")
ax.set_title("Age / Sex Distribution of Cumulative Missing Persons Cases [1969-2024]")

# Add N label (bottom-right)
ax.text(
    0.99, 0.01,
    f"N = {n_cases:,} cases",
    transform=ax.transAxes,
    ha='right',
    va='bottom',
    fontsize=11,
    color='gray'
)

ax.legend()
plt.tight_layout()
plt.savefig(r'F:\dsl_CLIMA\projects\submittable\missing persons\plots\regressions\demographics\population_pyramids\[1969-2024]mp_pop_pyramid.png', dpi=1200, bbox_inches='tight')
plt.show()