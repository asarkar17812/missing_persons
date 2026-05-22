"""
Age x sex population pyramid for NamUs cases.

Builds a Census-style population pyramid of cumulative NamUs missing-persons
cases over 1969-2024, with male on the left and female on the right.

Two non-obvious details:

    1. The pyramid is over the subject's *current* age, not their age at
       disappearance. A person who disappeared at 25 in 1990 and is still
       missing today shows up in the 55-59 bin. This is why the cumulative
       pyramid has a substantial middle-aged and older population that
       wouldn't appear if we aggregated by age-at-incidence.

    2. NamUs reports CurrentMinAge and CurrentMaxAge rather than a single
       age, since for unsolved cases the subject's current age is an
       interval. To preserve that uncertainty we expand each record into
       every age bin that overlaps [CurrentMinAge, CurrentMaxAge]. This
       overcounts slightly (a 22-26-year-old contributes to both the
       22-24 and 25-29 bins) but it's the only way to keep the interval
       information without dropping records.

The age bins are non-uniform on purpose -- they mirror standard Census
age-group cuts in the low and high tails so the pyramid can be visually
compared against published Census pyramids without rebinning.

Output: plots/regressions/demographics/population_pyramids/
            [1969-2024]mp_pop_pyramid.png

Run as:
    python scripts/us/visualization/population_pyramids.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ---------------------------------------------------------------------------
# Load the merged case+population frame and clip to the cumulative window.
# ---------------------------------------------------------------------------
df_namus = pd.read_csv(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\export\mp_term.csv'
)

df_namus['DisappearanceDate'] = pd.to_datetime(df_namus['DisappearanceDate'])

df_namus = df_namus[
    (df_namus['DisappearanceDate'] > pd.to_datetime('1968-12-31')) &
    (df_namus['DisappearanceDate'] < pd.to_datetime('2025-01-01'))
]

# Keep only rows with usable age + sex (CurrentMinAge / CurrentMaxAge are
# numeric, but pandas leaves them as object dtype after the merge -- we
# rely on direct comparison below rather than coercing).
df_plot = df_namus.dropna(subset=['CurrentMinAge', 'CurrentMaxAge', 'Sex']).copy()
df_plot['Sex'] = df_plot['Sex'].str.capitalize()

# N reported in the bottom-right of the plot reflects raw record count, not
# the post-expansion count below.
n_cases = df_plot.shape[0]

# ---------------------------------------------------------------------------
# Census-style age bins. Non-uniform on purpose -- finer cuts in the youth
# and senior tails match published Census pyramids without rebinning.
# ---------------------------------------------------------------------------
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
    """Expand each (CurrentMinAge, CurrentMaxAge) record into every overlapping bin.

    Returns a long-format DataFrame with one row per (record, overlapping
    bin) pair. The membership test is the inclusive interval-overlap rule
    `CurrentMaxAge >= bin_low and CurrentMinAge <= bin_high`.

    A record with current age 22-26 will produce two rows -- one for the
    22-24 bin and one for the 25-29 bin -- so the resulting counts
    overcount slightly, but the alternative (collapsing each record to a
    single age) would discard the interval information NamUs preserves.
    """
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

# Pivot into a (AgeBin x Sex) count matrix. .reindex enforces our age-bin
# ordering instead of pandas's default alphabetic sort, and fills missing
# bins with 0 cases (e.g. if no Female cases fall into a given bin).
counts = (
    expanded
    .groupby(['AgeBin', 'Sex'])
    .size()
    .unstack(fill_value=0)
    .reindex(age_labels)
)

# Ensure both Sex columns exist even if one is missing in this window.
counts = counts.reindex(columns=['Male', 'Female'], fill_value=0)

# Convert counts to percentages of the total expansion population.
total = counts.sum().sum()
male_percent = counts['Male'] / total * 100
female_percent = counts['Female'] / total * 100

y = np.arange(len(age_labels))

# ---------------------------------------------------------------------------
# Draw the pyramid -- male bars go negative (left), female bars go positive
# (right). We use a symmetric x-axis so the visual midline is at 0.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

ax.barh(y, -male_percent.values, color='steelblue', label='Male')
ax.barh(y,  female_percent.values, color='lightcoral', label='Female')

ax.set_yticks(y)
ax.set_yticklabels(age_labels)
ax.axvline(0, color='black', linewidth=1)

# X-axis shows positive percentages on both sides; the lambda strips the
# sign so the male side reads "5%" not "-5%".
ax.xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, _: f"{abs(x):.0f}%")
)

max_val = max(male_percent.max(), female_percent.max())
ax.set_xlim(-max_val * 1.1, max_val * 1.1)

ax.set_xlabel("Percent of Total (%)")
ax.set_title("Age / Sex Distribution of Cumulative Missing Persons Cases [1969-2024]")

# N annotation in the bottom-right shows the raw (pre-expansion) record count.
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
