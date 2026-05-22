"""
Demographic visualizations for the Mexico INEGI dataset.

Produces two figures from the raw INEGI cases dump:

    plots/mexico/INEGI_sex_pi_chart.png
        Pie chart of the sex distribution. Unlike the U.S. version we
        deliberately *do not* drop CONFIDENTIAL / MISSING rows, so the
        chart honestly shows the censorship overlay instead of pretending
        the conditional distribution among non-censored cases is the same
        as the overall distribution.

    plots/mexico/INEGI_age_at_incidence_barChart.png
        Histogram of the subject's age at the moment of disappearance,
        computed as (DATE_OF_INCIDENCE - DATE_OF_BIRTH) / 365. Because
        both date columns are sparse (59% and 43% missing respectively),
        the usable sample is roughly ~47k records out of the 129,830
        total.

Run as:
    python scripts/mexico/demographics.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# ---------------------------------------------------------------------------
# Load the raw INEGI cases dump and parse the two date columns we use.
# ---------------------------------------------------------------------------
df_inegi = pd.read_csv(r'F:\dsl_CLIMA\projects\submittable\missing persons\source\mexico_missing_persons\data.csv', dtype=str)

df_inegi["DATE_OF_BIRTH"] = pd.to_datetime(df_inegi["DATE_OF_BIRTH"], errors="coerce")
df_inegi["DATE_OF_INCIDENCE"] = pd.to_datetime(df_inegi["DATE_OF_INCIDENCE"], errors="coerce")


# -----------------------
# PIE CHART: sex
# -----------------------
# Normalize SEX values to upper-case and label NaN rows as 'MISSING' so
# they appear as their own category instead of vanishing.
df_inegi["SEX_CLEAN"] = (
    df_inegi["SEX"]
    .fillna("MISSING")
    .str.upper()
)

sex_counts = df_inegi["SEX_CLEAN"].value_counts()

# Enforce a fixed slice order so the visual ordering is consistent across
# regenerations (otherwise pandas sorts by count, which flips when the
# data refreshes).
desired_order = ["MISSING", "CONFIDENTIAL", "MALE", "FEMALE"]
sex_counts = sex_counts.reindex(
    [x for x in desired_order if x in sex_counts.index]
)

plt.figure(figsize=(7, 7))


def autopct_with_counts(values):
    """Build an autopct formatter that prints both percent and raw count.

    Matplotlib's `autopct` callback only receives the wedge's percent. To
    print the underlying count we close over the input `values` (the
    series we're piing) and back-compute the count from the percent on
    each call.
    """
    def inner(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n(n={count})"
    return inner


wedges, _, _ = plt.pie(
    sex_counts,
    autopct=autopct_with_counts(sex_counts),
    startangle=0
)

plt.title("Sex Distribution of INEGI Missing Persons Cases")

# Legend on the bottom edge with the count for each label, so the pie
# remains readable even at small sizes.
legend_labels = [f"{sex} (n={count})" for sex, count in sex_counts.items()]
plt.legend(
    wedges,
    legend_labels,
    title="Sex",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2
)

plt.axis("equal")
plt.tight_layout()
plt.savefig(r'F:\dsl_CLIMA\projects\submittable\missing persons\plots\mexico\INEGI_sex_pi_chart.png')
plt.show()


# -----------------------
# BAR CHART: age at incidence
# -----------------------
# Compute age at incidence in years.
# Integer-division-by-365 is approximate (no leap-year correction) but the
# bin granularity (10-year buckets) absorbs the rounding error.
today = pd.Timestamp(datetime.today())
df_inegi["AGE_MISSING"] = ((df_inegi['DATE_OF_INCIDENCE'] - df_inegi["DATE_OF_BIRTH"]).dt.days // 365)

# 10-year buckets with an open-ended 70+ bucket at the top. `right=False`
# means each bucket is [low, high) so 30 falls into "30-39", not "20-29".
bins = [0, 10, 20, 30, 40, 50, 60, 70, 200]
labels = ["0–9", "10–19", "20–29", "30–39",
          "40–49", "50–59", "60–69", "70+"]

df_inegi["AGE_BRACKET"] = pd.cut(df_inegi["AGE_MISSING"], bins=bins, labels=labels, right=False)

age_counts = df_inegi["AGE_BRACKET"].value_counts().sort_index()

plt.figure(figsize=(10, 6))
bars = plt.bar(age_counts.index.astype(str), age_counts.values)

plt.xlabel("Age Bracket")
plt.ylabel("Number of Cases")
plt.title("Distribution of INEGI Missing Persons Cases: Ages at Incidence")

# Annotate each bar with the raw count.
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(r'F:\dsl_CLIMA\projects\submittable\missing persons\plots\mexico\INEGI_age_at_incidence_barChart.png')
plt.show()
