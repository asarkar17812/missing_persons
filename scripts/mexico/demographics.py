import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Example: df already exists
df_inegi = pd.read_csv(r'F:\dsl_CLIMA\projects\submittable\missing persons\source\mexico_missing_persons\data.csv', dtype=str)

# Ensure DATE_OF_BIRTH is datetime
df_inegi["DATE_OF_BIRTH"] = pd.to_datetime(df_inegi["DATE_OF_BIRTH"], errors="coerce")
df_inegi["DATE_OF_INCIDENCE"] = pd.to_datetime(df_inegi["DATE_OF_INCIDENCE"], errors="coerce")

# -----------------------
# PIE CHART: SEX
# -----------------------
# Normalize SEX values and keep special categories
df_inegi["SEX_CLEAN"] = (
    df_inegi["SEX"]
    .fillna("MISSING")
    .str.upper()
)

# Count values
sex_counts = df_inegi["SEX_CLEAN"].value_counts()

# Desired order
desired_order = ["MISSING", "CONFIDENTIAL", "MALE", "FEMALE"]

# Reindex to enforce order and keep existing categories
sex_counts = sex_counts.reindex(
    [x for x in desired_order if x in sex_counts.index]
)

plt.figure(figsize=(7, 7))

def autopct_with_counts(values):
    def inner(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n(n={count})"
    return inner

wedges, _, _ = plt.pie(
    sex_counts,
    autopct=autopct_with_counts(sex_counts),
    startangle=0  # x = 0
)

plt.title("Sex Distribution of INEGI Missing Persons Cases")

# Legend at the bottom with counts
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
# BAR CHART: AGE BRACKETS
# -----------------------
today = pd.Timestamp(datetime.today())
df_inegi["AGE_MISSING"] = ((df_inegi['DATE_OF_INCIDENCE'] - df_inegi["DATE_OF_BIRTH"]).dt.days // 365)

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

# Add count labels above bars
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
