import pandas as pd
import matplotlib.pyplot as plt

df_namus = pd.read_csv(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\export\mp_term.csv'
)

df_namus['DisappearanceDate'] = pd.to_datetime(df_namus['DisappearanceDate'])

df_namus = df_namus[
    (df_namus['DisappearanceDate'] > pd.to_datetime('1968-12-31')) &
    (df_namus['DisappearanceDate'] < pd.to_datetime('2025-01-01'))
]

df_namus['Sex'] = df_namus['Sex'].astype(str).str.strip().str.capitalize()
df_namus['Ethnicity'] = df_namus['Ethnicity'].astype(str).str.strip()

df_namus = df_namus.dropna(subset=['Sex', 'Ethnicity'])

####################################################
# Pi Chart of NamUs Missing Persons Cases
####################################################
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
    f"Sex Distribution of Cumulative NamUs Missing Persons[1969-2024] Cases\n(N = {n_sex:,} cases)",
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()
plt.savefig(
    r'F:\dsl_CLIMA\projects\submittable\missing persons\plots\regressions\demographics\mp_sex_distribution.png',
    dpi=1200,
    bbox_inches='tight'
)
plt.show()