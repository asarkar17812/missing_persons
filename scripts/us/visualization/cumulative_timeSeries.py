import pandas as pd  
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df_primary = pd.read_csv(r'F:\dsl_CLIMA\projects\submittable\missing persons\export\mp_term.csv')
df_msa = pd.read_csv(r'F:\dsl_CLIMA\projects\Missing Persons Project\msa_cases_by_MSAcode[2010-2024].csv')

# --- Prepare and clean data
df_primary['DisappearanceDate'] = pd.to_datetime(df_primary['DisappearanceDate'], errors='coerce')
df_primary = df_primary.dropna(subset=['DisappearanceDate'])
df_primary = df_primary.set_index('DisappearanceDate').sort_index()

# --- Resample by month
disappearances_per_month = df_primary.resample('MS').size()

# --- Plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(disappearances_per_month.index, disappearances_per_month.values, color='darkred', linewidth=1.5)

# --- Axis labels and title
ax.set_title('Cumulative NamUS Cases per Month (2000–2024)', fontsize=32)
ax.set_xlabel('Month', fontsize=28)
ax.set_ylabel('Number of NamUS Cases', fontsize=26)
ax.grid(True)

# --- Show minor ticks every 3 months
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
ax.tick_params(axis='x', which='minor', length=4)  # short ticks for quarters

# --- Show major ticks only at the start of each year
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # e.g., Jan 2020

# --- Style
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(r'F:\dsl_CLIMA\projects\submittable\missing persons\plots\regressions\cumulative_cbsa\[2000-2024]cpm_ts_cases.png', dpi=1200, bbox_inches='tight')
plt.show()