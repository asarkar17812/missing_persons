"""
Monthly cumulative-cases time series for NamUs (2000-2024).

Resamples the NamUs case frame to monthly counts and plots the resulting
curve. The slope of this curve is effectively the monthly reporting rate;
visible kinks in the curve usually correspond to NamUs intake-policy
changes or new state-level participation rather than real shifts in
underlying disappearance rates.

This is a quick exploratory plot rather than a "published" figure -- the
output savefig line is commented out by default. Toggle it back on if you
want a PNG.

Run as:
    python scripts/us/visualization/cumulative_timeSeries.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# Load and clean the data.
# ---------------------------------------------------------------------------
df_primary = pd.read_csv(r'/Users/ayushsarkar/missing_persons/missing_persons/export/mp_term.csv')

# Parse the disappearance date and drop any rows where parsing failed --
# we can't plot a time series for a record with no date.
df_primary['DisappearanceDate'] = pd.to_datetime(df_primary['DisappearanceDate'], errors='coerce')
df_primary = df_primary.dropna(subset=['DisappearanceDate'])
df_primary = df_primary.set_index('DisappearanceDate').sort_index()

# ---------------------------------------------------------------------------
# Resample to month-start counts. resample('MS').size() returns one row per
# month with the count of cases whose disappearance date fell in that month.
# ---------------------------------------------------------------------------
disappearances_per_month = df_primary.resample('MS').size()

# ---------------------------------------------------------------------------
# Plot. Major ticks at year starts, minor ticks every 3 months for context.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(disappearances_per_month.index, disappearances_per_month.values, color='darkred', linewidth=1.5)

ax.set_title('Cumulative NamUS Cases per Month (2000–2024)', fontsize=32)
ax.set_xlabel('Month', fontsize=28)
ax.set_ylabel('Number of NamUS Cases', fontsize=26)
ax.grid(True)

# Quarterly minor ticks give context without crowding the axis.
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
ax.tick_params(axis='x', which='minor', length=4)

# Major ticks at year boundaries, formatted as "Jan YYYY".
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=18)
plt.tight_layout()
# Toggle the savefig below back on to write the figure to disk.
# plt.savefig(r'F:\dsl_CLIMA\projects\submittable\missing persons\plots\regressions\cumulative_cbsa\[2000-2024]cpm_ts_cases.png', dpi=1200, bbox_inches='tight')
plt.show()
