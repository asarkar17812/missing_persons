"""
Cumulative state-level scaling regression for Mexico (INEGI).

Fits the log-log regression  log10(cases_s) = gamma + beta * log10(pop_s)
across the 32 Mexican states (31 states + CDMX), using cumulative INEGI
case counts and 2025-projected state populations.

The output is the Mexico-side analog of the U.S. State panel in
scripts/us/visualization/regressions.py. Same axes, same fit method, same
readout (beta, gamma, 95% CIs, R-squared, total cases).

Expected output from the 07/02/2025 figshare snapshot:
    beta = 1.113 [0.659, 1.567]
    gamma = -3.856 [-6.802, -0.909]
    R^2 = 0.455
    Total cases: 126,894

Output: plots/mexico/cumulative_scaling.png

Run as:
    python scripts/mexico/regression.py
"""

import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Load the raw INEGI case dump and the cleaned population panel.
# ---------------------------------------------------------------------------
df_inegi = pd.read_csv('/Users/ayushsarkar/missing_persons/missing_persons/source/mexico_missing_persons/data.csv', dtype=str)
df_poblacion = pd.read_csv('/Users/ayushsarkar/missing_persons/missing_persons/export/poblacion.csv', dtype=str)

# Aggregate cases to per-state totals. We count VICTIM_ID rows rather than
# distinct victims; the raw INEGI file already deduplicates on victim where
# necessary.
df_inegi = df_inegi.groupby(['STATE_ID']).agg(
    Case_Count=('VICTIM_ID', 'count')
).sort_values('STATE_ID')

# ---------------------------------------------------------------------------
# Align the join keys.
# The population file uses `entidad_id` and the cases file uses STATE_ID;
# we rename + zero-pad both to two-digit strings so the merge succeeds
# regardless of how they were originally typed.
# ---------------------------------------------------------------------------
df_poblacion = df_poblacion.rename(columns={'entidad_id':'STATE_ID', 'Total':'State_pop'})

df_poblacion['STATE_ID'] = (
    df_poblacion['STATE_ID']
    .str.strip()
    .str.zfill(2)
)

# STATE_ID came in as the groupby index above; copy it into a column,
# zero-pad it, and reset the index.
df_inegi['STATE_ID'] = (
    df_inegi.index.astype(str)
    .str.strip()
    .str.zfill(2)
)
df_inegi = df_inegi.reset_index(drop=True)

df_scaling = df_inegi.merge(df_poblacion[['STATE_ID', 'State_pop', 'entidad']], on='STATE_ID', how='left')

# ---------------------------------------------------------------------------
# Build (log_pop, log_cases) for the regression.
# ---------------------------------------------------------------------------
df = df_scaling.copy()

# Numeric coercion: both columns came in as strings from the dtype=str load.
df['State_pop'] = pd.to_numeric(df['State_pop'], errors='coerce')
df['Case_Count'] = pd.to_numeric(df['Case_Count'], errors='coerce')

# Drop NaN and zero rows -- log10 of zero is -inf and would break the fit.
df = df.dropna(subset=['State_pop', 'Case_Count'])
df = df[(df['State_pop'] > 0) & (df['Case_Count'] > 0)]

df['log_pop'] = np.log10(df['State_pop'])
df['log_cases'] = np.log10(df['Case_Count'])

# ---------------------------------------------------------------------------
# Fit OLS in log space.
# ---------------------------------------------------------------------------
X = sm.add_constant(df['log_pop'])
y = df['log_cases']
model = sm.OLS(y, X).fit()

intercept, slope = model.params
conf_int = model.conf_int(alpha=0.05)
intercept_ci = conf_int.loc['const'].values
slope_ci = conf_int.loc['log_pop'].values
r2 = model.rsquared

# ---------------------------------------------------------------------------
# Build the prediction line and 95% CI band over a dense x-grid (200 points
# from x_min to x_max).
# ---------------------------------------------------------------------------
x_vals = np.linspace(df['log_pop'].min(), df['log_pop'].max(), 200)
x_vals_const = sm.add_constant(x_vals)
y_vals = model.predict(x_vals_const)
preds_ci = model.get_prediction(x_vals_const).summary_frame(alpha=0.05)

# Mean and median markers -- a noticeable gap between them flags skew.
mean_log_pop = df['log_pop'].mean()
mean_log_cases = df['log_cases'].mean()
median_log_pop = df['log_pop'].median()
median_log_cases = df['log_cases'].median()

# ---------------------------------------------------------------------------
# Render the figure.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(
    df['log_pop'], df['log_cases'],
    color='steelblue', alpha=0.7,
    label='$\log_{10}$(INEGI Case Count)'
)

ax.plot(
    x_vals, y_vals,
    color='darkred', linewidth=2,
    label='Regression line'
)

ax.fill_between(
    x_vals,
    preds_ci['mean_ci_lower'],
    preds_ci['mean_ci_upper'],
    color='lightcoral', alpha=0.3,
    label='95% CI'
)

ax.scatter(
    mean_log_pop, mean_log_cases,
    color='green', s=100, edgecolor='black',
    label='Mean point', zorder=5
)

ax.scatter(
    median_log_pop, median_log_cases,
    color='purple', s=100, edgecolor='black',
    label='Median point', zorder=5
)

# Stats block injected into the legend.
total_cases = df['Case_Count'].sum()
regression_label = (
    f"β = {slope:.3f} [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]\n"
    f"γ = {intercept:.3f} [{intercept_ci[0]:.3f}, {intercept_ci[1]:.3f}]\n"
    f"$R^2$ = {r2:.3f}\n"
    f"Total cases: {total_cases:,.0f}"
)
ax.plot([], [], ' ', label=regression_label)

ax.set_title(
    'Scaling of Cumulative INEGI Missing Persons Cases vs Mexican State Populations',
    fontsize=20
)
ax.set_xlabel('log(State Population)', fontsize=16)
ax.set_ylabel('log(INEGI Case Count)', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
ax.grid(True)
ax.legend(fontsize=12, loc='upper left')
ax.set_xlim(left=5.8)
ax.set_ylim(bottom=2)

plt.tight_layout()
plt.savefig('missing_persons/plots/mexico/cumulative_scaling.png', dpi=1200, bbox_inches='tight')
plt.show()
