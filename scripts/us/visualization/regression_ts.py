"""
Temporal scaling regression for U.S. NamUs cases vs. state population.

For each year t from 1969 to 2024, fits a log-log regression
    log10(cases_t) = gamma_t + beta_t * log10(pop_t)
using only the cases that occurred in that year, and tracks how the exponent
beta(t) evolves over time. Three figures are produced:

    plots/regressions/temporal/states/[1969-2024]cumulative_cases.png
        Cumulative monthly NamUs cases from 1969 onward. The slope is the
        running reporting rate; visible kinks usually correspond to NamUs
        intake-policy shifts rather than real disappearance-rate changes.

    plots/regressions/temporal/states/[1969-2024]_regression_ts_states_annual.png
        beta(t) with 95% CI error bars. A flat curve means the geographic
        distribution of NamUs reporting has reached a steady state; a
        drifting curve means it hasn't.

    plots/regressions/temporal/states/[1969-2024]_regression_comparison_ts_states_annual.png
        Two scatter plots side by side: the worst-R-squared year and the
        best-R-squared year, so we can see what tight vs. loose fits look
        like in detail.

    plots/regressions/temporal/states/[1969-2024]_r2_ts_states_annual.png
        R-squared(t) trace with the best and worst years highlighted.

Cases prior to 1969 are not dropped -- they're carried as a "baseline" count
that gets added to the cumulative effective total but doesn't enter any
yearly regression.

Run as:
    python scripts/us/visualization/regression_ts.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Load the merged case+population frame and parse dates.
# Commented-out filters above the index assignment are kept for reference:
# uncomment one to repeat the regression on MSAs / MicroSAs / CSAs instead
# of States.
# ---------------------------------------------------------------------------
df_primary = pd.read_csv(r'export/mp_term.csv')

df_primary['DisappearanceDate'] = pd.to_datetime(df_primary['DisappearanceDate'], errors='coerce')
# df_primary = df_primary[df_primary['CBSA Type'] == 'MSA']      # alt: MSA-only
# df_primary = df_primary[df_primary['CBSA Type'] == 'MicroSA']  # alt: MicroSA-only
# df_primary = df_primary[df_primary['CSA Type'] == 'CSA']       # alt: CSA-only

df_primary = df_primary.set_index('DisappearanceDate').sort_index()
print(df_primary.isna().sum())

df_primary['Year'] = df_primary.index.year

# ---------------------------------------------------------------------------
# Baseline: count of pre-1969 cases. These get folded into the running
# cumulative total but not into any yearly regression.
# ---------------------------------------------------------------------------
baseline_cases = df_primary[df_primary['Year'] < 1969].shape[0]
print(f"\nBaseline cases prior to 1969: {baseline_cases}")

# ---------------------------------------------------------------------------
# Build the monthly cumulative case count (for the cumulative-cases figure
# only -- the per-year regression below uses annual counts).
# ---------------------------------------------------------------------------
all_months = pd.date_range(start='1969-01-01', end='2024-12-31', freq='MS')
monthly_counts = df_primary.resample('MS').size().reindex(all_months, fill_value=0)
cumulative_disappearances = monthly_counts.cumsum()

plot_start = '1969-01-01'
cumulative_to_plot = cumulative_disappearances[cumulative_disappearances.index >= plot_start]

# January-only tick positions so the x-axis stays readable.
january_points = cumulative_to_plot[cumulative_to_plot.index.month == 1]

plt.figure(figsize=(14, 10))
plt.plot(cumulative_to_plot.index, cumulative_to_plot.values,
         color='darkgreen', linewidth=2, label='Cumulative NamUS Missing Persons Cases (1969–2024)')
plt.title('Cumulative NamUS Missing Persons Cases by Month Start (1969–2024)', fontsize=24)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Cumulative NamUS Missing Persons Cases', fontsize=28)
plt.grid(False)
plt.xticks(ticks=january_points.index, labels=[d.strftime('%Y') for d in january_points.index], fontsize=14, rotation=60)
y_ticks = [0, 2000, 4000, 6000, 8000, 10000, 11969, 14000, 16000, 18000, 20000, 22000, 24000]
plt.yticks(y_ticks, fontsize=14)
plt.ylim(y_ticks[0] + .5)
plt.xlim(all_months[0], all_months[-1])
plt.tight_layout()
plt.savefig(r'/Users/ayushsarkar/missing_persons/missing_persons/plots/regressions/temporal/states/[1969-2024]cumulative_cases.png', dpi=1200, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# Per-year state-level regression loop.
# We fit a fresh OLS in log space for each year t using only cases that
# occurred in year t, then store beta(t), its 95% CI, gamma(t), R-squared(t),
# total population that year, total cases that year, and the running
# effective total of all cases observed so far.
# ---------------------------------------------------------------------------
df = df_primary.copy()
years = list(range(1969, 2025))

plot_start_year = 1969
betas, beta_err_lower, beta_err_upper = [], [], []
intercepts, intercept_err_lower, intercept_err_upper = [], [], []
r2_values = []
total_populations, yearly_cases, effective_total_cases = [], [], []

running_total_cases = baseline_cases

for year in years:
    # Subset to this year's cases only.
    df_year = df[df['Year'] == year]

    # Aggregate to (State, State_pop) -> case count.
    grouped = (
        df_year.groupby(['State', 'State_pop'])
        .agg(case_count=('CaseID', 'count'))
        .reset_index()
    )

    grouped = grouped[grouped['case_count'] > 0]

    yearly_case_sum = grouped['case_count'].sum()
    running_total_cases += yearly_case_sum

    if len(grouped) > 1:
        # Enough states to fit a line. Log10 both axes and run OLS.
        X_log = np.log10(grouped['State_pop'].values)
        y_log = np.log10(grouped['case_count'].values)
        X_log_const = sm.add_constant(X_log)
        model = sm.OLS(y_log, X_log_const).fit()

        intercept, beta = model.params
        r2 = model.rsquared

        conf_int = model.conf_int()
        intercept_ci_lower, intercept_ci_upper = conf_int[0]
        beta_ci_lower, beta_ci_upper = conf_int[1]

        total_pop = grouped['State_pop'].sum()

        # Store this year's results. We push the CI as (lower-delta,
        # upper-delta) so the errorbar plot below can consume it directly.
        intercepts.append(intercept)
        intercept_err_lower.append(intercept - intercept_ci_lower)
        intercept_err_upper.append(intercept_ci_upper - intercept)
        betas.append(beta)
        beta_err_lower.append(beta - beta_ci_lower)
        beta_err_upper.append(beta_ci_upper - beta)
        r2_values.append(r2)
        total_populations.append(total_pop)
        yearly_cases.append(yearly_case_sum)
        effective_total_cases.append(running_total_cases)

        print(f"{year}: β = {beta:.4f} [{beta_ci_lower:.4f}, {beta_ci_upper:.4f}], "
              f"R² = {r2:.4f}, Yearly Cases = {yearly_case_sum}, Effective Total = {running_total_cases}")
    else:
        # Not enough states with cases this year to fit a line. Carry NaN
        # forward so the time series stays year-indexed and visually flags
        # the gap.
        betas.append(np.nan)
        beta_err_lower.append(np.nan)
        beta_err_upper.append(np.nan)
        intercepts.append(np.nan)
        intercept_err_lower.append(np.nan)
        intercept_err_upper.append(np.nan)
        r2_values.append(np.nan)
        total_populations.append(np.nan)
        yearly_cases.append(np.nan)
        effective_total_cases.append(running_total_cases)
        print(f"{year}: insufficient data")

years_to_plot = years

# ---------------------------------------------------------------------------
# Plot beta(t) with 95% CI error bars.
# ---------------------------------------------------------------------------
y_err = np.array([beta_err_lower, beta_err_upper])
plt.figure(figsize=(18, 12))
plt.plot(years_to_plot, betas, color='lightseagreen', linewidth=2, label='β-value Estimate')
plt.errorbar(
    years_to_plot,
    betas,
    yerr=y_err,
    fmt='o',
    ecolor='k',
    capsize=8,
    capthick=2,
    color='darkorange',
    elinewidth=2,
    label=r'$\beta$ with 95% CI'
)
plt.title(r'Scaling Exponent ($\beta$) of Annual NamUS Missing Person Cases vs State Population (1969–2024)', fontsize=28)
plt.xlabel('Year', fontsize=24)
plt.ylabel(r'Estimated Scaling Exponent Value ($\beta$)', fontsize=24)
plt.xticks(years_to_plot, rotation=65, fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
max_beta = np.nanmax(betas)
min_beta = np.nanmin(betas)
plt.ylim(min_beta - .45, max_beta + .35)
plt.figtext(0.975, 0.105, f"\n\nBaseline cases prior to 1969: {baseline_cases}", ha="right", fontsize=14)
plt.tight_layout()
plt.savefig(r'/Users/ayushsarkar/missing_persons/missing_persons/plots/regressions/temporal/states/[1969-2024]_regression_ts_states_annual.png', dpi=1200, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# Identify the best and worst R-squared years (ignoring NaN entries) so we
# can show the corresponding scatters side by side below.
# ---------------------------------------------------------------------------
r2_array = np.array(r2_values)
valid_r2_indices = np.where(~np.isnan(r2_array))[0]
best_year_idx = valid_r2_indices[np.argmax(r2_array[valid_r2_indices])]
worst_year_idx = valid_r2_indices[np.argmin(r2_array[valid_r2_indices])]
best_year = years_to_plot[best_year_idx]
worst_year = years_to_plot[worst_year_idx]


def plot_regression_scatter(ax, year, df, title_prefix=''):
    """Draw the (log pop, log cases) scatter and fit for one year.

    Used to show what tight vs. loose annual fits actually look like. Pulls
    cases that occurred in `year`, aggregates them to (State, State_pop) ->
    case count, log-transforms both axes, fits OLS, and renders the
    regression line plus a 95% CI band.

    Falls back to a "Not enough data" title if fewer than two states have
    cases in `year` (the regression would be undefined).
    """
    df_year = df[df['Year'] == year]
    grouped = (
        df_year.groupby(['State', 'State_pop'])
        .agg(case_count=('CaseID', 'count'))
        .reset_index()
    )
    grouped = grouped[(grouped['State_pop'] > 0) & (grouped['case_count'] > 0)]
    grouped['log_pop'] = np.log10(grouped['State_pop'])
    grouped['log_cases'] = np.log10(grouped['case_count'])
    grouped = grouped[np.isfinite(grouped['log_pop']) & np.isfinite(grouped['log_cases'])]
    X = grouped['log_pop'].values
    y = grouped['log_cases'].values
    if len(X) < 2:
        ax.set_title(f"{title_prefix}{year}: Not enough data", fontsize=16)
        return
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    y_pred = model.predict(X_const)
    intercept, beta = model.params
    conf_int = model.conf_int()
    intercept_ci_lower, intercept_ci_upper = conf_int[0]
    beta_ci_lower, beta_ci_upper = conf_int[1]
    ax.scatter(X, y, color='steelblue', alpha=0.7, label='Data Points')
    ax.plot(X, y_pred, color='crimson', linewidth=2, label='Regression Line')
    X_range = np.linspace(X.min(), X.max(), 200)
    y_lower = intercept + beta_ci_lower * X_range
    y_upper = intercept + beta_ci_upper * X_range
    ax.fill_between(X_range, y_lower, y_upper, color='crimson', alpha=0.2, label='95% CI')
    ax.set_title(f"{title_prefix}{year} (R² = {model.rsquared:.3f})", fontsize=22)
    ax.set_xlabel(r'$\log_{10}$(Population)', fontsize=24)
    ax.set_ylabel(r'$\log_{10}$(Cases)', fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend(title=(
        f"β = {beta:.3f}; CI: [{beta_ci_lower:.3f}, {beta_ci_upper:.3f}]\n"
        f"γ = {intercept:.3f}; CI: [{intercept_ci_lower:.3f}, {intercept_ci_upper:.3f}]"
    ), fontsize=14, title_fontsize=16)


fig, axes = plt.subplots(1, 2, figsize=(18, 8))
plot_regression_scatter(axes[0], worst_year, df, title_prefix='Worst Year (Annual): ')
plot_regression_scatter(axes[1], best_year, df, title_prefix='Best Year (Annual): ')
plt.tight_layout()
plt.savefig(r'/Users/ayushsarkar/missing_persons/missing_persons/plots/regressions/temporal/states/[1969-2024]_regression_comparison_ts_states_annual.png', dpi=1200, bbox_inches='tight')
plt.show()


def plot_r2_timeseries(years, r2_values):
    """Plot the R-squared(t) curve with the best and worst years highlighted.

    Complement to the beta(t) plot above -- a low R-squared in a year
    typically indicates that a small number of high-leverage states
    (California, Texas, Florida) are dominating that year's fit.
    """
    r2_array = np.array(r2_values)
    plt.figure(figsize=(14, 6))
    plt.plot(years, r2_array, marker='o', color='darkgreen', linewidth=3.5, label=r'$R^2$ value')
    if np.any(~np.isnan(r2_array)):
        best_idx = np.nanargmax(r2_array)
        worst_idx = np.nanargmin(r2_array)
        plt.scatter(years[best_idx], r2_array[best_idx], color='darkorange', s=100, label=f'Best R²: {years[best_idx]}')
        plt.scatter(years[worst_idx], r2_array[worst_idx], color='crimson', s=100, label=f'Worst R²: {years[worst_idx]}')
    plt.title(r'Time Series of $R^2$ Values (1969–2024) for $\beta$ (Annual Cases)', fontsize=32)
    plt.xlabel('Year', fontsize=28)
    plt.ylabel(r'$R^2$ Value', fontsize=28)
    plt.xticks(years, rotation=65, fontsize=16)
    plt.yticks(fontsize=24)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.6)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(r'/Users/ayushsarkar/missing_persons/missing_persons/plots/regressions/temporal/states/[1969-2024]_r2_ts_states_annual.png', dpi=1200, bbox_inches='tight')
    plt.show()


plot_r2_timeseries(years, r2_array)
