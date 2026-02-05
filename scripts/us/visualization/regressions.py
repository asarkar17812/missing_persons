import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

df_primary = pd.read_csv(r'export/mp_term.csv')
df_primary['DisappearanceDate'] = pd.to_datetime(df_primary['DisappearanceDate'])

df_primary = df_primary[
    (df_primary['DisappearanceDate'] > pd.to_datetime('1968-12-31')) &
    (df_primary['DisappearanceDate'] < pd.to_datetime('2025-01-01'))
]

df_counties = df_primary.groupby('FIPS').agg(
    Case_Count=('CaseID','count'),
    County_Title=('County','last'),
    County_pop=('County_pop','last')
)

df_msa = df_primary.groupby('MSA Code').agg(
    Case_Count=('CaseID', 'count'),      # count the number of rows per MSA
    MSA_Title=('MSA Title', 'last'),   # take the last MSA Title
    CBSA_Type=('CBSA Type', 'last'),    # take the last CBSA Type 
    MSA_pop=('MSA_pop', 'last')
).reset_index().copy()

df_csa = df_primary.groupby('CSA Code').agg(
    Case_Count=('CaseID','count'),
    CSA_Title=('CSA Title', 'first'),
    CSA_Type=('CSA Type', 'first'),
    CSA_pop=('CSA_pop','last')
).reset_index().copy()

df_state = df_primary.groupby('State').agg(
    Case_Count=('CaseID', 'count'),
    State_Title=('State','first'),
    State_pop=('State_pop', 'last')
)

# --- Clean and prepare data
df_state = df_state[(df_state['Case_Count'] > 0) & (df_state['State_pop'] > 0)].dropna()
df_state['log_cases'] = np.log10(df_state['Case_Count'])
df_state['log_pop'] = np.log10(df_state['State_pop'])

df_counties = df_counties[(df_counties['Case_Count'] > 0) & (df_counties['County_pop'] > 0)].dropna()
df_counties['log_cases'] = np.log10(df_counties['Case_Count'])
df_counties['log_pop'] = np.log10(df_counties['County_pop'])

df_counties = df_counties[(df_counties['Case_Count'] > 0) & (df_counties['County_pop'] > 0)].dropna()
df_counties['log_cases'] = np.log10(df_counties['Case_Count'])
df_counties['log_pop'] = np.log10(df_counties['County_pop'])

df_csa = df_csa[(df_csa['Case_Count'] > 0) & (df_csa['CSA_pop'] > 0)].dropna()
df_csa['log_cases'] = np.log10(df_csa['Case_Count'])
df_csa['log_pop'] = np.log10(df_csa['CSA_pop'])

df_msa = df_msa[(df_msa['Case_Count'] > 0) & (df_msa['MSA_pop'] > 0)].dropna()
df_msa['log_cases'] = np.log10(df_msa['Case_Count'])
df_msa['log_pop'] = np.log10(df_msa['MSA_pop'])

# Subsets
datasets = {
    'States': df_state,
    'Counties': df_counties,
    'CSAs': df_csa,
    'CBSAs': df_msa,
    'MSAs': df_msa[df_msa['CBSA_Type'] == 'MSA'],
    'MicroSAs': df_msa[df_msa['CBSA_Type'] == 'MicroSA'],
}

# --- Plot setup
fig, axes = plt.subplots(2, 3, figsize=(24, 8), sharey=True)
axes = axes.flatten()

for ax, (title, df) in zip(axes, datasets.items()):
    # Fit log-log regression
    X = sm.add_constant(df['log_pop'])
    y = df['log_cases']
    model = sm.OLS(y, X).fit()

    intercept, slope = model.params
    conf_int = model.conf_int(alpha=0.05)
    intercept_ci = conf_int.loc['const'].values
    slope_ci = conf_int.loc['log_pop'].values
    r2 = model.rsquared

    # Line and confidence interval
    x_vals = np.linspace(0, df['log_pop'].max(), 200)
    x_vals_const = sm.add_constant(x_vals)
    y_vals = model.predict(x_vals_const)
    preds_ci = model.get_prediction(x_vals_const).summary_frame(alpha=0.05)

    # Calculate mean and median points
    mean_log_pop = df['log_pop'].mean()
    mean_log_cases = df['log_cases'].mean()
    median_log_pop = df['log_pop'].median()
    median_log_cases = df['log_cases'].median()

    # Plotting
    ax.scatter(df['log_pop'], df['log_cases'], color='steelblue', alpha=0.7, label='$log_{10}$(Number of Cases)')
    ax.plot(x_vals, y_vals, color='darkred', linewidth=2, label='Regression line')
    ax.fill_between(x_vals, preds_ci['mean_ci_lower'], preds_ci['mean_ci_upper'],
                    color='lightcoral', alpha=0.3, label='95% CI band')

    ax.scatter(mean_log_pop, mean_log_cases, color='green', s=100, edgecolor='black', label='Mean point', zorder=5)
    ax.scatter(median_log_pop, median_log_cases, color='purple', s=100, edgecolor='black', label='Median point', zorder=5)

    # Total cases as sum, not count
    total_cases = df['Case_Count'].sum()
    regression_label = (
        f"β = {slope:.3f} [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]\n"
        f"γ = {intercept:.3f} [{intercept_ci[0]:.3f}, {intercept_ci[1]:.3f}]\n"
        f"$R^2$ = {r2:.3f}\n"
        f"Total cases: {total_cases:,.0f}"  # formatted with commas
    )
    ax.plot([], [], ' ', label=regression_label)

    ax.set_title(title, fontsize=28)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True)
    ax.set_xlim(left=2)
    ax.set_ylim(bottom=0)

axes[0].set_ylabel('log(NamUS Case Counts)\n[1969-2024]', fontsize=20)
axes[3].set_ylabel('log(NamUS Case Counts)\n[1969-2024]', fontsize=20)
axes[0].set_xlabel('log(State Population)', fontsize=20)
axes[1].set_xlabel('log(County Population)', fontsize=20)
axes[2].set_xlabel('log(CSA Population)', fontsize=20)
axes[3].set_xlabel('log(CBSA Population)', fontsize=20)
axes[4].set_xlabel('log(MSA Population)', fontsize=20)
axes[5].set_xlabel('log(MicroSA Population)', fontsize=20)

fig.suptitle('Scaling Exponent (β) of NamUs Missing Persons Cases vs GEOID Population [1969-2024]', fontsize=28)
plt.tight_layout(rect=[0.025, 0, 1, 1], h_pad=0.01)
plt.show()

fig.savefig(r'plots/regressions/cumulative/[1969-2024]regressions.png', dpi=1500, bbox_inches='tight')