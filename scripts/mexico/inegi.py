"""
Mexico INEGI diagnostic and choropleth visualizations.

This script handles the Mexican-side equivalents of the U.S. data
diagnostics and choropleths. Three things live here:

    1. A helper (`print_unknown_confidential_counts`) that reports per-column
       missingness rates for the INEGI dump. This is what motivates the
       methodological choice to restrict Mexican analyses to cumulative
       state-level aggregates only.
    2. A choropleth helper (`plot_valid_entries_choropleth_shp`) that maps
       the number of *valid* INEGI entries per state, log-color-scaled.
       Used as a sanity check that the missingness budget is roughly
       spatially uniform rather than concentrated in a handful of states.
    3. A choropleth helper (`plot_population_choropleth_shp`) for the
       2025 INEGI state-population projections, same projection and
       color scale as the case map. Holding the two maps side by side
       shows where the case distribution departs from the population
       distribution (i.e. which states sit above or below the scaling
       regression line).

The plotting functions are gated behind `if False:`-style call sites at
the bottom; uncomment whichever one you want to render.

Run as:
    python scripts/mexico/inegi.py
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FuncFormatter
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import missingno as msno
import unicodedata
import re


def normalize_state_name(s):
    """Normalize a Mexican state name to a canonical upper-case ASCII form.

    Removes accents (NFKD decomposition + ASCII filter), upper-cases the
    result, strips punctuation, and applies a small dictionary of canonical
    aliases (e.g. "Veracruz de Ignacio de la Llave" -> "VERACRUZ", "Ciudad
    de Mexico" / "Distrito Federal" -> "CDMX") so that the INEGI cases
    file, the population file, and the OSM shapefile all line up on the
    same state names when we join them.
    """
    if pd.isna(s):
        return s

    # NFKD decomposition splits combined characters (n-tilde, e-acute) into
    # base + combining-accent; the ASCII filter then drops the accents.
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ASCII', 'ignore').decode('utf-8')
    s = s.upper().strip()

    # Strip anything that isn't a letter or whitespace, then collapse runs
    # of whitespace to single spaces.
    s = re.sub(r'[^A-Z ]', '', s)
    s = re.sub(r'\s+', ' ', s)

    # Canonical aliases for INEGI/OSM spelling differences.
    canonical_map = {
        'VERACRUZ DE IGNACIO DE LA LLAVE': 'VERACRUZ',
        'MICHOACAN DE OCAMPO': 'MICHOACAN',
        'COAHUILA DE ZARAGOZA': 'COAHUILA',
        'CIUDAD DE MEXICO': 'CDMX',
        'DISTRITO FEDERAL': 'CDMX',
        'ESTADO DE MEXICO': 'MEXICO',
        'BAJA CALIFORNIA NORTE': 'BAJA CALIFORNIA',
        'BAJA CALIFORNIA SUR': 'BAJA CALIFORNIA SUR'
    }

    return canonical_map.get(s, s)


# ---------------------------------------------------------------------------
# Load the INEGI cases dump and the cleaned population panel.
# ---------------------------------------------------------------------------
df_inegi = pd.read_csv(r'/Users/ayushsarkar/missing_persons/missing_persons/source/mexico_missing_persons/data.csv', dtype=str)
df_poblacion = pd.read_csv(r'/Users/ayushsarkar/missing_persons/missing_persons/export/poblacion.csv', dtype=str)


def print_unknown_confidential_counts(df):
    """Print per-column counts of 'CENSORED' tokens.

    The intended audience is the methodology section -- this is the
    diagnostic that quantifies how much of each INEGI column is unusable.
    """
    for col in df.columns:
        count = df[col].isin(['CENSORED']).sum()
        total = df[col].size
        percent = (count / total) * 100 if total > 0 else 0
        print(f"{col}: (Count: {count}, Total: {total}); {percent:.2f}%")


# Diagnostic subset used during development: records whose origin agency
# appears exactly once in the dataset *and* whose municipality is unknown.
# These are the long-tail edge cases the cleaning pipeline has to handle.
df_filtered = df_inegi[
    (df_inegi['ORIGIN_AGENCY'].map(df_inegi['ORIGIN_AGENCY'].value_counts()) == 1) &
    (df_inegi['MUNICIPALITY'] == "UNKNOWN")
]

print(df_inegi['ORIGIN_AGENCY'])


# Per-column missingness rates from a representative July 2025 INEGI dump.
# Combined (Confidential + Unknown):
# VICTIM_ID: 0.00%
# ORIGIN_AGENCY: 0.00%
# DATE_OF_BIRTH: 59.31%
# SEX: 37.05%
# DATE_OF_INCIDENCE: 43.13%
# DATE_OF_REPORT: 41.52%
# VICTIM_STATUS: 96.13%
# STATE_ID: 0.00%
# STATE: 2.26%
# MUNICIPALITY_ID: 0.00%
# MUNICIPALITY: 40.92%
#
# Unknown only:
# DATE_OF_BIRTH: 22.53%
# SEX: 0.28%
# DATE_OF_INCIDENCE: 6.35%
# DATE_OF_REPORT: 4.74%
# MUNICIPALITY: 4.15%
#
# Confidential only:
# DATE_OF_BIRTH: 36.78%
# SEX: 36.78%
# DATE_OF_INCIDENCE: 36.78%
# DATE_OF_REPORT: 36.78%
# VICTIM_STATUS: 36.78%
# MUNICIPALITY: 36.78%
# The Confidential-only rates being identical across columns is the
# signature of redaction-as-a-record: when a case is flagged Confidential,
# all of its sensitive fields are blanked simultaneously.


def plot_valid_entries_choropleth_shp(
    df,
    state_col='STATE',
    columns_to_check=None,
    special_values=None,
    shapefile_path=None,
    shapefile_state_col='name'
):
    """Render a Mexican-states choropleth of valid (non-missing) INEGI entries.

    For each state, counts the number of non-missing values across
    `columns_to_check`. `special_values` is the list of tokens that count
    as "missing" (default: UNKNOWN + CONFIDENTIAL). Output is log-color-
    scaled because the count distribution is heavy-tailed -- a handful of
    large states dominate.

    Parameters mirror the U.S. choropleth helper so the two pipelines have
    a consistent interface.
    """
    if shapefile_path is None:
        raise ValueError("Please provide the path to a .shp file.")

    if special_values is None:
        special_values = ['UNKNOWN', 'CONFIDENTIAL']
    if columns_to_check is None:
        columns_to_check = [col for col in df.columns if col != state_col]

    # Normalize state names so the join against the OSM shapefile succeeds.
    df = df.copy()
    df[state_col] = df[state_col].apply(normalize_state_name)
    df[columns_to_check] = df[columns_to_check].replace(special_values, pd.NA)

    # Count "valid entries" = non-NA cells across all checked columns.
    valid_counts = (
        df.groupby(state_col)[columns_to_check]
        .apply(lambda g: g.notna().sum().sum())
        .reset_index()
    )
    valid_counts.columns = [state_col, 'valid_count']

    gdf = gpd.read_file(shapefile_path)
    gdf[shapefile_state_col] = gdf[shapefile_state_col].apply(normalize_state_name)

    merged = gdf.merge(valid_counts, left_on=shapefile_state_col, right_on=state_col, how='left')

    # Replace zeros and NaNs with the smallest positive count so the log
    # color scale doesn't blow up.
    min_valid = merged['valid_count'][merged['valid_count'] > 0].min()
    merged['valid_count'] = merged['valid_count'].fillna(min_valid)

    # Pre-compute colorbar tick positions (powers of 10 spanning the data).
    vmin = min_valid
    vmax = merged['valid_count'].max()
    min_exp = int(np.floor(np.log10(vmin)))
    max_exp = int(np.ceil(np.log10(vmax)))
    major_ticks = [10**i for i in range(3, 5)]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    merged.plot(
        column='valid_count',
        ax=ax,
        cmap='viridis',
        edgecolor='black',
        linewidth=0.6,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        legend=True,
        legend_kwds={
            'label': 'Valid Cumulative INEGI Case Count (log scaled)',
            'orientation': 'horizontal',
            'fraction': 0.05,
            'pad': 0.02
        }
    )

    # Format the colorbar with explicit power-of-10 tick labels.
    cbar = ax.get_figure().axes[-1]
    cbar.set_xticks(major_ticks)
    cbar.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x))}}}$"))
    cbar.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
    cbar.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: ''))
    cbar.tick_params(axis='x', labelsize=14)
    cbar.set_xlabel('Valid Cumulative INEGI Case Count (log scaled)', fontsize=18)

    ax.set_title("Valid INEGI Missing Persons Cases by Mexican State (Log Scaled)", fontsize=24)
    ax.axis('off')
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig('plots/mexico/cases_choropleth.png', dpi=1500, bbox_inches='tight')
    plt.show()


# Example call -- uncomment to render the cases choropleth.
# plot_valid_entries_choropleth_shp(
#     df=df_inegi,
#     state_col='STATE',
#     columns_to_check=[
#         'DATE_OF_BIRTH',
#         'SEX',
#         'DATE_OF_INCIDENCE',
#         'DATE_OF_REPORT',
#         'VICTIM_STATUS'
#     ],
#     special_values=['UNKNOWN'],
#     shapefile_path=r'source/shape files/mexico/mexican-states.shp',
#     shapefile_state_col='name'
# )


def plot_population_choropleth_shp(
    df,
    state_col='STATE',
    pop_col='State_pop',
    shapefile_path=None,
    shapefile_state_col='name',
    log_scale=True
):
    """Render a Mexican-states choropleth of state populations.

    Same projection and color scale as the cases choropleth above, so the
    two maps can be visually compared. Use `log_scale=False` for a linear
    color ramp; default is log because Mexican state populations span ~3
    orders of magnitude (Colima ~750k, Estado de Mexico ~17M).
    """
    if shapefile_path is None:
        raise ValueError("Please provide the path to a .shp file.")

    df = df.copy()
    df[state_col] = df[state_col].apply(normalize_state_name)
    df[pop_col] = pd.to_numeric(df[pop_col], errors='coerce')
    df = df.dropna(subset=[pop_col])

    gdf = gpd.read_file(shapefile_path)
    gdf[shapefile_state_col] = gdf[shapefile_state_col].apply(normalize_state_name)

    merged = gdf.merge(
        df[[state_col, pop_col]],
        left_on=shapefile_state_col,
        right_on=state_col,
        how='left'
    )

    # Choose a log or linear color norm based on the caller's preference.
    vmin = merged[pop_col].min()
    vmax = merged[pop_col].max()

    if log_scale:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        legend_label = '2025 Mexican State Population (Log Scaled)'
    else:
        norm = None
        legend_label = '2025 Mexican State Population'

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    merged.plot(
        column=pop_col,
        ax=ax,
        cmap='viridis',
        edgecolor='black',
        linewidth=0.6,
        norm=norm,
        legend=True,
        legend_kwds={
            'label': legend_label,
            'orientation': 'horizontal',
            'fraction': 0.05,
            'pad': 0.02
        }
    )

    # Log-scale colorbar formatting: pin major ticks at powers of 10 within
    # the data range, and label them as 10^n for legibility.
    if log_scale:
        cbar = ax.get_figure().axes[-1]
        min_exp = int((np.log10(1000000)))
        max_exp = int((np.log10(vmax)))
        major_ticks = [10**i for i in range(min_exp, max_exp + 1)]

        cbar.set_xticks(major_ticks)
        cbar.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x))}}}$")
        )
        cbar.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
        )
        cbar.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: ''))
        cbar.tick_params(axis='x', labelsize=14)
        cbar.set_xlabel(legend_label, fontsize=18)

    ax.set_title(
        "2025 Mexican State Population Choropleth (Log Scaled)",
        fontsize=24
    )
    ax.axis('off')

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(
        '/Users/ayushsarkar/missing_persons/missing_persons/plots/mexico/state_population_choropleth.png',
        dpi=1500,
        bbox_inches='tight'
    )
    plt.show()


# Example call -- uncomment to render the population choropleth.
# plot_population_choropleth_shp(
#     df=df_poblacion,
#     state_col='entidad',
#     pop_col='Total',
#     shapefile_path=r'/Users/ayushsarkar/missing_persons/missing_persons/source/shape files/mexico/mexican-states.shp',
#     shapefile_state_col='name'
# )
