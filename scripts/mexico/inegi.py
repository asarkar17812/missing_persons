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
    if pd.isna(s):
        return s

    # Normalize accents
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ASCII', 'ignore').decode('utf-8')
    s = s.upper().strip()

    # Remove punctuation and extra spaces
    s = re.sub(r'[^A-Z ]', '', s)
    s = re.sub(r'\s+', ' ', s)

    # Canonical Mexican state names
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



df_inegi = pd.read_csv(r'source/mexico_missing_persons/data.csv', dtype=str)

# def print_unknown_confidential_counts(df):
#     for col in df.columns:
#         count = df[col].isin(['UNKNOWN', 'CONFIDENTIAL', 'MISSING']).sum()
#         total = df[col].size
#         percent = (count / total) * 100 if total > 0 else 0
#         print(f"{col}: (Count: {count}, Total: {total}); {percent:.2f}%")

# print_unknown_confidential_counts(df_inegi)
# VICTIM_ID: (Count: 0, Total: 129830); 0.00%
# ORIGIN_AGENCY: (Count: 0, Total: 129830); 0.00%
# DATE_OF_BIRTH: (Count: 76997, Total: 129830); 59.31%
# SEX: (Count: 48105, Total: 129830); 37.05%
# DATE_OF_INCIDENCE: (Count: 55991, Total: 129830); 43.13%
# DATE_OF_REPORT: (Count: 53904, Total: 129830); 41.52%
# VICTIM_STATUS: (Count: 124808, Total: 129830); 96.13%
# STATE_ID: (Count: 0, Total: 129830); 0.00%
# STATE: (Count: 2936, Total: 129830); 2.26%
# MUNICIPALITY_ID: (Count: 0, Total: 129830); 0.00%
# MUNICIPALITY: (Count: 53132, Total: 129830); 40.92%

# def plot_missing_value_correlation(df, replace_special=True, special_values=None):

#     # Replace special placeholder strings with np.nan
#     if replace_special:
#         if special_values is None:
#             special_values = ['UNKNOWN']
#         df = df.replace(special_values, np.nan)

#     # Plot missing value correlation heatmap
#     msno.heatmap(df)
#     plt.title("Missing Value Correlation Heatmap", fontsize=24)
#     plt.tight_layout()
#     # plt.savefig(r'F:\dsl_CLIMA\projects\submittable\missing persons\plots\mexico\missingValue_correlation_matrix.png', dpi=1200, bbox_inches='tight')
#     plt.show()

# plot_missing_value_correlation(df_inegi)

def plot_valid_entries_choropleth_shp(
    df,
    state_col='STATE',
    columns_to_check=None,
    special_values=None,
    shapefile_path=None,
    shapefile_state_col='name'
):
    if shapefile_path is None:
        raise ValueError("Please provide the path to a .shp file.")

    if special_values is None:
        special_values = ['UNKNOWN', 'CONFIDENTIAL']
    if columns_to_check is None:
        columns_to_check = [col for col in df.columns if col != state_col]

    # Normalize dataframe states
    df = df.copy()
    df[state_col] = df[state_col].apply(normalize_state_name)
    df[columns_to_check] = df[columns_to_check].replace(special_values, pd.NA)

    # Count valid entries per state
    valid_counts = (
        df.groupby(state_col)[columns_to_check]
        .apply(lambda g: g.notna().sum().sum())
        .reset_index()
    )
    valid_counts.columns = [state_col, 'valid_count']

    # Load shapefile
    gdf = gpd.read_file(shapefile_path)

    # Normalize shapefile state names
    gdf[shapefile_state_col] = gdf[shapefile_state_col].apply(normalize_state_name)

    # Merge data with shapefile
    merged = gdf.merge(valid_counts, left_on=shapefile_state_col, right_on=state_col, how='left')

    # Find minimum positive valid count (excluding zeros and NaNs)
    min_valid = merged['valid_count'][merged['valid_count'] > 0].min()

    # Fill missing counts with minimum positive value to avoid log(0)
    merged['valid_count'] = merged['valid_count'].fillna(min_valid)

    # Determine dynamic major ticks for colorbar
    vmin = min_valid
    vmax = merged['valid_count'].max()
    min_exp = int(np.floor(np.log10(vmin)))
    max_exp = int(np.ceil(np.log10(vmax)))
    major_ticks = [10**i for i in range(3, 5)]

    # Plot choropleth
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

    # Format colorbar
    cbar = ax.get_figure().axes[-1]
    cbar.set_xticks(major_ticks)  # major ticks up to data max
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


plot_valid_entries_choropleth_shp(
    df=df_inegi,
    state_col='STATE',
    columns_to_check=[
        'DATE_OF_BIRTH',
        'SEX',
        'DATE_OF_INCIDENCE',
        'DATE_OF_REPORT',
        'VICTIM_STATUS'
    ],
    special_values=['UNKNOWN'],
    shapefile_path=r'source/shape files/mexico/mexican-states.shp',
    shapefile_state_col='name'
)