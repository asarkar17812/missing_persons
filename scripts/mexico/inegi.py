import pandas as pd 
import geopandas as gpd
import numpy as np 
import matplotlib.pyplot as plt
import missingno as msno
from datetime import datetime

df_inegi = pd.read_csv(r'F:\dsl_CLIMA\projects\submittable\missing persons\source\mexico_missing_persons\data.csv', dtype=str)

def print_unknown_confidential_counts(df):
    for col in df.columns:
        count = df[col].isin(['UNKNOWN', 'CONFIDENTIAL', 'MISSING']).sum()
        total = df[col].size
        percent = (count / total) * 100 if total > 0 else 0
        print(f"{col}: (Count: {count}, Total: {total}); {percent:.2f}%")

print_unknown_confidential_counts(df_inegi)
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

def plot_missing_value_correlation(df, replace_special=True, special_values=None):

    # Replace special placeholder strings with np.nan
    if replace_special:
        if special_values is None:
            special_values = ['UNKNOWN']
        df = df.replace(special_values, np.nan)

    # Plot missing value correlation heatmap
    msno.heatmap(df)
    plt.title("Missing Value Correlation Heatmap", fontsize=24)
    plt.tight_layout()
    plt.savefig(r'F:\dsl_CLIMA\projects\submittable\missing persons\plots\mexico\missingValue_correlation_matrix.png', dpi=1200, bbox_inches='tight')
    plt.show()

plot_missing_value_correlation(df_inegi)

df_temp = df_inegi.copy()
df_temp = df_temp[df_temp['STATE'] == 'UNKNOWN']
# print(df_temp)

def plot_valid_entries_choropleth_shp(
    df,
    state_col='STATE',
    columns_to_check=None,
    special_values=None,
    shapefile_path=None,
    shapefile_state_col='name'
):
    """
    Creates a static choropleth map of Mexican states showing number of valid (non-missing) entries.

    Parameters:
    - df (pd.DataFrame): Input dataset
    - state_col (str): Column in df identifying states (e.g. 'STATE')
    - columns_to_check (list): Columns to check for valid entries (non-'UNKNOWN')
    - special_values (list): List of strings to treat as missing (default: ['UNKNOWN', 'CONFIDENTIAL'])
    - shapefile_path (str): Path to the .shp file for Mexican state boundaries
    - shapefile_state_col (str): Column in shapefile with state names to match against df[state_col]
    """

    if shapefile_path is None:
        raise ValueError("Please provide the path to a .shp file (shapefile_path).")

    if special_values is None:
        special_values = ['UNKNOWN', 'CONFIDENTIAL']
    if columns_to_check is None:
        columns_to_check = [col for col in df.columns if col != state_col]

    df = df.copy()
    df[state_col] = df[state_col].astype(str).str.upper()

    # Replace special values with NA
    df[columns_to_check] = df[columns_to_check].replace(special_values, pd.NA)

    # Count valid entries per state
    valid_counts = df.groupby(state_col)[columns_to_check].apply(lambda g: g.notna().sum().sum()).reset_index()
    valid_counts.columns = [state_col, 'valid_count']

    # Load shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf[shapefile_state_col] = gdf[shapefile_state_col].astype(str).str.upper()

    # Merge with shapefile
    merged = gdf.merge(valid_counts, left_on=shapefile_state_col, right_on=state_col, how='left')
    merged['valid_count'] = merged['valid_count'].fillna(0)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    merged.plot(column='valid_count',
                ax=ax,
                legend=True,
                cmap='viridis',
                edgecolor='black',
                legend_kwds={'label': "Valid Entry Count", 'orientation': "vertical"})

    ax.set_title("Valid Entries by Mexican State", fontsize=15)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(r'F:\dsl_CLIMA\projects\submittable\missing persons\plots\mexico\cases_choropleth.png', dpi=1200, bbox_inches='tight')
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
    shapefile_path=r'F:\dsl_CLIMA\projects\submittable\missing persons\source\shape files\mexico\mexican-states.shp'
)