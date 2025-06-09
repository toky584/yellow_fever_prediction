# src/S1_data_preparation.py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import normalize_text # Assuming utils.py is in the same directory (src)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

EPIZOOTIC_RAW_FILE = BASE_DATA_DIR / "epizootic_cases.csv"
HUMAN_CASES_RAW_FILE = BASE_DATA_DIR / "number_case_data.csv" 
POPULATION_HISTORICAL_FILE = BASE_DATA_DIR / "brazil_municipalities_population.csv"
POPULATION_2024_FILE = BASE_DATA_DIR / "po_2024.xlsx"
PRIMATE_RICHNESS_FILE = BASE_DATA_DIR / "primate_richness.csv"
MUNICIPALITY_COORDS_FILE = BASE_DATA_DIR / "municipalities_coordinates.csv" # If needed for unique lat/lon per Region

OUTPUT_PREPARED_MAIN_DATA_FILE = PROCESSED_DATA_DIR / "S1_prepared_main_data.csv"

# --- Helper Functions ---

def load_and_clean_event_data(filepath, event_type_name,
                              year_col_raw, month_col_raw, region_col_raw,
                              lat_col_raw='lat', lon_col_raw='lon'):
    """
    Loads and performs initial cleaning on event-based data (human cases or epizootics).
    Returns a DataFrame with 'Year', 'Months', 'Region', 'lat', 'lon'.
    One row per event.
    """
    print(f"Loading raw {event_type_name} data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='iso-8859-1')
    except FileNotFoundError:
        print(f"Error: {event_type_name} file not found at {filepath}")
        return pd.DataFrame()

    rename_map = {
        year_col_raw: 'Year', month_col_raw: 'Months',
        region_col_raw: 'Region_Raw', # Keep raw for now, normalize later
        lat_col_raw: 'lat', lon_col_raw: 'lon'
    }
    cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=cols_to_rename, inplace=True)

    required_cols = ['Year', 'Months', 'Region_Raw', 'lat', 'lon']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: {event_type_name} data missing required columns: {missing}")
        return pd.DataFrame()

    df['Region'] = df['Region_Raw'].apply(normalize_text)
    df.dropna(subset=['Year', 'Months', 'Region', 'lat', 'lon'], inplace=True)

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    df['Months'] = pd.to_numeric(df['Months'], errors='coerce').astype('Int64')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df.dropna(subset=['Year', 'Months', 'lat', 'lon'], inplace=True) # Drop rows where conversion failed

    print(f"{event_type_name} data shape after initial cleaning: {df.shape}")
    return df[['Year', 'Months', 'Region', 'lat', 'lon']].copy()


def aggregate_monthly_events(event_df, count_col_name):
    """Aggregates events to get monthly counts per Year, Month, Region, lat, lon."""
    if event_df.empty:
        return pd.DataFrame(columns=['Year', 'Months', 'Region', 'lat', 'long', count_col_name])
    print(f"Aggregating {count_col_name} to monthly counts per location...")
    
    # Group by all identifying columns and count occurrences
    monthly_counts = event_df.groupby(['Year', 'Months', 'Region', 'lat', 'lon'])\
                             .size()\
                             .reset_index(name=count_col_name)
    print(f"Aggregated {count_col_name} shape: {monthly_counts.shape}")
    return monthly_counts


def load_and_process_population_standalone(historical_pop_file, pop_2024_file):
    """Loads and processes population data (historical and 2024 estimates) standalone."""
    # ... (This function remains the same as in the previous version of S1)
    all_pop_data = []
    # Load historical population
    try:
        pop_hist = pd.read_csv(historical_pop_file)
        pop_hist['Region'] = pop_hist['name_municipality'].apply(normalize_text)
        pop_hist.rename(columns={'year': 'Year', 'population': 'Population'}, inplace=True)
        all_pop_data.append(pop_hist[['Year', 'Region', 'Population']].copy())
        print(f"Loaded {len(pop_hist)} historical population records.")
    except FileNotFoundError:
        print(f"Warning: Historical population file not found at {historical_pop_file}.")

    # Load 2024 population estimates
    try:
        pop_2024_raw = pd.read_excel(pop_2024_file, skiprows=2)
        if 'Unnamed: 5' in pop_2024_raw.columns:
            pop_2024_raw = pop_2024_raw.drop(columns=['Unnamed: 5'])
        pop_2024_raw.rename(columns={'NOME DO MUNICÍPIO': 'Region', 'POPULAÇÃO ESTIMADA': 'Population'}, inplace=True)
        pop_2024_raw['Region'] = pop_2024_raw['Region'].apply(normalize_text)
        
        pop_2024_df_list = []
        for yr_estimate in [2023, 2024]: 
            temp_pop = pop_2024_raw[['Region', 'Population']].copy()
            temp_pop['Year'] = yr_estimate
            pop_2024_df_list.append(temp_pop)
        if pop_2024_df_list:
            pop_2024_final = pd.concat(pop_2024_df_list)
            all_pop_data.append(pop_2024_final)
            print(f"Loaded and prepared {len(pop_2024_raw)} population estimates for 2023/2024.")
    except FileNotFoundError:
        print(f"Warning: 2024 population file not found at {pop_2024_file}.")

    if not all_pop_data:
        return pd.DataFrame(columns=['Year', 'Region', 'Population'])

    combined_pop_df = pd.concat(all_pop_data).drop_duplicates(subset=['Year', 'Region'], keep='last')
    combined_pop_df['Population'] = pd.to_numeric(combined_pop_df['Population'], errors='coerce')
    return combined_pop_df


def add_primate_richness(main_df, primate_richness_filepath):
    """Loads primate richness and merges it to the main DataFrame based on 'Region'."""
    # ... (This function remains largely the same, ensure 'Region' keys match)
    if main_df.empty:
        return main_df
    print(f"Loading primate richness data from {primate_richness_filepath}...")
    try:
        primate_df = pd.read_csv(primate_richness_filepath)
    except FileNotFoundError:
        print(f"Warning: Primate richness file not found. 'SpRich' will be 0.")
        main_df['SpRich'] = 0
        return main_df

    # Assuming primate_df has 'Region' (or similar normalized name) and 'spRich'
    # Normalize region name in primate_df if it's not already done
    if 'Region' in primate_df.columns: # If primate_df has a 'Region' column directly
        primate_df['Region'] = primate_df['Region'].apply(normalize_text)
    elif 'muni.no' in primate_df.columns and 'COD_MUN_LPI' in primate_df.columns: # Example if muni.no needs mapping to Region
        # This part would require a mapping file or complex logic if region names aren't directly in primate_df
        print("Warning: Primate richness merging by 'muni.no' requires 'Region' name mapping or 'muni.no' in main_df.")
        # For now, try merging if 'Region' is present in primate_df
    else: # Fallback if no clear key
        print("Warning: Primate richness file has no clear 'Region' or 'muni.no' key for merging. 'SpRich' will be 0.")
        main_df['SpRich'] = 0
        return main_df

    if 'Region' in primate_df.columns and 'spRich' in primate_df.columns:
        # Assuming one richness value per region (static)
        primate_to_merge = primate_df[['Region', 'spRich']].drop_duplicates(subset=['Region'])
        main_df = pd.merge(main_df, primate_to_merge, on='Region', how='left')
    
    main_df['SpRich'].fillna(0, inplace=True)
    print(f"Main DataFrame shape after adding primate richness: {main_df.shape}")
    return main_df

def create_master_spatiotemporal_grid(aggregated_human_df, aggregated_epizootic_df, municipality_coords_filepath=None):
    """
    Creates a master grid of all unique Year, Month, Region, lat, lon combinations
    from aggregated human and epizootic data.
    """
    print("Creating master spatio-temporal grid...")
    
    # Combine unique locations (Region, lat, lon) from both human and epizootic data
    all_locations = []
    if not aggregated_human_df.empty and all(c in aggregated_human_df for c in ['Region', 'lat', 'lon']):
        all_locations.append(aggregated_human_df[['Region', 'lat', 'lon']].drop_duplicates())
    if not aggregated_epizootic_df.empty and all(c in aggregated_epizootic_df for c in ['Region', 'lat', 'lon']):
        all_locations.append(aggregated_epizootic_df[['Region', 'lat', 'lon']].drop_duplicates())

    if not all_locations: # If both are empty or lack coordinate info
        print("No location data from human or epizootic sources to build grid. Trying municipality coordinates file.")
        if municipality_coords_filepath and Path(municipality_coords_filepath).exists():
            try:
                coords_master = pd.read_csv(municipality_coords_filepath)
                coords_master.rename(columns={'Latitude':'lat', 'Longitude':'long', 'Municipality':'Region_raw_coords'}, inplace=True)
                coords_master['Region'] = coords_master['Region_raw_coords'].apply(normalize_text)
                unique_coords_df = coords_master[['Region', 'lat', 'long']].drop_duplicates()
                if unique_coords_df.empty:
                    print("Municipality coordinates file yielded no unique locations.")
                    return pd.DataFrame()
            except Exception as e:
                print(f"Error loading municipality coordinates file {municipality_coords_filepath}: {e}")
                return pd.DataFrame()
        else:
            print("No fallback municipality coordinates file provided or found.")
            return pd.DataFrame()
    else:
        unique_coords_df = pd.concat(all_locations).drop_duplicates(subset=['Region', 'lat', 'lon']).reset_index(drop=True)

    if unique_coords_df.empty:
        print("No unique coordinates found to build master grid.")
        return pd.DataFrame()

    # Determine overall year range
    min_year = 2000
    max_year = 2024 
    if not aggregated_human_df.empty and 'Year' in aggregated_human_df:
        min_year = min(min_year, int(aggregated_human_df['Year'].min()))
        max_year = max(max_year, int(aggregated_human_df['Year'].max()))
    if not aggregated_epizootic_df.empty and 'Year' in aggregated_epizootic_df:
        min_year = min(min_year, int(aggregated_epizootic_df['Year'].min()))
        max_year = max(max_year, int(aggregated_epizootic_df['Year'].max()))

    year_range = range(min_year, max_year + 1)
    month_range = range(1, 13)

    master_grid_list = []
    for year_val in year_range:
        for month_val in month_range:
            for _, coord_row in unique_coords_df.iterrows():
                master_grid_list.append({
                    'Year': year_val, 'Months': month_val,
                    'Region': coord_row['Region'],
                    'lat': coord_row['lat'], 'long': coord_row['long']
                })
    master_df = pd.DataFrame(master_grid_list)

    # Merge aggregated human cases
    if not aggregated_human_df.empty:
        master_df = pd.merge(master_df, aggregated_human_df,
                             on=['Year', 'Months', 'Region', 'lat', 'long'], how='left')
        master_df['Cases'].fillna(0, inplace=True)
    else:
        master_df['Cases'] = 0
        
    # Merge aggregated epizootic cases
    if not aggregated_epizootic_df.empty:
        master_df = pd.merge(master_df, aggregated_epizootic_df,
                             on=['Year', 'Months', 'Region', 'lat', 'long'], how='left')
        master_df['Epizootic_cases'].fillna(0, inplace=True)
    else:
        master_df['Epizootic_cases'] = 0
        
    print(f"Master spatio-temporal grid created with shape: {master_df.shape}")
    return master_df

# --- Main Execution ---
def main():
    print("--- S1: Starting Data Preparation ---")

    # 1. Load and pre-process raw epizootic event data
    raw_epizootics_df = load_and_clean_event_data(EPIZOOTIC_RAW_FILE, "epizootic",
                                                year_col_raw='ANO_OCOR', month_col_raw='MES_OCOR',
                                                region_col_raw='MUN_OCOR', lat_col_raw='lat', lon_col_raw='lon')
    aggregated_epizootics_df = aggregate_monthly_events(raw_epizootics_df, 'Epizootic_cases')

    # 2. Load and pre-process raw human case event data
    # Assuming human_cases_raw_file has 'Year', 'Months', 'Region', 'lat', 'long' per case
    # Your notebook pg 57 loads 'number_case_data.csv' which has Year, Months, Cases, Region, lat, long.
    # This is ALREADY aggregated if 'Cases' is a count for that Year, Month, Region, lat, long.
    # If 'number_case_data.csv' has one row per *event*, then aggregate_monthly_events is needed.
    # If 'number_case_data.csv' has 'Cases' as a pre-aggregated count for that specific row's Year/Month/Region/lat/lon:
    print(f"Loading human case data from {HUMAN_CASES_RAW_FILE}...")
    try:
        human_df_pre_agg = pd.read_csv(HUMAN_CASES_RAW_FILE)
        human_df_pre_agg['Region'] = human_df_pre_agg['Region'].apply(normalize_text)
        human_df_pre_agg.rename(columns={'long': 'lon'}, inplace=True, errors='ignore') # Ensure 'lon'
        # Ensure types
        human_df_pre_agg['Year'] = pd.to_numeric(human_df_pre_agg['Year'], errors='coerce').astype('Int64')
        human_df_pre_agg['Months'] = pd.to_numeric(human_df_pre_agg['Months'], errors='coerce').astype('Int64')
        human_df_pre_agg.dropna(subset=['Year', 'Months', 'Region', 'lat', 'lon'], inplace=True)
        if 'Cases' not in human_df_pre_agg.columns:
            print("Warning: 'Cases' column not in human data, assuming 1 case per row for aggregation.")
            human_df_pre_agg['Cases_temp_count'] = 1 # Temporary column for counting
            aggregated_human_df = aggregate_monthly_events(human_df_pre_agg, 'Cases')
            aggregated_human_df.drop(columns=['Cases_temp_count'], inplace=True, errors='ignore')
        else: # 'Cases' column already exists, assume it's the count for that row's unique Y/M/R/lat/lon
            aggregated_human_df = human_df_pre_agg[['Year', 'Months', 'Region', 'lat', 'lon', 'Cases']].copy()
            # Sum cases if there are multiple entries for the exact same Y,M,R,lat,lon
            aggregated_human_df = aggregated_human_df.groupby(['Year', 'Months', 'Region', 'lat', 'lon'])['Cases'].sum().reset_index()
        print(f"Processed aggregated human cases data shape: {aggregated_human_df.shape}")
    except FileNotFoundError:
        print(f"Error: Human cases file not found at {HUMAN_CASES_RAW_FILE}")
        aggregated_human_df = pd.DataFrame()


    # 3. Create Master Spatio-Temporal Grid and merge counts
    master_df = create_master_spatiotemporal_grid(aggregated_human_df, aggregated_epizootics_df, MUNICIPALITY_COORDS_FILE)
    if master_df.empty:
        print("Critical error: Master grid creation failed. Exiting.")
        return

    # 4. Load, process, and merge population data onto the master grid
    print("Processing and merging comprehensive population data onto master grid...")
    population_df = load_and_process_population_standalone(
        POPULATION_HISTORICAL_FILE,
        POPULATION_2024_FILE
    )
    if not population_df.empty:
        pop_to_merge = population_df[['Year', 'Region', 'Population']].drop_duplicates(subset=['Year', 'Region'])
        master_df = pd.merge(master_df, pop_to_merge, on=['Year', 'Region'], how='left')
        master_df.sort_values(by=['Region', 'Year', 'Months'], inplace=True)
        master_df['Population'] = master_df.groupby('Region')['Population'].apply(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        master_df['Population'] = master_df.groupby('Region')['Population'].ffill().bfill()
        master_df['Population'].fillna(0, inplace=True)
        print("Population data merged and interpolated onto master grid.")
    else:
        master_df['Population'] = 0
        print("Warning: Population data could not be loaded. 'Population' set to 0.")

    # 5. Add Primate Richness
    master_df = add_primate_richness(master_df, PRIMATE_RICHNESS_FILE)

    # 6. Define Spillover target variable
    if 'Cases' in master_df.columns:
        master_df["Spillover"] = master_df["Cases"].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)
    else:
        print("Error: 'Cases' column not found for defining 'Spillover' target. 'Spillover' set to 0.")
        master_df["Spillover"] = 0

    # 7. Final checks, column ordering, and save
    final_cols_s1 = ['Year', 'Months', 'Region', 'lat', 'long',
                     'Cases', 'Epizootic_cases', 'Population', 'SpRich', 'Spillover']
    for col in final_cols_s1:
        if col not in master_df.columns:
            master_df[col] = 0
            print(f"Warning: Column '{col}' was missing from master_df, added with default value 0.")
    master_df = master_df[final_cols_s1].copy()

    print(f"\nFinal prepared DataFrame for S1 head:\n{master_df.head()}")
    print(f"Shape of final prepared data for S1: {master_df.shape}")
    print(f"Missing values in S1 output before final save:\n{master_df.isnull().sum()}")
    for col in ['lat', 'long', 'Cases', 'Epizootic_cases', 'Population', 'SpRich', 'Spillover']:
        if master_df[col].isnull().any():
            master_df[col].fillna(0, inplace=True)

    master_df.to_csv(OUTPUT_PREPARED_MAIN_DATA_FILE, index=False)
    print(f"S1: Main data preparation complete. Output saved to {OUTPUT_PREPARED_MAIN_DATA_FILE}")
    print("--- S1: Finished ---")

if __name__ == "__main__":
    main()