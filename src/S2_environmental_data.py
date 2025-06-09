# src/S2_environmental_data.py
import pandas as pd
import numpy as np
import xarray as xr # For reading NetCDF files efficiently
from netCDF4 import Dataset # Alternative for NetCDF
from pathlib import Path
from tqdm import tqdm
from utils import haversine_vectorized # Not strictly needed if using simple distance for nearest grid

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
RAW_DATA_DIR = PROJECT_ROOT / "data" # Assuming environmental raw data is here

# Input file from S1
S1_PREPARED_DATA_FILE = PROCESSED_DATA_DIR / "S1_prepared_main_data.csv"

# Environmental data paths (adjust as needed)
# Temperature: Assuming a folder with multiple NetCDF files, one per period, or a single combined one
TEMPERATURE_DATA_DIR = RAW_DATA_DIR / "temperature"
# Example: CRU_mean_temperature_mon_0.5x0.5_global_2000_v4.03.nc

# Precipitation:
# Option 1: Pre-processed CSV
PRECIPITATION_CSV_FILE = RAW_DATA_DIR / "precipitation.csv"
# Option 2: Directory with TRMM HDF files (as on notebook pg 51-54)
RAINFALL_HDF_DIR = RAW_DATA_DIR / "rainfall" # e.g., contains 3B43.YYYYMMDD.7A.HDF files

# Output file from this script
OUTPUT_ENV_MERGED_DATA_FILE = PROCESSED_DATA_DIR / "S2_data_with_environment.csv"

# --- Helper Functions for Environmental Data ---

def load_and_process_temperature_data(temp_dir_path, target_years_months_coords):
    """
    Loads temperature data from NetCDF files, filters for Brazil,
    and prepares it for merging.
    (Adapted from notebook pages 8-10)
    """
    print(f"Loading and processing temperature data from {temp_dir_path}...")
    all_temp_data = []
    # Ensure target_years_months_coords has unique combinations to avoid redundant lookups
    unique_targets = target_years_months_coords[['Year', 'Months', 'lat', 'long']].drop_duplicates()

    # Your notebook shows loading multiple files or one large file.
    # This example assumes multiple NetCDF files in temp_dir_path, one per year or period.
    # If it's one large file, the loop structure needs to change.
    # For now, let's assume one file as on notebook page 8 for CRU 2000s.
    # You'll need to adapt this if you have many yearly/monthly NetCDF files.
    
    # Example: Processing a single CRU NetCDF file (adapt if multiple)
    # This logic needs to be robust to how your temperature files are organized.
    # The notebook (pg 9) shows a loop for files in `pattern = 'temperature'` directory
    
    temp_files = list(temp_dir_path.glob("*.nc"))
    if not temp_files:
        print(f"No NetCDF files found in {temp_dir_path}. Skipping temperature processing.")
        return pd.DataFrame(columns=['Year', 'Months', 'lat_env', 'lon_env', 'temp']) # Return empty with expected cols

    processed_files_count = 0
    for nc_file_path in temp_files:
        print(f"Processing temperature file: {nc_file_path.name}")
        try:
            # Using xarray is often more convenient for NetCDF
            with xr.open_dataset(nc_file_path) as ds:
                # Assuming 'tas' is temperature, 'time', 'lat', 'lon' are coordinates
                if not all(var in ds for var in ['tas', 'time', 'lat', 'lon']):
                    print(f"Warning: Skipping {nc_file_path.name}, missing required variables (tas, time, lat, lon).")
                    continue

                # Convert time to pandas datetime if necessary (xarray often handles this)
                # Ensure time dimension is compatible for year/month extraction
                # ds_time_pd = pd.to_datetime(ds['time'].values) # Example if manual conversion needed
                
                # Convert to DataFrame and melt for easier processing
                df_temp = ds['tas'].to_dataframe().reset_index()
                df_temp.rename(columns={'tas': 'temp', 'lat': 'lat_env', 'lon': 'lon_env'}, inplace=True)

                # Extract Year and Month
                df_temp['Year'] = pd.to_datetime(df_temp['time']).dt.year
                df_temp['Months'] = pd.to_datetime(df_temp['time']).dt.month
                
                # Filter for Brazil's approximate lat/lon bounds (as in notebook pg 9)
                # Make these bounds configurable if needed
                lat_min, lat_max = -33.75, 5.25
                lon_min, lon_max = -75.0, -34.0
                df_temp_brazil = df_temp[
                    (df_temp['lat_env'] >= lat_min) & (df_temp['lat_env'] <= lat_max) &
                    (df_temp['lon_env'] >= lon_min) & (df_temp['lon_env'] <= lon_max)
                ]
                all_temp_data.append(df_temp_brazil[['Year', 'Months', 'lat_env', 'lon_env', 'temp']])
                processed_files_count +=1
        except Exception as e:
            print(f"Error processing temperature file {nc_file_path.name}: {e}")
    
    if not all_temp_data:
        print("No temperature data successfully processed.")
        return pd.DataFrame(columns=['Year', 'Months', 'lat_env', 'lon_env', 'temp'])
        
    combined_temp_df = pd.concat(all_temp_data).drop_duplicates().reset_index(drop=True)
    print(f"Combined temperature data shape from {processed_files_count} files: {combined_temp_df.shape}")
    return combined_temp_df


def load_and_process_precipitation_data_csv(precip_csv_path):
    """
    Loads precipitation data from a pre-processed CSV.
    Assumes CSV has Year, Months, lat_env, lon_env, precipitation.
    (From notebook page 8 `precipitation = pd.read_csv('precipitation.csv')`)
    """
    print(f"Loading precipitation data from CSV: {precip_csv_path}...")
    try:
        df = pd.read_csv(precip_csv_path)
        # Ensure required columns and types
        required_cols = ['Year', 'Months', 'lat', 'lon', 'precipitation']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Precipitation CSV missing one of required columns: {required_cols}")
            return pd.DataFrame(columns=['Year', 'Months', 'lat_env', 'lon_env', 'precipitation'])
        df.rename(columns={'lat': 'lat_env', 'lon': 'lon_env'}, inplace=True)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
        df['Months'] = pd.to_numeric(df['Months'], errors='coerce').astype('Int64')
        print(f"Loaded precipitation data shape: {df.shape}")
        return df[['Year', 'Months', 'lat_env', 'lon_env', 'precipitation']]
    except FileNotFoundError:
        print(f"Error: Precipitation CSV file not found at {precip_csv_path}")
        return pd.DataFrame(columns=['Year', 'Months', 'lat_env', 'lon_env', 'precipitation'])

# Note: Processing TRMM HDF files (notebook pg 51-54) is more complex.
# It involves calculating lat/lon from grid indices and iterating through many files.
# For now, this script assumes either a pre-processed CSV or the CRU-style NetCDF for temperature.

def merge_env_data_to_main(main_df, env_df, env_var_name, lat_col_main='lat', lon_col_main='long',
                           lat_col_env='lat_env', lon_col_env='lon_env'):
    """
    Merges an environmental variable (temp or precip) to the main DataFrame
    by finding the nearest environmental grid cell for each main_df record
    for the same Year and Month.
    """
    if main_df.empty or env_df.empty:
        print(f"Warning: Main data or environmental data for {env_var_name} is empty. Skipping merge.")
        if env_var_name not in main_df.columns:
            main_df[env_var_name] = np.nan
        return main_df

    print(f"Merging {env_var_name} data...")
    main_df[env_var_name] = np.nan # Initialize column

    # For faster lookups, create a dictionary of environmental data per Year/Month
    env_data_grouped = {}
    for name, group in env_df.groupby(['Year', 'Months']):
        env_data_grouped[name] = group[[lat_col_env, lon_col_env, env_var_name]].values

    for index, row in tqdm(main_df.iterrows(), total=len(main_df), desc=f"Processing {env_var_name}"):
        year_val = row['Year']
        month_val = row['Months']
        target_lat = row[lat_col_main]
        target_lon = row[lon_col_main]

        if (year_val, month_val) in env_data_grouped:
            month_year_env_data = env_data_grouped[(year_val, month_val)]
            # month_year_env_data is now a numpy array: [[lat_e, lon_e, val_e], ...]
            
            if month_year_env_data.size > 0:
                env_lats = month_year_env_data[:, 0]
                env_lons = month_year_env_data[:, 1]
                env_values = month_year_env_data[:, 2]

                # Simple squared Euclidean distance for finding nearest grid cell
                # For large datasets or global scales, Haversine might be better but slower here.
                distances_sq = (env_lats - target_lat)**2 + (env_lons - target_lon)**2
                
                if distances_sq.size > 0:
                    nearest_idx = np.nanargmin(distances_sq) # Use nanargmin to handle potential NaNs in distances
                    if pd.notna(nearest_idx):
                         main_df.loc[index, env_var_name] = env_values[nearest_idx]
    
    # Handle cases where no match was found (e.g., fill with mean or a specific strategy)
    # For now, they remain NaN, will be handled in S3 or modeling stage if needed.
    missing_count = main_df[env_var_name].isnull().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} records could not be matched for {env_var_name} and remain NaN.")
        # Example: main_df[env_var_name].fillna(main_df[env_var_name].mean(), inplace=True)

    print(f"Finished merging {env_var_name}. Main data shape: {main_df.shape}")
    return main_df


# --- Main Execution ---
def main():
    print("--- S2: Starting Environmental Data Processing and Merging ---")

    # 1. Load main prepared data from S1
    try:
        main_df = pd.read_csv(S1_PREPARED_DATA_FILE)
        print(f"Loaded main data from S1 with shape: {main_df.shape}")
    except FileNotFoundError:
        print(f"Error: Main data file from S1 not found at {S1_PREPARED_DATA_FILE}. Exiting.")
        return

    # Create a subset of unique Year, Months, lat, long for efficient env data lookup
    target_coords_time = main_df[['Year', 'Months', 'lat', 'long']].drop_duplicates().reset_index(drop=True)

    # 2. Load, process, and merge Temperature data
    # Adapt the path or logic if you have a single combined temperature file.
    temp_df_processed = load_and_process_temperature_data(TEMPERATURE_DATA_DIR, target_coords_time)
    if not temp_df_processed.empty:
        main_df = merge_env_data_to_main(main_df, temp_df_processed, 'temp')
    else:
        print("Temperature data processing yielded no data. 'temp' column will be NaN.")
        main_df['temp'] = np.nan


    # 3. Load, process, and merge Precipitation data
    # Using the pre-processed CSV option for this example.
    # If using TRMM HDF, you would call a function to process those here.
    precip_df_processed = load_and_process_precipitation_data_csv(PRECIPITATION_CSV_FILE)
    if not precip_df_processed.empty:
        main_df = merge_env_data_to_main(main_df, precip_df_processed, 'precipitation')
    else:
        print("Precipitation data processing yielded no data. 'precipitation' column will be NaN.")
        main_df['precipitation'] = np.nan

    # 4. Save the merged data
    print(f"\nFinal DataFrame with environmental data head:\n{main_df.head()}")
    print(f"Shape of final data with environment: {main_df.shape}")
    print(f"Missing values in S2 output before final save:\n{main_df.isnull().sum()}")
    
    # Ensure required columns exist before saving, even if all NaN
    for col in ['temp', 'precipitation']:
        if col not in main_df.columns:
            main_df[col] = np.nan

    main_df.to_csv(OUTPUT_ENV_MERGED_DATA_FILE, index=False)
    print(f"S2: Environmental data merging complete. Output saved to {OUTPUT_ENV_MERGED_DATA_FILE}")
    print("--- S2: Finished ---")

if __name__ == "__main__":
    main()