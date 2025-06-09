# src/S4_feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
# Assuming utils.py is in the same directory (src) or PYTHONPATH is set
from utils import haversine_vectorized # Ensure this function is in your utils.py

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Input file from S3
S3_DATA_WITH_GENOMIC_FILE = PROCESSED_DATA_DIR / "S3_final_data_with_genomic_clusters.csv"

# Output file from this script
OUTPUT_FEATURE_ENGINEERED_DATA_FILE = PROCESSED_DATA_DIR / "S4_feature_engineered_data.csv"

# --- Main Feature Engineering Logic ---
def main():
    print("--- S4: Starting Feature Engineering ---")

    # 1. Load data from S3
    try:
        data = pd.read_csv(S3_DATA_WITH_GENOMIC_FILE)
        print(f"Loaded data from S3 with shape: {data.shape}")
    except FileNotFoundError:
        print(f"Error: S3 output file not found at {S3_DATA_WITH_GENOMIC_FILE}. Exiting.")
        return
    except Exception as e:
        print(f"Error loading S3 data: {e}. Exiting.")
        return

    # Ensure 'Spillover' column exists and is 0 or 1.
    # S1 should have created it from 'Cases'.
    if 'Spillover' not in data.columns:
        if 'Cases' in data.columns:
            print("Defining 'Spillover' target variable from 'Cases'.")
            data["Spillover"] = data["Cases"].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)
        else:
            print("Error: 'Spillover' or 'Cases' column not found. Cannot proceed.")
            return
    else: # Ensure it's binary
        data["Spillover"] = data["Spillover"].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)


    # 2. Drop unnecessary raw columns 
    cols_to_drop = ['Cases'] # 'Cases' is converted to 'Spillover'

    data = data.drop(columns=cols_to_drop, errors='ignore')
    print(f"Shape after dropping initial columns: {data.shape}")


    # 3. Cyclical Encoding for Months 
    if 'Months' not in data.columns:
        print("Error: 'Months' column required for cyclical encoding is missing.")
        return
    data["month_sin"] = np.sin(2 * np.pi * data["Months"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["Months"] / 12)
    print("Created cyclical month features (month_sin, month_cos).")

    # 4. Interaction Features 
    # Ensure 'temp', 'precipitation', 'population', 'SpRich' exist
    required_interaction_cols = ['temp', 'precipitation', 'Population', 'SpRich']
    for col in required_interaction_cols:
        if col not in data.columns:
            print(f"Warning: Column '{col}' for interaction features is missing. Filling with 0 or NaN.")
            data[col] = 0 # Or np.nan, but 0 might be safer for multiplication
    
    data["temp_precip"] = data["temp"] * data["precipitation"]
    data["pop_spRich"] = data["Population"] * data["SpRich"]
    print("Created interaction features (temp_precip, pop_spRich).")

    # 5. Rolling/Lagged Spillover Features
    # Data must be sorted by Region, Year, Months for correct shift and rolling operations
    if not all(c in data.columns for c in ["Region", "Year", "Months"]):
        print("Error: 'Region', 'Year', 'Months' columns required for rolling features are missing.")
        return
        
    data = data.sort_values(by=["Region", "Year", "Months"]).reset_index(drop=True)

    # Spillover_lag1 (from notebook pg 101)
    data["Spillover_lag1"] = data.groupby("Region")["Spillover"].shift(1)

    # Spillover_rolling3 (3-month avg before current)
    # This calculates mean of previous 3 months (t-1, t-2, t-3)
    # shift(1) first to exclude current month, then rolling(3) on that.
    data["Spillover_rolling3"] = data.groupby("Region")["Spillover"]\
                                       .shift(1)\
                                       .rolling(window=3, min_periods=1)\
                                       .mean()\
                                       .reset_index(drop=True) # Reset index from rolling

    print("Created lagged spillover features (Spillover_lag1, Spillover_rolling3).")

    # 6. Cumulative Spillover Per Region 
    # This calculates cumulative sum of spillovers *before* the current month
    data["Spillover_cum"] = data.groupby("Region")["Spillover"]\
                                  .shift(1)\
                                  .cumsum() # cumsum naturally handles NaNs from shift(1) at start of group
    print("Created cumulative spillover feature (Spillover_cum).")


    # 7. Distance to Nearest Past Spillover (Haversine Distance) 
    print("Calculating distance to nearest past spillover (this may take time)...")
    data["Nearest_Spillover_Dist"] = np.nan
    
    # Create a DataFrame of only past spillover locations for efficient lookup
    spillover_locations = data[data["Spillover"] > 0][['Year', 'Months', 'lat', 'long']].copy()
    if spillover_locations.empty:
        print("No past spillover events found in data. 'Nearest_Spillover_Dist' will be NaN.")
    else:
        # Convert Year/Months to a comparable 'time_index' for chronological filtering
        # For simplicity, let's create a numeric time index: Year * 100 + Month
        data['time_index_internal'] = data['Year'] * 100 + data['Months']
        spillover_locations['time_index_internal'] = spillover_locations['Year'] * 100 + spillover_locations['Months']

        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Finding Nearest Spillover"):
            current_time_index = row['time_index_internal']
            
            # Filter for spillovers that occurred strictly before the current record's time
            past_spillovers_df = spillover_locations[spillover_locations['time_index_internal'] < current_time_index]

            if not past_spillovers_df.empty:
                distances = haversine_vectorized(
                    row["lat"], row["long"],
                    past_spillovers_df["lat"].values, past_spillovers_df["long"].values
                )
                if distances.size > 0: # Ensure distances array is not empty
                    data.loc[idx, "Nearest_Spillover_Dist"] = np.nanmin(distances) # Use nanmin
        
        data.drop(columns=['time_index_internal'], inplace=True)

    # Fill remaining NaNs in Nearest_Spillover_Dist (e.g., for first events or isolated areas)
    if "Nearest_Spillover_Dist" in data.columns and data["Nearest_Spillover_Dist"].isnull().any():
        max_dist = data["Nearest_Spillover_Dist"].max()
        if pd.notna(max_dist):
            data["Nearest_Spillover_Dist"].fillna(max_dist, inplace=True)
            print(f"Filled NaNs in 'Nearest_Spillover_Dist' with max observed distance: {max_dist:.2f} km.")
        else: # If all are NaN (no spillovers at all, or only one)
            data["Nearest_Spillover_Dist"].fillna(2200, inplace=True) # A large default, e.g., ~max Brazil extent
            print("Filled all NaNs in 'Nearest_Spillover_Dist' with a default large distance (2200km).")
    print("Calculated 'Nearest_Spillover_Dist'.")

    # 8. Handle NaNs generated by shift() or rolling() operations by filling with 0
    # These features represent counts or averages, so 0 is a reasonable fill for initial periods.
    cols_to_fillna_zero = ["Spillover_lag1", "Spillover_rolling3", "Spillover_cum"]
    for col in cols_to_fillna_zero:
        if col in data.columns:
            data[col].fillna(0, inplace=True)
    print("Filled NaNs in lagged/rolling features with 0.")

    # 9. Final Checks and Save
    # Columns like 'Region', 'Year', 'Months', 'lat', 'long' are kept for splitting/identification later
    # 'cluster_labels' from genomic processing is also a feature
    
    # Ensure all feature columns intended for modeling are numeric and have no NaNs
    feature_cols_for_model = [
        'lat', 'long', 'Population', 'Epizootic_cases', 'temp', 'precipitation',
        'SpRich', 'month_sin', 'month_cos', 'temp_precip', 'pop_spRich',
        'Spillover_lag1', 'Spillover_rolling3', 'Spillover_cum',
        'Nearest_Spillover_Dist', 'cluster_labels' # Assuming this is the chosen genomic cluster feature
    ]
    # Add other important columns needed for splitting data later, like Year, Region, Months, Spillover target
    all_cols_to_keep = ['Year', 'Months', 'Region'] + feature_cols_for_model + ['Spillover']
    
    final_df_cols = []
    for col in all_cols_to_keep:
        if col in data.columns:
            final_df_cols.append(col)
        else:
            print(f"Warning: Expected column '{col}' not found in DataFrame after feature engineering.")

    final_featured_data = data[final_df_cols].copy()

    print(f"\nFinal feature engineered DataFrame head:\n{final_featured_data.head()}")
    print(f"Shape of final feature engineered data: {final_featured_data.shape}")
    print(f"Missing values in S4 output:\n{final_featured_data.isnull().sum().sort_values(ascending=False)}")

    final_featured_data.to_csv(OUTPUT_FEATURE_ENGINEERED_DATA_FILE, index=False)
    print(f"S4: Feature engineering complete. Output saved to {OUTPUT_FEATURE_ENGINEERED_DATA_FILE}")
    print("--- S4: Finished ---")

if __name__ == "__main__":
    main()