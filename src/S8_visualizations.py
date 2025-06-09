# src/S8_visualizations.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import geopandas as gpd # For loading shapefile if utils functions don't handle it directly

from utils import (
    function_for_one_year,
    plot_flexible_data_single_map,
)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
RAW_DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Input file from S4 (contains all features needed for grouping and actual 2018 data)
MAIN_DATA_FILE = PROCESSED_DATA_DIR / "S4_feature_engineered_data.csv"
SHP_FILE_BRAZIL_ADM1 = RAW_DATA_DIR / "BRA_adm1.shp"

# Prediction files for 2018 (outputs from S5, S6, S7)
PREDS_RF_FILE = PROCESSED_DATA_DIR / "S5_test_predictions_rf.csv"
PREDS_XGB_FILE = PROCESSED_DATA_DIR / "S5_test_predictions_xgb.csv"
# LSTM predictions are tricky to map directly from S6 output as they are for sequences.
# Assuming S6_test_predictions_lstm.csv has been processed to include lat/long for 2018 events if plottable this way.
# Or, we load the original test_df for 2018 and map the sequence predictions back.
# For now, let's assume a plottable format if PREDS_LSTM_FILE is used.
PREDS_LSTM_FILE = PROCESSED_DATA_DIR / "S6_test_predictions_lstm.csv" # Needs lat/long/Year/Predicted_LSTM
PREDS_AE_FILE = PROCESSED_DATA_DIR / "S7_test_predictions_ae.csv" # Has original Year/lat/long


# --- Main Visualization Logic ---
def main():
    print("--- S8: Starting Focused Visualizations ---")

    # 1. Load main data for historical plots
    try:
        data_full = pd.read_csv(MAIN_DATA_FILE)
        print(f"Loaded main data for visualizations: {data_full.shape}")
    except FileNotFoundError:
        print(f"Error: Main data file not found at {MAIN_DATA_FILE}. Cannot generate plots.")
        data_full = pd.DataFrame() # Allow script to continue if some plots don't need it

    mini_lat = -35 # Default, will be overwritten if data_full is loaded
    if not data_full.empty:
        valid_lats_temp = data_full['lat'][(data_full['lat'] >= -35) & (data_full['lat'] <= 6)]
        if not valid_lats_temp.empty:
            mini_lat = valid_lats_temp.min()

        # --- Plot 1: Temperature and Precipitation for 2017 ---
        print("Generating Temperature and Precipitation map for 2017...")
        fig1, axes1 = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        data_grouped_temp = data_full.groupby(['Year', 'lat', 'long'])['temp'].mean().reset_index()
        data_grouped_prec = data_full.groupby(['Year', 'lat', 'long'])['precipitation'].mean().reset_index()

        data_temp_2017 = data_grouped_temp[
            (data_grouped_temp['Year'] == 2017) & (data_grouped_temp['lat'] > mini_lat)
        ].copy()
        if not data_temp_2017.empty and 'temp' in data_temp_2017.columns and data_temp_2017['temp'].notna().any():
            function_for_one_year(
                shp_file_path=SHP_FILE_BRAZIL_ADM1, data=data_temp_2017,
                maxi=data_temp_2017['temp'].max(), mini=data_temp_2017['temp'].min(),
                target_name="temp", plot_title="Mean Temperature Brazil (2017)",
                colorbar_label="Mean Temperature Value", cmap_name="coolwarm",
                axes_obj=axes1[0], year_to_plot=2017
            )
            sm_temp = cm.ScalarMappable(norm=mcolors.Normalize(vmin=data_temp_2017['temp'].min(), vmax=data_temp_2017['temp'].max()), cmap=plt.get_cmap("coolwarm"))
            fig1.colorbar(sm_temp, ax=axes1[0], orientation="horizontal", pad=0.05, shrink=0.8, label="Mean Temperature Value")
        else: axes1[0].set_title("Mean Temperature Brazil (2017) - No Data"); axes1[0].set_axis_off()

        data_prec_2017 = data_grouped_prec[
            (data_grouped_prec['Year'] == 2017) & (data_grouped_prec['lat'] > mini_lat)
        ].copy()
        if not data_prec_2017.empty and 'precipitation' in data_prec_2017.columns and data_prec_2017['precipitation'].notna().any():
            function_for_one_year(
                shp_file_path=SHP_FILE_BRAZIL_ADM1, data=data_prec_2017,
                maxi=data_prec_2017['precipitation'].max(), mini=data_prec_2017['precipitation'].min(),
                target_name="precipitation", plot_title="Mean Precipitation Brazil (2017)",
                colorbar_label="Mean Precipitation Value per Hour", cmap_name="BrBG",
                axes_obj=axes1[1], year_to_plot=2017
            )
            sm_prec = cm.ScalarMappable(norm=mcolors.Normalize(vmin=data_prec_2017['precipitation'].min(), vmax=data_prec_2017['precipitation'].max()), cmap=plt.get_cmap("BrBG"))
            fig1.colorbar(sm_prec, ax=axes1[1], orientation="horizontal", pad=0.05, shrink=0.8, label="Mean Precipitation Value per Hour")
        else: axes1[1].set_title("Mean Precipitation Brazil (2017) - No Data"); axes1[1].set_axis_off()
        plt.savefig(FIGURES_DIR / 'temperature_precipitation_brazil_2017.pdf', format='pdf', bbox_inches='tight')
        plt.show()

        # --- Plot 2: Population for 2017---
        print("Generating Population map for 2017...")
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)
        data_grouped_population = data_full.groupby(['Year', 'lat', 'long'])['Population'].mean().reset_index()
        data_pop_2017 = data_grouped_population[
            (data_grouped_population['Year'] == 2017) & (data_grouped_population['lat'] > mini_lat)
        ].copy()
        if not data_pop_2017.empty and 'Population' in data_pop_2017.columns and data_pop_2017['Population'].notna().any():
            data_pop_2017['log_pop'] = np.log10(data_pop_2017['Population'].replace(0, 1))
            if not data_pop_2017['log_pop'].empty and data_pop_2017['log_pop'].notna().any():
                function_for_one_year(
                    shp_file_path=SHP_FILE_BRAZIL_ADM1, data=data_pop_2017,
                    maxi=data_pop_2017['log_pop'].max(), mini=data_pop_2017['log_pop'].min(),
                    target_name="log_pop", plot_title="Population Brazil (2017 - Log Scale)",
                    colorbar_label="Log10(Number of Population)", cmap_name="PuOr",
                    axes_obj=ax2, year_to_plot=2017
                )
                sm_pop = cm.ScalarMappable(norm=mcolors.Normalize(vmin=data_pop_2017['log_pop'].min(), vmax=data_pop_2017['log_pop'].max()), cmap=plt.get_cmap("PuOr"))
                fig2.colorbar(sm_pop, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.8, label="Log10(Number of Population)")
            else: ax2.set_title("Population Brazil (2017 - Log Scale) - No Valid Data for log_pop"); ax2.set_axis_off()
        else: ax2.set_title("Population Brazil (2017 - Log Scale) - No Data"); ax2.set_axis_off()
        plt.savefig(FIGURES_DIR / 'population_brazil_2017_log.pdf', format='pdf', bbox_inches='tight')
        plt.show()

        # --- Plot 3: YF Situation (Cases, Spillover, Epizootics) 2000-2024  ---
        print("Generating YF Situation maps (Cases, Spillover, Epizootics) for 2000-2024...")
        fig3, axes3 = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
        data_grouped_cases = data_full.groupby(['Year', 'lat', 'long'])['Cases'].sum().reset_index()
        data_grouped_human_spillover = data_full.groupby(['Year', 'lat', 'long'])['Spillover'].sum().reset_index()
        data_grouped_epizootic = data_full.groupby(['Year', 'lat', 'long'])['Epizootic_cases'].sum().reset_index()

        plot_cases = data_grouped_cases[(data_grouped_cases['Cases'] > 0) & (data_grouped_cases['lat'] > mini_lat)].copy()
        plot_spillover = data_grouped_human_spillover[(data_grouped_human_spillover['Spillover'] > 0) & (data_grouped_human_spillover['lat'] > mini_lat)].copy()
        plot_epizootic = data_grouped_epizootic[(data_grouped_epizootic['Epizootic_cases'] > 0) & (data_grouped_epizootic['lat'] > mini_lat)].copy()

        if not plot_cases.empty:
            plot_flexible_data_single_map(shp_file_path=SHP_FILE_BRAZIL_ADM1, data=plot_cases, target_name="Cases", plot_title="Yellow Fever Cases (All Years)", data_type="cases", cmap_name="plasma", legend_title="Cases Count", ax=axes3[0], year_colorbar=False)
        else: axes3[0].set_title("Yellow Fever Cases (All Years) - No Data"); axes3[0].set_axis_off()
        if not plot_spillover.empty:
            plot_flexible_data_single_map(shp_file_path=SHP_FILE_BRAZIL_ADM1, data=plot_spillover, target_name="Spillover", plot_title="Spillover Events (All Years)", data_type="cases", cmap_name="viridis", legend_title="Spillover Count", ax=axes3[1], year_colorbar=False)
        else: axes3[1].set_title("Spillover Events (All Years) - No Data"); axes3[1].set_axis_off()
        if not plot_epizootic.empty:
            plot_flexible_data_single_map(shp_file_path=SHP_FILE_BRAZIL_ADM1, data=plot_epizootic, target_name="Epizootic_cases", plot_title="Epizootic Cases (All Years)", data_type="cases", cmap_name="magma", legend_title="Epizootic Count", ax=axes3[2], year_colorbar=True)
        else: axes3[2].set_title("Epizootic Cases (All Years) - No Data"); axes3[2].set_axis_off()
        fig3.suptitle("Yellow Fever Activity in Brazil (All Recorded Years)", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(FIGURES_DIR / 'yellow_fever_situation_overall.pdf', format='pdf', bbox_inches='tight')
        plt.show()


    # --- Plot 4: Model Predictions for 2018 (NEW SECTION) ---
    print("Generating maps of Real vs Predicted Spillover for 2018...")
    
    plot_datasets_2018 = []
    plot_titles_2018 = []
    plot_targets_2018 = [] # Column name for predicted values

    # Load actual 2018 data
    if not data_full.empty:
        data_2018_actual_for_plot = data_full[data_full['Year'] == 2018].copy()
        # Aggregate actual spillovers for 2018 by location for plotting
        actual_spillover_map_data = data_2018_actual_for_plot.groupby(['lat', 'long'])['Spillover'].sum().reset_index()
        if not actual_spillover_map_data.empty:
            plot_datasets_2018.append(actual_spillover_map_data)
            plot_titles_2018.append('Real Spillover (2018)')
            plot_targets_2018.append('Spillover')
    else:
        print("Warning: Main data_full is empty, cannot plot actual 2018 spillovers.")

    # Load and prepare RF predictions
    try:
        preds_rf_df = pd.read_csv(PREDS_RF_FILE) # Contains Year, lat, long, RF_pred_binary
        rf_map_data = preds_rf_df[preds_rf_df['Year'] == 2018].groupby(['lat', 'long'])['RF_pred_binary'].sum().reset_index()
        if not rf_map_data.empty:
            plot_datasets_2018.append(rf_map_data)
            plot_titles_2018.append('Predicted RF (2018)')
            plot_targets_2018.append('RF_pred_binary')
    except FileNotFoundError:
        print(f"Warning: RF prediction file not found at {PREDS_RF_FILE}")

    # Load and prepare XGBoost predictions
    try:
        preds_xgb_df = pd.read_csv(PREDS_XGB_FILE) # Contains Year, lat, long, XGB_pred_binary
        xgb_map_data = preds_xgb_df[preds_xgb_df['Year'] == 2018].groupby(['lat', 'long'])['XGB_pred_binary'].sum().reset_index()
        if not xgb_map_data.empty:
            plot_datasets_2018.append(xgb_map_data)
            plot_titles_2018.append('Predicted XGBoost (2018)')
            plot_targets_2018.append('XGB_pred_binary')
    except FileNotFoundError:
        print(f"Warning: XGBoost prediction file not found at {PREDS_XGB_FILE}")

    # Load and prepare Autoencoder predictions
    try:
        preds_ae_df = pd.read_csv(PREDS_AE_FILE) # Contains Year, lat, long, AE_Predicted_Anomaly
        ae_map_data = preds_ae_df[preds_ae_df['Year'] == 2018].groupby(['lat', 'long'])['AE_Predicted_Anomaly'].sum().reset_index()
        if not ae_map_data.empty:
            plot_datasets_2018.append(ae_map_data)
            plot_titles_2018.append('Predicted Autoencoder (2018)')
            plot_targets_2018.append('AE_Predicted_Anomaly')
    except FileNotFoundError:
        print(f"Warning: Autoencoder prediction file not found at {PREDS_AE_FILE}")
        
    # LSTM Prediction Plotting (More Complex - requires mapping sequence predictions)
    # The S6_test_predictions_lstm.csv contains predictions for *sequences*, not directly for original rows.
    # To plot LSTM results on a map like others, you need to:
    # 1. Load the original `test_df` that was used to create LSTM sequences in S6.
    # 2. Load the `S6_test_predictions_lstm.csv` which has `true_label` and `predicted_binary_0.3_thresh`.
    # 3. Align these sequence predictions back to the corresponding original `test_df` rows.
    #    This requires knowing which original row each sequence target corresponds to.
    #    The `create_sequences_for_lstm` function output `R_test_seq` (region IDs) and `y_test_seq` (targets).
    #    The predictions in `S6_test_predictions_lstm.csv` align with `y_test_seq`.
    #    You'd need to also save the `X_test_curr` (which contains lat/lon for the predicted step) from S6
    #    along with the predictions to be able to plot them spatially.
    # For now, I'll add a placeholder. If you have processed LSTM preds with lat/lon, enable this.
    try:
        preds_lstm_df = pd.read_csv(PROCESSED_DATA_DIR / "S6_test_predictions_lstm_with_coords.csv") # Assumes you create this
        lstm_map_data = preds_lstm_df[preds_lstm_df['Year'] == 2018].groupby(['lat', 'long'])['Predicted_LSTM_Binary'].sum().reset_index() # Adjust col name
        if not lstm_map_data.empty:
            plot_datasets_2018.append(lstm_map_data)
            plot_titles_2018.append('Predicted LSTM (2018)')
            plot_targets_2018.append('Predicted_LSTM_Binary')
    except FileNotFoundError:
        print(f"Warning: Processed LSTM prediction file for mapping not found.")


    if not plot_datasets_2018:
        print("No prediction data available to plot for 2018.")
    else:
        num_pred_plots = len(plot_datasets_2018)
        # Dynamically create subplot layout (e.g. 2 rows if > 3 plots)
        ncols_fig4 = min(num_pred_plots, 3) # Max 3 plots per row
        nrows_fig4 = (num_pred_plots + ncols_fig4 -1) // ncols_fig4 # Calculate needed rows
        
        fig4, axes4_list = plt.subplots(nrows_fig4, ncols_fig4, figsize=(6 * ncols_fig4, 6 * nrows_fig4), squeeze=False)
        axes4 = axes4_list.flatten()

        # Determine a common v_max for the colorbar across all prediction plots
        common_vmax = 0
        for df_plot, target_col_name in zip(plot_datasets_2018, plot_targets_2018):
            if not df_plot.empty and target_col_name in df_plot.columns:
                current_max = df_plot[df_plot[target_col_name] > 0][target_col_name].max() # Max of positive predictions/actuals
                if pd.notna(current_max) and current_max > common_vmax:
                    common_vmax = current_max
        if common_vmax == 0 : common_vmax = 1 # Default if no positive cases/predictions

        norm_pred_plot = mcolors.Normalize(vmin=0, vmax=common_vmax)
        cmap_pred_plot = cm.get_cmap('viridis') # Consistent colormap

        for i in range(num_pred_plots):
            ax_curr = axes4[i]
            data_to_plot = plot_datasets_2018[i]
            title_curr = plot_titles_2018[i]
            target_curr = plot_targets_2018[i]
            
            # Filter for non-zero points to plot
            plot_points = data_to_plot[data_to_plot[target_curr] > 0].copy()
            if 'Year' not in plot_points.columns: plot_points['Year'] = 2018 # Ensure Year column

            function_for_one_year(
                shp_file_path=SHP_FILE_BRAZIL_ADM1,
                data=plot_points, # Pass only data with points to plot
                maxi=common_vmax, mini=0, target_name=target_curr,
                plot_title=title_curr, colorbar_label="", # Shared colorbar, so no individual label
                cmap_name='viridis', axes_obj=ax_curr, year_to_plot=2018,
                # Suppress individual colorbars in function_for_one_year if it has such an option
                # or remove them manually after the call if it adds them.
            )
            # If function_for_one_year adds its own colorbar, remove it for a shared one
            if hasattr(ax_curr, 'images') and ax_curr.images:
                for im_obj in ax_curr.images:
                    if hasattr(im_obj, 'colorbar') and im_obj.colorbar:
                        im_obj.colorbar.remove()
            if len(fig4.axes) > num_pred_plots * (1 if nrows_fig4 ==1 else ncols_fig4) : # If extra axes were added for colorbars
                 # This logic to remove extra colorbar axes might be tricky, better if function_for_one_year can suppress it
                 pass


        # Remove any unused subplots if num_pred_plots is not a multiple of ncols_fig4
        for i in range(num_pred_plots, nrows_fig4 * ncols_fig4):
            fig4.delaxes(axes4[i])

        # Add a single shared colorbar for the prediction plots
        if num_pred_plots > 0 :
            # Adjust position: [left, bottom, width, height]
            cbar_ax_pred = fig4.add_axes([0.35, 0.05 / nrows_fig4, 0.3, 0.015 * (1/nrows_fig4)]) # Adjust vertical position based on rows
            mappable_pred = cm.ScalarMappable(norm=norm_pred_plot, cmap=cmap_pred_plot)
            cb_pred = fig4.colorbar(mappable_pred, cax=cbar_ax_pred, orientation="horizontal")
            cb_pred.set_label('Number of Spillover Events (Actual/Predicted)', fontsize=12)
        
        fig4.suptitle('Actual vs. Predicted Spillover Events (2018)', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.96]) # Adjust rect to make space for shared colorbar and suptitle
        plt.savefig(FIGURES_DIR / "model_predictions_comparison_2018.pdf", format='pdf', bbox_inches='tight')
        plt.show()

    print("--- S8: Visualizations Complete ---")

if __name__ == "__main__":
    main()