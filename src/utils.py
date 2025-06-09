# src/utils.py
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import unicodedata

# --- Text Normalization ---
def normalize_text(text):
    if pd.isna(text): return text
    text_str = str(text)
    try:
        text_normalized = unicodedata.normalize('NFKD', text_str)
        text_no_accents = ''.join(c for c in text_normalized if not unicodedata.combining(c))
    except TypeError: text_no_accents = text_str
    return text_no_accents.upper()

# --- Calculation Utilities ---
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) # Clip 'a' to avoid domain errors with sqrt
    return R * c

def calculate_fp_fn(y_true, y_pred): # Used by RF/XGB for simple FP/FN
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    fp = np.sum((y_pred_arr == 1) & (y_true_arr == 0))
    fn = np.sum((y_pred_arr == 0) & (y_true_arr == 1))
    return int(fp), int(fn)

# --- Plotting Utilities ---

def function_for_one_year(shp_file_path, data, maxi, mini, target_name,
                          plot_title, colorbar_label, cmap_name, axes_obj, year_to_plot,
                          plot_only_positive_targets=True, add_colorbar_to_axes=False):
    """
    Plots data for a single specified year on a map of Brazil, colored by target_name.
    Assumes `data` is already filtered for the `year_to_plot` if only single year context.
    If `data` contains multiple years, it will filter internally.
    """
    try:
        gdf_boundaries = gpd.read_file(shp_file_path)
    except Exception as e:
        print(f"Error loading shapefile {shp_file_path} in function_for_one_year: {e}")
        axes_obj.set_title(f"{plot_title} (Shapefile Error)", fontsize=14); axes_obj.set_axis_off()
        return

    if data.empty or target_name not in data.columns:
        axes_obj.set_title(f"{plot_title} (No Data)", fontsize=14); axes_obj.set_axis_off()
        gdf_boundaries.plot(ax=axes_obj, color='lightgray', edgecolor='white', linewidth=0.5)
        return

    # Prepare GeoDataFrame from data
    plot_gdf = data.copy() # Assume data might already be a GeoDataFrame
    if 'geometry' not in plot_gdf.columns:
        if not all(c in plot_gdf.columns for c in ['lat', 'lon']):
            print(f"Error: 'lat' or 'lon' missing in data for {plot_title}")
            axes_obj.set_title(f"{plot_title} (Coord Error)", fontsize=14); axes_obj.set_axis_off()
            return
        geometry = [Point(xy) for xy in zip(plot_gdf['lon'], plot_gdf['lat'])]
        plot_gdf = gpd.GeoDataFrame(plot_gdf, geometry=geometry, crs="EPSG:4326")
    
    plot_gdf = plot_gdf.to_crs(gdf_boundaries.crs)
    
    # Filter data for the specific year IF 'Year' column exists and data is multi-year
    if 'Year' in plot_gdf.columns:
        data_to_plot_year = plot_gdf[plot_gdf['Year'] == year_to_plot].copy()
    else: # Assume data is already for the specific year
        data_to_plot_year = plot_gdf.copy()

    if data_to_plot_year.empty:
        axes_obj.set_title(f"{plot_title} (No Data for {year_to_plot})", fontsize=14); axes_obj.set_axis_off()
        gdf_boundaries.plot(ax=axes_obj, color='white', edgecolor='gray', linewidth=0.5)
        return

    # Further filter points to plot (e.g., only positive values if it's a count)
    if plot_only_positive_targets and data_to_plot_year[target_name].min() >= 0 :
        plot_points_data = data_to_plot_year[data_to_plot_year[target_name] > 0]
    else:
        plot_points_data = data_to_plot_year
    
    # Ensure mini and maxi are valid, fallback if data is empty or uniform
    actual_mini = plot_points_data[target_name].min() if not plot_points_data.empty else mini
    actual_maxi = plot_points_data[target_name].max() if not plot_points_data.empty else maxi
    if pd.isna(actual_mini) or pd.isna(actual_maxi) or actual_mini == actual_maxi:
        actual_mini = 0
        actual_maxi = 1 if actual_maxi == actual_mini else actual_maxi # Avoid norm error

    norm = mcolors.Normalize(vmin=actual_mini, vmax=actual_maxi)
    cmap = plt.get_cmap(cmap_name)

    gdf_boundaries.plot(ax=axes_obj, color='white', edgecolor='gray', linewidth=0.5)
    if not plot_points_data.empty:
        axes_obj.scatter(
            plot_points_data.geometry.x, plot_points_data.geometry.y,
            c=plot_points_data[target_name], cmap=cmap, norm=norm,
            s=100, linewidth=0.5, alpha=0.75
        )
    axes_obj.set_title(plot_title, fontsize=14)
    axes_obj.set_xlim([-75, -30]) # Brazil typical bounds
    axes_obj.set_ylim([-35, 5])
    axes_obj.set_axis_off()

    if add_colorbar_to_axes: # If called for a single plot where its own colorbar is desired
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        # Position colorbar carefully if axes_obj is part of a larger figure
        cax = axes_obj.inset_axes([0.2, -0.08, 0.6, 0.03]) # Example position below subplot
        cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(colorbar_label, fontsize=10)


def plot_flexible_data_single_map(shp_file_path, data_df, target_name, plot_title,
                                  data_type="cases", year_col="Year", lat_col="lat", lon_col="long",
                                  cmap_name="viridis", size_scale=500, alpha=0.7,
                                  legend_title="Legend", ax=None, year_colorbar=True,
                                  output_filename=None):
    """
    Plots data (e.g., epizootic cases) as proportional circles, colored by year, on a single map.
    """
    # ... (This function remains the same as in the previous utils.py version I provided) ...
    # Ensure it's the complete version that handles size scaling and legends correctly.
    # For brevity, I will not paste its full content here again, but assume it's the robust one.
    # Key elements:
    # - Loads shapefile
    # - Converts data_df to GeoDataFrame
    # - Normalizes circle sizes based on data_type ('cases' -> sqrt, 'population' -> log10)
    # - Interpolates sizes
    # - Plots base map
    # - Iterates through years, plotting circles colored by year and sized by target_name
    # - Adds a colorbar for 'Year'
    # - Adds a legend for circle sizes
    # - Sets title and axis properties
    # - Saves if output_filename is provided
    # (Refer to the version from the "S8 and complete utils.py" response for its full body)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.get_figure()

    try:
        gdf_boundaries = gpd.read_file(shp_file_path)
    except Exception as e:
        print(f"Error loading shapefile {shp_file_path} in plot_flexible_data: {e}")
        ax.set_title(f"{plot_title} (Shapefile Error)", fontsize=12); ax.set_axis_off()
        return

    if data_df.empty or not all(col in data_df.columns for col in [lon_col, lat_col, year_col, target_name]):
        ax.set_title(f"{plot_title} (No Data / Missing Cols)", fontsize=12); ax.set_axis_off()
        gdf_boundaries.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)
        if output_filename: plt.savefig(output_filename, format='pdf', bbox_inches='tight')
        # plt.show()
        return

    plot_gdf = gpd.GeoDataFrame(
        data_df, geometry=gpd.points_from_xy(data_df[lon_col], data_df[lat_col]),
        crs="EPSG:4326"
    ).to_crs(gdf_boundaries.crs)

    if data_type == "cases":
        min_size_plot = 10
        size_values_raw = pd.to_numeric(plot_gdf[target_name], errors='coerce').fillna(0)
        size_col_for_interp = np.sqrt(size_values_raw.clip(lower=0))
    elif data_type == "population":
        min_size_plot = 5
        size_values_raw = pd.to_numeric(plot_gdf[target_name], errors='coerce').fillna(1)
        size_col_for_interp = np.log10(size_values_raw.clip(lower=1))
    else:
        raise ValueError("data_type must be 'cases' or 'population'")

    size_col_for_interp = pd.to_numeric(size_col_for_interp, errors='coerce').fillna(0)
    if len(size_col_for_interp) == 0 or size_col_for_interp.min() == size_col_for_interp.max():
        plot_gdf['plot_size'] = min_size_plot
    else:
        plot_gdf['plot_size'] = np.interp(
            size_col_for_interp,
            (size_col_for_interp.min(), size_col_for_interp.max()),
            (min_size_plot, size_scale)
        )
    plot_gdf['plot_size'] = plot_gdf['plot_size'].fillna(min_size_plot)

    gdf_boundaries.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)
    unique_years = sorted(plot_gdf[year_col].unique(), reverse=True)
    if not unique_years:
        ax.set_title(f"{plot_title} (No Years)", fontsize=12); ax.set_axis_off()
        if output_filename: plt.savefig(output_filename, format='pdf', bbox_inches='tight')
        return

    year_norm = Normalize(vmin=min(unique_years), vmax=max(unique_years))
    year_cmap_obj = plt.get_cmap(cmap_name)

    # Collect handles for year legend
    year_legend_handles = {}
    for year_val in unique_years:
        year_data = plot_gdf[plot_gdf[year_col] == year_val]
        if not year_data.empty:
            sc = ax.scatter(
                x=year_data.geometry.x, y=year_data.geometry.y,
                s=year_data['plot_size'], color=[year_cmap_obj(year_norm(year_val))] * len(year_data),
                alpha=alpha, edgecolor='k', linewidth=0.2,
                # label=str(year_val) # Adding label here creates too many legend entries
            )
            if str(year_val) not in year_legend_handles : # Only add handle once per year
                 year_legend_handles[str(year_val)] = sc


    if year_colorbar and unique_years: # Add colorbar for year
        mappable_year = cm.ScalarMappable(cmap=year_cmap_obj, norm=year_norm)
        mappable_year.set_array([])
        cbar_year_pos = [0.20, 0.08, 0.6, 0.02] # Position for horizontal cbar below plot if ax is main
        if fig.get_constrained_layout(): fig.set_constrained_layout(False) # Temp disable for manual cbar
        cax_year = fig.add_axes(cbar_year_pos)
        if fig.get_constrained_layout(): fig.set_constrained_layout(True) # Re-enable

        cb_year = fig.colorbar(mappable_year, cax=cax_year, orientation='horizontal')
        cb_year.set_label('Year', fontsize=10)
        # Make year ticks integers if they are years
        if all(isinstance(y, (int, float)) and y == int(y) for y in unique_years):
             cb_year.set_ticks(np.linspace(min(unique_years), max(unique_years), num=min(5, len(unique_years))).astype(int))


    legend_elements_size = []
    if not plot_gdf[target_name].empty and plot_gdf[target_name].nunique() > 0:
        q_vals = np.percentile(plot_gdf[target_name].dropna(), [10, 50, 90])
        legend_raw_vals_for_size = sorted(list(set(q_vals[q_vals > 0]))) # Use positive quantiles
        if not legend_raw_vals_for_size and plot_gdf[target_name].max() > 0: # Fallback if quantiles are zero
            legend_raw_vals_for_size = [plot_gdf[target_name].max()]
        if not legend_raw_vals_for_size: legend_raw_vals_for_size = [1]


        if data_type == "cases":
            legend_interp_in = np.sqrt(np.clip(legend_raw_vals_for_size, 0, None))
        else: # population
            legend_interp_in = np.log10(np.clip(legend_raw_vals_for_size, 1, None))

        if len(size_col_for_interp) == 0 or size_col_for_interp.min() == size_col_for_interp.max():
            legend_plot_sizes = [min_size_plot] * len(legend_interp_in)
        else:
            legend_plot_sizes = np.interp(legend_interp_in,
                                        (size_col_for_interp.min(), size_col_for_interp.max()),
                                        (min_size_plot, size_scale))
        legend_labels_size = [f"{int(v):,}" for v in legend_raw_vals_for_size]
        legend_elements_size = [
            Line2D([0], [0], marker='o', color='w', label=lab,
                   markersize=np.sqrt(max(1,siz)),
                   markerfacecolor='gray', alpha=0.7)
            for siz, lab in zip(legend_plot_sizes, legend_labels_size) if pd.notna(siz)
        ]
    if legend_elements_size:
        ax.legend(handles=legend_elements_size, title=legend_title, loc='upper right',
                  frameon=False, fontsize='small', title_fontsize='small')

    ax.set_title(plot_title, fontsize=12) # Reduced from 16 for potentially smaller subplot
    ax.set_axis_off()

    if output_filename:
        plt.savefig(output_filename, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
    # No plt.show() here, should be called by the main script