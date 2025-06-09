# src/S5_train_evaluate_supervised.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns # For nicer feature importance plots
import joblib # For saving scikit-learn models

from sklearn.model_selection import train_test_split # Not used for time-based split here
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
import xgboost as xgb

# Assuming utils.py is in the same directory (src) or PYTHONPATH is set
from utils import calculate_fp_fn # For false positive/negative counts

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "trained_models"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Input file from S4
S4_FEATURE_ENGINEERED_DATA_FILE = PROCESSED_DATA_DIR / "S4_feature_engineered_data.csv"

# --- Model Training and Evaluation Functions ---

def evaluate_classification_model(model_name, y_true, y_pred_proba, y_pred_binary,
                                  output_dir=TABLES_DIR):
    """Calculates and prints common classification metrics, returns them as a dict."""
    print(f"\n--- Evaluating {model_name} ---")
    auc = roc_auc_score(y_true, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred_binary)
    fp_manual, fn_manual = calculate_fp_fn(y_true, y_pred_binary) # From utils

    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"False Positives: {fp_manual}, False Negatives: {fn_manual}")

    metrics = {
        'model': model_name, 'auc': auc, 'accuracy': accuracy,
        'precision': precision, 'recall': recall, 'f1_score': f1,
        'fp': fp_manual, 'fn': fn_manual
    }
    pd.DataFrame([metrics]).to_csv(output_dir / f"{model_name}_metrics.csv", index=False)
    return metrics

def plot_feature_importance(importances, feature_names, model_name, top_n=20, output_dir=FIGURES_DIR):
    """Plots feature importances for tree-based models."""
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, min(8, top_n * 0.4))) # Adjust height based on top_n
    sns.barplot(x='importance', y='feature', data=fi_df, palette="viridis")
    plt.title(f'Top {top_n} Feature Importances from {model_name}')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_feature_importance.pdf", format='pdf', bbox_inches='tight')
    print(f"Feature importance plot saved for {model_name}.")
    plt.show()


# --- Main Execution ---
def main():
    print("--- S5: Starting Training and Evaluation of Random Forest and XGBoost ---")

    # 1. Load feature-engineered data from S4
    try:
        data = pd.read_csv(S4_FEATURE_ENGINEERED_DATA_FILE)
        print(f"Loaded feature-engineered data with shape: {data.shape}")
    except FileNotFoundError:
        print(f"Error: S4 output file not found at {S4_FEATURE_ENGINEERED_DATA_FILE}. Exiting.")
        return
    except Exception as e:
        print(f"Error loading S4 data: {e}. Exiting.")
        return

    # Handle potential NaNs that might have slipped through or been created (e.g., if a feature col was all NaN)
    # For simplicity, filling with 0 here. More sophisticated imputation might be needed based on feature.
    cols_for_model = data.columns.drop(['Year', 'Months', 'Region', 'Spillover']) # Features
    for col in cols_for_model:
        if data[col].isnull().any():
            print(f"Warning: Column '{col}' contains NaNs. Filling with 0 before scaling/modeling.")
            data[col].fillna(0, inplace=True)


    # 2. Define Features (X) and Target (y)
    # S4 script saves 'cluster_labels' as the chosen genomic feature.
    feature_columns = [
        "lat", "long", "Population", "Epizootic_cases", "temp", "precipitation",
        "cluster_labels", "SpRich", "month_sin", "month_cos",
        "temp_precip", "pop_spRich", "Spillover_lag1", "Spillover_rolling3",
        "Spillover_cum", "Nearest_Spillover_Dist"
    ]
    # Verify all feature columns exist
    missing_features = [f for f in feature_columns if f not in data.columns]
    if missing_features:
        print(f"Error: The following feature columns are missing: {missing_features}. Exiting.")
        return
        
    target_column = "Spillover"

    # 3. Split data based on Year
    # Train: Year < 2018
    # Validation: Year > 2018 (Notebook used 2019-2021 for validation, and 2018 for plotting test)
    # Test: Year == 2018
    
    train_df = data[data['Year'] < 2018].copy()
    # Validation set: (Year > 2018 & Year < 2022) OR (Year == 2018 & Months >= 10)
    # Let's define validation as 2019-2024
    # Paper mentions validation 2019-2024, test 2018.
    val_df = data[(data['Year'] > 2018) & (data['Year'] <= 2024)].copy() 
    test_df = data[data['Year'] == 2018].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        print("Error: Train, validation, or test set is empty after splitting. Check year ranges and data.")
        return

    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_val = val_df[feature_columns]
    y_val = val_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    # 4. Scale Numerical Features
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    # Fit scaler on training data and transform all sets
    X_train_scaled[feature_columns] = scaler.fit_transform(X_train[feature_columns])
    X_val_scaled[feature_columns] = scaler.transform(X_val[feature_columns])
    X_test_scaled[feature_columns] = scaler.transform(X_test[feature_columns])
    print("Features scaled.")

    # --- 5. Random Forest ---
    print("\n--- Training Random Forest ---")
    # (class_weight for imbalance)
    rf_params = {
        'n_estimators': 100,
        'max_depth': 6, 
        'class_weight': {0: 1, 1: 30}, 
        'random_state': 42,
        'n_jobs': -1
    }
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train_scaled, y_train)
    print("Random Forest training complete.")

    # Evaluate RF on Validation set
    y_val_pred_proba_rf = rf_model.predict_proba(X_val_scaled)[:, 1]
    y_val_pred_binary_rf = (y_val_pred_proba_rf > 0.1).astype(int)
    print("\nRandom Forest - Validation Set Performance:")
    evaluate_classification_model("RF_Validation", y_val, y_val_pred_proba_rf, y_val_pred_binary_rf)

    # Evaluate RF on Test set
    y_test_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred_binary_rf = (y_test_pred_proba_rf > 0.3).astype(int)
    print("\nRandom Forest - Test Set Performance:")
    rf_test_metrics = evaluate_classification_model("RF_Test", y_test, y_test_pred_proba_rf, y_test_pred_binary_rf)
    
    # Save RF model and predictions
    joblib.dump(rf_model, MODELS_DIR / "random_forest_model.joblib")
    test_df_with_rf_preds = test_df.copy()
    test_df_with_rf_preds['RF_pred_proba'] = y_test_pred_proba_rf
    test_df_with_rf_preds['RF_pred_binary'] = y_test_pred_binary_rf
    test_df_with_rf_preds.to_csv(PROCESSED_DATA_DIR / "S5_test_predictions_rf.csv", index=False)

    # Plot RF Feature Importance
    plot_feature_importance(rf_model.feature_importances_, feature_columns, "RandomForest", output_dir=FIGURES_DIR)

    # --- 6. XGBoost ---
    print("\n--- Training XGBoost ---")
    
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc", # Can also use 'logloss', 'error'
        "scale_pos_weight": 20, 
        "learning_rate": 0.01,
        "max_depth": 6,
        "n_estimators": 100, # Number of boosting rounds
        "random_state": 42,
        "use_label_encoder": False # Suppress warning for newer XGBoost versions
    }
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_scaled, y_train,
                  eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
                  verbose=False, early_stopping_rounds=10) # Added early stopping
    print("XGBoost training complete.")

    # Evaluate XGB on Validation set
    y_val_pred_proba_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]
    # Threshold
    y_val_pred_binary_xgb = (y_val_pred_proba_xgb > 0.2).astype(int)
    print("\nXGBoost - Validation Set Performance:")
    evaluate_classification_model("XGB_Validation", y_val, y_val_pred_proba_xgb, y_val_pred_binary_xgb)

    # Evaluate XGB on Test set
    y_test_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred_binary_xgb = (y_test_pred_proba_xgb > 0.2).astype(int)
    print("\nXGBoost - Test Set Performance:")
    xgb_test_metrics = evaluate_classification_model("XGB_Test", y_test, y_test_pred_proba_xgb, y_test_pred_binary_xgb)

    # Save XGB model and predictions
    xgb_model.save_model(MODELS_DIR / "xgboost_model.json") # Standard way to save XGB
    test_df_with_xgb_preds = test_df.copy()
    test_df_with_xgb_preds['XGB_pred_proba'] = y_test_pred_proba_xgb
    test_df_with_xgb_preds['XGB_pred_binary'] = y_test_pred_binary_xgb
    test_df_with_xgb_preds.to_csv(PROCESSED_DATA_DIR / "S5_test_predictions_xgb.csv", index=False)

    plot_feature_importance(xgb_model.feature_importances_, feature_columns, "XGBoost", output_dir=FIGURES_DIR)

if __name__ == "__main__":
    main()