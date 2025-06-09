# src/S7_train_evaluate_autoencoders.py
import pandas as pd
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# from torchinfo import summary # Can be used if desired

from sklearn.preprocessing import RobustScaler # As used in notebook for AE
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# Assuming utils.py is in the same directory (src) or PYTHONPATH is set
from utils import calculate_confusion_counts

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
RESULTS_DIR = PROJECT_ROOT / "results"
AE_MODEL_DIR = RESULTS_DIR / "trained_models" / "autoencoders"
AE_FIGURES_DIR = RESULTS_DIR / "figures" / "autoencoders"
AE_TABLES_DIR = RESULTS_DIR / "tables" / "autoencoders"

for dir_path in [AE_MODEL_DIR, AE_FIGURES_DIR, AE_TABLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Input file from S4
S4_FEATURE_ENGINEERED_DATA_FILE = PROCESSED_DATA_DIR / "S4_feature_engineered_data.csv"

# Autoencoder
AE_FEATURES = [ 
    "Population", "Epizootic_cases", "temp", "precipitation",
    "SpRich", "month_sin", "month_cos", "temp_precip", "pop_spRich",
    "Spillover_lag1", "cluster_labels", "Spillover_rolling3_prev",
    "Nearest_Spillover_Dist", "Spillover_cum_prev"
]
AE_TARGET_COL = "Spillover"

AE_INPUT_DIM = len(AE_FEATURES) # Should be 14 
AE_BATCH_SIZE = 512
AE_NUM_EPOCHS = 50 # As in notebook
AE_LEARNING_RATE = 0.0001 # For Autoencoder (L1Loss)
AE_THRESHOLD_STD_FACTOR = 0.1 # For anomaly threshold (mean + factor*std)

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Dataset Classes ---
class AutoencoderDatasetTrain(Dataset): # For training AE (only normal data)
    def __init__(self, X_data_tensor):
        self.X = X_data_tensor
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx] # Only features, AE learns to reconstruct them

class AutoencoderDatasetEval(Dataset): # For evaluating AE (features + true labels)
    def __init__(self, X_data_tensor, y_labels_tensor):
        self.X = X_data_tensor
        self.y = y_labels_tensor
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Autoencoder Model Definition ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim=AE_INPUT_DIM): # input_dim was 14 in notebook
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 512) # Bottleneck / encoded representation
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(), # Added missing 128 in decoder based on symmetry
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, input_dim) # Output matches input dim
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- Helper for Reconstruction Error and Evaluation ---
def compute_reconstruction_errors_and_labels(model, dataloader, device, is_vae=False):
    model.eval()
    all_errors = []
    all_true_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            if is_vae:
                reconstructed, _, _ = model(X_batch)
            else: # Autoencoder
                reconstructed = model(X_batch)
            
            # Per-sample MSE reconstruction error
            loss_per_sample = torch.mean((reconstructed - X_batch)**2, dim=1)
            all_errors.extend(loss_per_sample.cpu().numpy())
            all_true_labels.extend(y_batch.cpu().numpy())
    return np.array(all_errors), np.array(all_true_labels)

# --- Main Execution ---
def main():
    print("--- S7: Starting Autoencoder and VAE Training and Evaluation ---")

    # 1. Load feature-engineered data from S4
    try:
        data = pd.read_csv(S4_FEATURE_ENGINEERED_DATA_FILE)
        print(f"Loaded S4 data with shape: {data.shape}")
    except FileNotFoundError:
        print(f"Error: S4 output file not found at {S4_FEATURE_ENGINEERED_DATA_FILE}. Exiting.")
        return

    # Select features for Autoencoders
    missing_ae_features = [f for f in AE_FEATURES if f not in data.columns]
    if missing_ae_features:
        print(f"Error: AE features missing: {missing_ae_features}. Exiting.")
        return
        
    data_ae_full = data[AE_FEATURES + [AE_TARGET_COL] + ['Year']].copy()
    data_ae_full.fillna(0, inplace=True) # Simple fill for any remaining NaNs

    # 2. Data Splitting 
    # AE train: Year < 2018 AND Spillover == 0 (normal data)
    # AE val: Year > 2018 (all data, for anomaly detection eval)
    # AE test: Year == 2018 (all data, for anomaly detection eval)
    
    train_ae_df_normal = data_ae_full[
        (data_ae_full['Year'] < 2018) & (data_ae_full[AE_TARGET_COL] == 0)
    ].copy()
    val_ae_df = data_ae_full[data_ae_full['Year'] > 2018].copy() # Using 2019+ for validation
    test_ae_df = data_ae_full[data_ae_full['Year'] == 2018].copy()

    print(f"AE Train (normal) shape: {train_ae_df_normal.shape}")
    print(f"AE Val shape: {val_ae_df.shape}")
    print(f"AE Test shape: {test_ae_df.shape}")

    if train_ae_df_normal.empty:
        print("Error: AE training set (normal data) is empty. Cannot train Autoencoder.")
        return

    X_train_ae_features = train_ae_df_normal[AE_FEATURES]
    X_val_ae_features = val_ae_df[AE_FEATURES]
    y_val_ae_labels = val_ae_df[AE_TARGET_COL]
    X_test_ae_features = test_ae_df[AE_FEATURES]
    y_test_ae_labels = test_ae_df[AE_TARGET_COL]

    # 3. Scaling 
    # Note: Notebook scales all features based on `data_input_train` which might differ slightly
    # from `train_ae_df_normal[AE_FEATURES]`. Here, fit on AE's specific normal training data.
    scaler_ae = RobustScaler()
    X_train_ae_scaled = scaler_ae.fit_transform(X_train_ae_features)
    X_val_ae_scaled = scaler_ae.transform(X_val_ae_features)
    X_test_ae_scaled = scaler_ae.transform(X_test_ae_features)
    print("AE/VAE features scaled with RobustScaler.")

    # Convert to Tensors
    X_train_ae_tensor = torch.tensor(X_train_ae_scaled, dtype=torch.float32).to(DEVICE)
    X_val_ae_tensor = torch.tensor(X_val_ae_scaled, dtype=torch.float32).to(DEVICE)
    y_val_ae_tensor = torch.tensor(y_val_ae_labels.values, dtype=torch.float32).to(DEVICE)
    X_test_ae_tensor = torch.tensor(X_test_ae_scaled, dtype=torch.float32).to(DEVICE)
    y_test_ae_tensor = torch.tensor(y_test_ae_labels.values, dtype=torch.float32).to(DEVICE)

    # Create DataLoaders
    train_ae_dataset = AutoencoderDatasetTrain(X_train_ae_tensor)
    val_ae_dataset = AutoencoderDatasetEval(X_val_ae_tensor, y_val_ae_tensor)
    test_ae_dataset = AutoencoderDatasetEval(X_test_ae_tensor, y_test_ae_tensor)

    train_ae_loader = DataLoader(train_ae_dataset, batch_size=AE_BATCH_SIZE, shuffle=True)
    val_ae_loader = DataLoader(val_ae_dataset, batch_size=AE_BATCH_SIZE, shuffle=False)
    test_ae_loader = DataLoader(test_ae_dataset, batch_size=AE_BATCH_SIZE, shuffle=False)

    # --- STANDARD AUTOENCODER ---
    print("\n--- Training Standard Autoencoder ---")
    ae_model = Autoencoder(input_dim=AE_INPUT_DIM).to(DEVICE)
    # criterion_ae = nn.L1Loss() 
    criterion_ae = nn.MSELoss() # MSE is more common for reconstruction error calculation later
    optimizer_ae = optim.Adam(ae_model.parameters(), lr=AE_LEARNING_RATE)
    
    ae_train_recon_errors_all_epochs = [] # To store all training reconstruction errors

    for epoch in range(AE_NUM_EPOCHS):
        ae_model.train()
        epoch_loss = 0
        epoch_train_batch_recon_errors = []
        for X_batch_train in train_ae_loader: # Only X_batch for unsupervised AE training
            X_batch_train = X_batch_train.to(DEVICE)
            optimizer_ae.zero_grad()
            outputs = ae_model(X_batch_train)
            loss = criterion_ae(outputs, X_batch_train)
            loss.backward()
            optimizer_ae.step()
            epoch_loss += loss.item()
            
            # Store per-sample MSE reconstruction errors for this batch
            with torch.no_grad():
                batch_errors = torch.mean((outputs - X_batch_train)**2, dim=1).cpu().numpy()
                epoch_train_batch_recon_errors.extend(batch_errors)

        avg_epoch_loss = epoch_loss / len(train_ae_loader)
        ae_train_recon_errors_all_epochs.extend(epoch_train_batch_recon_errors) # Append errors from this epoch
        print(f"AE Epoch {epoch+1}/{AE_NUM_EPOCHS}, Train Loss: {avg_epoch_loss:.6f}")

        # Dynamic threshold calculation (optional, or calculate once after all training)
        # if (epoch + 1) % 10 == 0 or epoch == AE_NUM_EPOCHS -1: # Evaluate on Val periodically
            # current_threshold = np.mean(epoch_train_batch_recon_errors) + \
            #                     AE_THRESHOLD_STD_FACTOR * np.std(epoch_train_batch_recon_errors)
            # errors_val, labels_val = compute_reconstruction_errors_and_labels(ae_model, val_ae_loader, DEVICE)
            # preds_val = (errors_val > current_threshold).astype(int)
            # auc_val = roc_auc_score(labels_val, errors_val) # AUC on raw errors
            # prec_val, rec_val, f1_val, _ = precision_recall_fscore_support(labels_val, preds_val, average='binary', zero_division=0)
            # print(f"  Val Interim | Threshold: {current_threshold:.6f} AUC: {auc_val:.4f} F1: {f1_val:.4f}")

    print("Autoencoder training complete.")
    torch.save(ae_model.state_dict(), AE_MODEL_DIR / "autoencoder_model.pth")

    # Final threshold calculation based on ALL training reconstruction errors
    final_ae_threshold = np.mean(ae_train_recon_errors_all_epochs) + \
                         AE_THRESHOLD_STD_FACTOR * np.std(ae_train_recon_errors_all_epochs)
    print(f"Final AE Anomaly Threshold: {final_ae_threshold:.6f}")

    # Evaluate AE on Validation Set
    print("\nAutoencoder - Validation Set Performance:")
    errors_val_ae, labels_val_ae = compute_reconstruction_errors_and_labels(ae_model, val_ae_loader, DEVICE)
    preds_binary_val_ae = (errors_val_ae > final_ae_threshold).astype(int)
    auc_val_ae = roc_auc_score(labels_val_ae, errors_val_ae) # AUC on raw errors is often preferred
    prec_val_ae, rec_val_ae, f1_val_ae, _ = precision_recall_fscore_support(labels_val_ae, preds_binary_val_ae, average='binary', zero_division=0)
    print(f"  AUC: {auc_val_ae:.4f}, Precision: {prec_val_ae:.4f}, Recall: {rec_val_ae:.4f}, F1: {f1_val_ae:.4f}")
    pd.DataFrame([{'model':'AE_Validation', 'auc':auc_val_ae, 'precision':prec_val_ae, 'recall':rec_val_ae, 'f1_score':f1_val_ae, 'threshold':final_ae_threshold}])\
        .to_csv(AE_TABLES_DIR / "ae_validation_metrics.csv", index=False)


    # Evaluate AE on Test Set
    print("\nAutoencoder - Test Set Performance:")
    errors_test_ae, labels_test_ae = compute_reconstruction_errors_and_labels(ae_model, test_ae_loader, DEVICE)
    preds_binary_test_ae = (errors_test_ae > final_ae_threshold).astype(int)
    auc_test_ae = roc_auc_score(labels_test_ae, errors_test_ae)
    prec_test_ae, rec_test_ae, f1_test_ae, _ = precision_recall_fscore_support(labels_test_ae, preds_binary_test_ae, average='binary', zero_division=0)
    print(f"  AUC: {auc_test_ae:.4f}, Precision: {prec_test_ae:.4f}, Recall: {rec_test_ae:.4f}, F1: {f1_test_ae:.4f}")
    pd.DataFrame([{'model':'AE_Test', 'auc':auc_test_ae, 'precision':prec_test_ae, 'recall':rec_test_ae, 'f1_score':f1_test_ae, 'threshold':final_ae_threshold}])\
        .to_csv(AE_TABLES_DIR / "ae_test_metrics.csv", index=False)

    # Save AE predictions for test set 
    test_ae_df_with_preds = test_ae_df.copy() # Original test_ae_df before scaling
    test_ae_df_with_preds['AE_Reconstruction_Error'] = errors_test_ae
    test_ae_df_with_preds['AE_Predicted_Anomaly'] = preds_binary_test_ae
    test_ae_df_with_preds.to_csv(PROCESSED_DATA_DIR / "S7_test_predictions_ae.csv", index=False)

    print("--- S7: Finished ---")

if __name__ == "__main__":
    main()