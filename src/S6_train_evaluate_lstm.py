# src/S6_train_evaluate_lstm.py
import pandas as pd
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt # For potential loss plots
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary # For model summary

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# Assuming utils.py is in the same directory (src) or PYTHONPATH is set
from utils import calculate_fp_fn
# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
RESULTS_DIR = PROJECT_ROOT / "results"
LSTM_MODEL_DIR = RESULTS_DIR / "trained_models" / "lstm"
LSTM_FIGURES_DIR = RESULTS_DIR / "figures" / "lstm"
LSTM_TABLES_DIR = RESULTS_DIR / "tables" / "lstm"

for dir_path in [LSTM_MODEL_DIR, LSTM_FIGURES_DIR, LSTM_TABLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Input file from S4
S4_FEATURE_ENGINEERED_DATA_FILE = PROCESSED_DATA_DIR / "S4_feature_engineered_data.csv"

# Model & Training Hyperparameters
SEQ_LENGTH = 3 
FEATURE_COLS_LSTM = [ 
    "lat", "long", "Population", "Epizootic_cases", "temp",
    "precipitation", "cluster_labels", "SpRich", "month_sin",
    "month_cos", "temp_precip", "pop_spRich", "Spillover_rolling3", #
    "Nearest_Spillover_Dist"
]
TARGET_COL_LSTM = "Spillover"

HIDDEN_SIZE_LSTM = 128
NUM_LAYERS_LSTM = 6
OUTPUT_SIZE_LSTM = 1
EMBEDDING_DIM_LSTM = 10
DROPOUT_PROB_LSTM = 0.7 # High dropout
BATCH_SIZE_LSTM = 512
NUM_EPOCHS_LSTM = 100
LEARNING_RATE_LSTM = 0.001
WEIGHT_DECAY_LSTM = 1e-5
POS_WEIGHT_LSTM = torch.tensor([30.0]) # For imbalanced classes
PREDICTION_THRESHOLD_LSTM = 0.2

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
POS_WEIGHT_LSTM = POS_WEIGHT_LSTM.to(DEVICE)


# --- Data Preparation for LSTM ---
def create_sequences_for_lstm(df, feature_cols, target_label, region_id_label, seq_len):
    """
    Creates sequences of past features, past targets, current features, and current target.
    Output:
        current_features_np: Features for the time step to predict.
        past_combined_seq_np: Sequence of (past_features + past_target) for LSTM input.
        current_target_np: Target label for the time step to predict.
        region_ids_np: Region ID for each sequence.
    """
    print(f"Creating sequences with length {seq_len}...")
    current_features_list = []
    past_combined_seq_list = [] # Will store [past_feat_1, past_target_1], [past_feat_2, past_target_2]...
    current_target_list = []
    region_ids_list = []

    df_sorted = df.sort_values(by=[region_id_label, "Year", "Months"])

    for _, group in tqdm(df_sorted.groupby(region_id_label), desc="Processing Regions for LSTM"):
        region_feature_values = group[feature_cols].values
        target_values = group[target_label].values # This is the actual spillover (0 or 1)
        region_id_val = group[region_id_label].iloc[0]

        if len(target_values) <= seq_len: # Need seq_len past points + 1 current point
            continue

        for i in range(len(region_feature_values) - seq_len):
            # Past sequence for LSTM input: (past_features + past_target_values)
            # LSTM input: past_target_values (from t-seq_len to t-1)
            # Current features: features at time t
            
            # Past target sequence (t-seq_len to t-1)
            past_target_seq = target_values[i : i + seq_len].reshape(-1, 1)
            # Past features sequence (t-seq_len to t-1)
            past_feat_seq = region_feature_values[i : i + seq_len]
            
            # Combine past features and past targets for LSTM input 
            # The model `LSTMWithPastTargetAndFeatures` seems to take only past_targets for LSTM
            # and current_features separately. Let's align with that.
            # This means the LSTM input should be a sequence of [features_at_t-k, target_at_t-k]
            
            # Corrected sequence creation for LSTM input:
            # Each step in the input sequence for the LSTM should have 15 features
            # (14 original features + 1 past target value)
            _past_combined_features_for_lstm_input = []
            for step_idx in range(seq_len):
                combined_step = np.concatenate(
                    (region_feature_values[i + step_idx], target_values[i + step_idx].reshape(1)), axis=0
                )
                _past_combined_features_for_lstm_input.append(combined_step)
            
            past_combined_seq_list.append(np.array(_past_combined_features_for_lstm_input)) # Shape: (seq_len, num_features+1)

            # Current features (at time t, which is i + seq_len)
            current_features_list.append(region_feature_values[i + seq_len])
            
            # Current target (at time t)
            current_target_list.append(target_values[i + seq_len])
            region_ids_list.append(region_id_val)

    return (np.array(current_features_list), np.array(past_combined_seq_list),
            np.array(current_target_list), np.array(region_ids_list))


class TimeSeriesRegionPastDataset(Dataset):
    """PyTorch Dataset for LSTM with current features and past sequence."""
    def __init__(self, X_current_features, X_past_combined_seq, X_region_id, y_labels):
        self.X_current_features = X_current_features
        self.X_past_combined_seq = X_past_combined_seq # This is what LSTM processes
        self.X_region_id = X_region_id
        self.y_labels = y_labels

    def __len__(self):
        return len(self.y_labels) # Length is determined by number of targets

    def __getitem__(self, idx):
        return (self.X_current_features[idx],
                self.X_past_combined_seq[idx],
                self.X_region_id[idx],
                self.y_labels[idx])

# --- LSTM Model Definition ---
class LSTMWithPastTargetAndFeatures(nn.Module):
    def __init__(self, current_feature_dim, lstm_input_dim, hidden_size, num_layers,
                 output_size, num_regions, embedding_dim, dropout_prob=0.5):
        super().__init__()
        self.region_embedding = nn.Embedding(num_regions, embedding_dim)
        
        # LSTM processes sequences of (past_features + past_target)
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout_prob if num_layers > 1 else 0)
        
        # Fully connected layer takes LSTM output, current features, and region embedding
        combined_fc_input_dim = hidden_size + current_feature_dim + embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_fc_input_dim, hidden_size), # First FC layer
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, output_size) # Output layer
        )

    def forward(self, current_features, past_combined_seq, region_id):
        # past_combined_seq shape: [batch, seq_len, lstm_input_dim]
        # current_features shape: [batch, current_feature_dim]
        # region_id shape: [batch]
        
        lstm_output, (hn, _) = self.lstm(past_combined_seq)
        # Use the hidden state of the last layer from the last time step
        lstm_processed = hn[-1] # Shape: [batch, hidden_size]

        region_embedded = self.region_embedding(region_id) # Shape: [batch, embedding_dim]

        combined_for_fc = torch.cat([lstm_processed, current_features, region_embedded], dim=1)
        
        output = self.fc(combined_for_fc) # Shape: [batch, output_size]
        return output


# --- Main Execution ---
def main():
    print("--- S6: Starting LSTM Model Training and Evaluation ---")

    # 1. Load feature-engineered data from S4
    try:
        data_full = pd.read_csv(S4_FEATURE_ENGINEERED_DATA_FILE)
        print(f"Loaded S4 data with shape: {data_full.shape}")
    except FileNotFoundError:
        print(f"Error: S4 output file not found at {S4_FEATURE_ENGINEERED_DATA_FILE}. Exiting.")
        return

    # Fill any NaNs - LSTMs are sensitive.
    data_full.fillna(0, inplace=True) # Simple fillna for now

    # 2. Encode 'Region' to 'Region_ID'
    region_encoder = LabelEncoder()
    data_full['Region_ID'] = region_encoder.fit_transform(data_full['Region'])
    NUM_UNIQUE_REGIONS = len(data_full['Region_ID'].unique())
    print(f"Number of unique regions: {NUM_UNIQUE_REGIONS}")

    # 3. Data Splitting (time-based)
    train_df = data_full[data_full['Year'] < 2018].copy()
    val_df = data_full[(data_full['Year'] > 2018) & (data_full['Year'] <= 2024)].copy() # Match S5
    test_df = data_full[data_full['Year'] == 2018].copy() # Match S5 for plotting/final eval

    print(f"Train df shape: {train_df.shape}, Val df shape: {val_df.shape}, Test df shape: {test_df.shape}")
    if train_df.empty or val_df.empty or test_df.empty:
        print("Error: One of the data splits is empty. Check data and year ranges.")
        return

    # 4. Scaling numerical features (from FEATURE_COLS_LSTM)
    # The current LSTM model uses 'current_features' which includes 'cluster_labels'.
    numerical_cols_to_scale = [col for col in FEATURE_COLS_LSTM if data_full[col].dtype in ['int64', 'float64']]
    
    scalers = {}
    for col in numerical_cols_to_scale:
        scaler = MinMaxScaler() 
        train_df[col] = scaler.fit_transform(train_df[[col]])
        val_df[col] = scaler.transform(val_df[[col]])
        test_df[col] = scaler.transform(test_df[[col]])
        scalers[col] = scaler
    print("Numerical features scaled.")

    # 5. Create sequences
    # Current features are those in FEATURE_COLS_LSTM
    # LSTM input sequence will consist of these features + the past target (Spillover)
    # So, lstm_input_dim = len(FEATURE_COLS_LSTM) + 1
    
    X_train_curr, X_train_past, y_train_seq, R_train_seq = create_sequences_for_lstm(
        train_df, FEATURE_COLS_LSTM, TARGET_COL_LSTM, 'Region_ID', SEQ_LENGTH
    )
    X_val_curr, X_val_past, y_val_seq, R_val_seq = create_sequences_for_lstm(
        val_df, FEATURE_COLS_LSTM, TARGET_COL_LSTM, 'Region_ID', SEQ_LENGTH
    )
    X_test_curr, X_test_past, y_test_seq, R_test_seq = create_sequences_for_lstm(
        test_df, FEATURE_COLS_LSTM, TARGET_COL_LSTM, 'Region_ID', SEQ_LENGTH
    )
    
    # Convert to Tensors
    X_train_curr_t = torch.tensor(X_train_curr, dtype=torch.float32)
    X_train_past_t = torch.tensor(X_train_past, dtype=torch.float32)
    y_train_seq_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
    R_train_seq_t = torch.tensor(R_train_seq, dtype=torch.long)

    X_val_curr_t = torch.tensor(X_val_curr, dtype=torch.float32)
    X_val_past_t = torch.tensor(X_val_past, dtype=torch.float32)
    y_val_seq_t = torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1)
    R_val_seq_t = torch.tensor(R_val_seq, dtype=torch.long)

    X_test_curr_t = torch.tensor(X_test_curr, dtype=torch.float32)
    X_test_past_t = torch.tensor(X_test_past, dtype=torch.float32)
    y_test_seq_t = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1)
    R_test_seq_t = torch.tensor(R_test_seq, dtype=torch.long)

    print(f"Train tensors: X_curr={X_train_curr_t.shape}, X_past={X_train_past_t.shape}, y={y_train_seq_t.shape}, R={R_train_seq_t.shape}")

    # 6. Create Datasets and DataLoaders
    train_dataset = TimeSeriesRegionPastDataset(X_train_curr_t, X_train_past_t, R_train_seq_t, y_train_seq_t)
    val_dataset = TimeSeriesRegionPastDataset(X_val_curr_t, X_val_past_t, R_val_seq_t, y_val_seq_t)
    test_dataset = TimeSeriesRegionPastDataset(X_test_curr_t, X_test_past_t, R_test_seq_t, y_test_seq_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_LSTM, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_LSTM, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_LSTM, shuffle=False)

    # 7. Initialize Model, Loss, Optimizer
    current_feature_dim = len(FEATURE_COLS_LSTM)
    lstm_input_dim = len(FEATURE_COLS_LSTM) + 1 # features + past_target
    
    model = LSTMWithPastTargetAndFeatures(
        current_feature_dim=current_feature_dim,
        lstm_input_dim=lstm_input_dim, # Input to LSTM cells
        hidden_size=HIDDEN_SIZE_LSTM,
        num_layers=NUM_LAYERS_LSTM,
        output_size=OUTPUT_SIZE_LSTM,
        num_regions=NUM_UNIQUE_REGIONS,
        embedding_dim=EMBEDDING_DIM_LSTM,
        dropout_prob=DROPOUT_PROB_LSTM
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT_LSTM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_LSTM, weight_decay=WEIGHT_DECAY_LSTM)

    # Print model summary (optional)
    # summary(model, input_data=(X_train_curr_t[:BATCH_SIZE_LSTM], X_train_past_t[:BATCH_SIZE_LSTM], R_train_seq_t[:BATCH_SIZE_LSTM]))
    # For summary, shapes must match exactly what forward expects.
    # Correct shapes for summary:
    # current_features_sample = torch.randn(BATCH_SIZE_LSTM, current_feature_dim).to(DEVICE)
    # past_combined_seq_sample = torch.randn(BATCH_SIZE_LSTM, SEQ_LENGTH, lstm_input_dim).to(DEVICE)
    # region_id_sample = torch.randint(0, NUM_UNIQUE_REGIONS, (BATCH_SIZE_LSTM,)).to(DEVICE)
    # summary(model, input_data=[current_features_sample, past_combined_seq_sample, region_id_sample])


    # 8. Training Loop 
    print("\n--- Starting LSTM Training ---")
    start_training_time = time.time()
    train_losses, val_losses = [], []
    best_val_f1 = -1 # For saving best model based on F1 or other metric

    for epoch in range(NUM_EPOCHS_LSTM):
        model.train()
        epoch_train_loss = 0
        all_train_labels_epoch, all_train_probs_epoch = [], []

        for X_curr_batch, X_past_batch, R_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS_LSTM} Training", leave=False):
            X_curr_batch = X_curr_batch.to(DEVICE)
            X_past_batch = X_past_batch.to(DEVICE) # This is the input for LSTM part
            R_batch = R_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_curr_batch, X_past_batch, R_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            all_train_labels_epoch.extend(y_batch.cpu().numpy().flatten())
            all_train_probs_epoch.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_preds_binary = (np.array(all_train_probs_epoch) > PREDICTION_THRESHOLD_LSTM).astype(int) # Use configured threshold
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            all_train_labels_epoch, train_preds_binary, average='binary', zero_division=0)
        train_auc_epoch = roc_auc_score(all_train_labels_epoch, all_train_probs_epoch)


        # Validation
        model.eval()
        epoch_val_loss = 0
        all_val_labels_epoch, all_val_probs_epoch = [], []
        with torch.no_grad():
            for X_curr_batch, X_past_batch, R_batch, y_batch in val_loader:
                X_curr_batch, X_past_batch, R_batch, y_batch = (
                    X_curr_batch.to(DEVICE), X_past_batch.to(DEVICE),
                    R_batch.to(DEVICE), y_batch.to(DEVICE)
                )
                outputs = model(X_curr_batch, X_past_batch, R_batch)
                loss = criterion(outputs, y_batch)
                epoch_val_loss += loss.item()
                all_val_labels_epoch.extend(y_batch.cpu().numpy().flatten())
                all_val_probs_epoch.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_preds_binary = (np.array(all_val_probs_epoch) > PREDICTION_THRESHOLD_LSTM).astype(int)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            all_val_labels_epoch, val_preds_binary, average='binary', zero_division=0)
        val_auc_epoch = roc_auc_score(all_val_labels_epoch, all_val_probs_epoch)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS_LSTM} | "
              f"Train Loss: {avg_train_loss:.4f} AUC: {train_auc_epoch:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} AUC: {val_auc_epoch:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), LSTM_MODEL_DIR / "best_lstm_model.pth")
            print(f"  Best model saved with Val F1: {best_val_f1:.4f}")

    end_training_time = time.time()
    print(f"LSTM Training complete! Total time: {(end_training_time - start_training_time)/60:.2f} minutes.")

    # 9. Final Evaluation on Test Set using the best model
    print("\n--- Evaluating Best LSTM Model on Test Set ---")
    model.load_state_dict(torch.load(LSTM_MODEL_DIR / "best_lstm_model.pth"))
    model.eval()
    all_test_labels, all_test_probs = [], []
    with torch.no_grad():
        for X_curr_batch, X_past_batch, R_batch, y_batch in test_loader:
            X_curr_batch, X_past_batch, R_batch, y_batch = (
                X_curr_batch.to(DEVICE), X_past_batch.to(DEVICE),
                R_batch.to(DEVICE), y_batch.to(DEVICE)
            )
            outputs = model(X_curr_batch, X_past_batch, R_batch)
            all_test_labels.extend(y_batch.cpu().numpy().flatten())
            all_test_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())

    test_preds_binary = (np.array(all_test_probs) > 0.3).astype(int)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        all_test_labels, test_preds_binary, average='binary', zero_division=0)
    test_auc = roc_auc_score(all_test_labels, all_test_probs)
    fp_test, fn_test = calculate_fp_fn(all_test_labels, test_preds_binary) # From utils

    print(f"Test Set | AUC: {test_auc:.4f} | Precision: {test_prec:.4f} | "
          f"Recall: {test_rec:.4f} | F1: {test_f1:.4f}")
    print(f"Test Set | False Positives: {fp_test}, False Negatives: {fn_test}")

    # Save test metrics and predictions
    lstm_test_metrics = {
        'model': 'LSTM_Test', 'auc': test_auc,
        'precision': test_prec, 'recall': test_rec, 'f1_score': test_f1,
        'fp': fp_test, 'fn': fn_test, 'threshold': 0.3
    }
    pd.DataFrame([lstm_test_metrics]).to_csv(LSTM_TABLES_DIR / "lstm_test_metrics.csv", index=False)

    # Save test predictions with original test_df context
    # Need to reconstruct the original test_df rows that correspond to the sequences
    # This is tricky because create_sequences_for_lstm drops initial rows.
    # For simplicity, save the predicted probabilities and binary labels directly for the sequences.
    # If you need to link back to original rows, the `test_df` used for `create_sequences_for_lstm`
    # and the `R_test_seq` (region IDs) would be needed to carefully reconstruct.
    
    # Save predictions made on the *sequenced* test data
    # Note: len(all_test_labels) will be less than len(test_df_original) due to sequence creation
    # This saves predictions for the 'predictable' part of test_df
    results_df_lstm_test = pd.DataFrame({
        'true_label': all_test_labels,
        'predicted_proba': all_test_probs,
        'predicted_binary_0.3_thresh': test_preds_binary
    })
    results_df_lstm_test.to_csv(PROCESSED_DATA_DIR / "S6_test_predictions_lstm.csv", index=False)

    # Plot training/validation loss (optional)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('LSTM Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LSTM_FIGURES_DIR / "lstm_loss_curve.pdf")
    print("LSTM loss curve saved.")
    plt.show()
    
    print("--- S6: Finished ---")

if __name__ == "__main__":
    main()