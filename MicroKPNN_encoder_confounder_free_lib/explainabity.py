"""
explainability.py

This module provides functionality to generate SHAP explanations for:
1) Input features -> final disease output
2) First hidden-layer nodes -> final disease output

It uses a trained GAN model (with a hierarchical MaskedLinear layer),
loads data from CSV files, and saves SHAP plots/values to disk.
"""

import os
import sys
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local modules can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_root)

# Local imports from your project structure
from models import GAN
from data_utils import get_data
from config import config # Import config to get model parameters

# ---------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------
# Paths (Adjust as needed)
# ---------------------------------------------------------------------
# These paths are derived from your config.py and main.py
DATA_DIR = "dataset/MetaCardis_data"
RESULTS_DIR = "Results/MicroKPNN_encoder_confounder_free_plots"

# Data paths from your config file
TRAIN_ABUNDANCE_PATH = os.path.join(DATA_DIR, config["data"]["train_abundance_path"].split('/')[-1])
TRAIN_METADATA_PATH = os.path.join(DATA_DIR, config["data"]["train_metadata_path"].split('/')[-1])
TEST_ABUNDANCE_PATH = os.path.join(DATA_DIR, config["data"]["test_abundance_path"].split('/')[-1])
TEST_METADATA_PATH = os.path.join(DATA_DIR, config["data"]["test_metadata_path"].split('/')[-1])

# Results and prior knowledge paths
FEATURE_COLUMNS_PATH = os.path.join(RESULTS_DIR, "feature_columns.csv")
EDGE_LIST_PATH = os.path.join(RESULTS_DIR, "required_data/EdgeList.csv")
PARENT_DICT_PATH = os.path.join(RESULTS_DIR, "required_data/parent_dict_main.csv")



# ---------------------------------------------------------------------
# Load Data and Feature Information
# ---------------------------------------------------------------------
print("Loading and preparing data...")
# 1) Load the CLR-transformed and scaled training data
# get_data returns train and test, we only need train data for the background
train_merged, _ = get_data(
    TRAIN_ABUNDANCE_PATH, TRAIN_METADATA_PATH,
    TEST_ABUNDANCE_PATH, TEST_METADATA_PATH
)

# 2) Load the exact feature column order used during training
feature_columns = pd.read_csv(FEATURE_COLUMNS_PATH).iloc[:, 0].astype(str).tolist()

# ---------------------------------------------------------------------
# Map taxon IDs to species names
# ---------------------------------------------------------------------
species_map_df = pd.read_csv("Default_Database/species_ids.csv")
id_to_species = dict(zip(species_map_df["taxon_id"].astype(str), species_map_df["species"]))

# Replace IDs in feature_columns with species names if available
feature_columns_named = [
    id_to_species.get(str(col), col) for col in feature_columns
]


# 3) Prepare the feature matrix X for SHAP
# Ensure columns are in the correct order and convert to numpy array
X = train_merged[feature_columns].values
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
print(f"Data loaded. Feature matrix shape: {X.shape}")

# ---------------------------------------------------------------------
# Build the Taxonomic Mask
# ---------------------------------------------------------------------
def build_mask(edge_list_path: str, species_list: list, parent_dict_path: str) -> torch.Tensor:
    """Builds the binary mask connecting species to their parent taxonomic groups."""
    edge_df = pd.read_csv(edge_list_path)
    parent_map_df = pd.read_csv(parent_dict_path)

    parent_nodes = parent_map_df['Parent'].astype(str).tolist()
    parent_dict = {str(k): i for i, k in enumerate(parent_nodes)}
    child_dict = {k: i for i, k in enumerate(species_list)}

    mask = torch.zeros(len(species_list), len(parent_nodes))

    for _, row in edge_df.iterrows():
        child_id = str(row['child'])
        parent_id = str(row['parent'])
        if child_id in child_dict and parent_id in parent_dict:
            mask[child_dict[child_id], parent_dict[parent_id]] = 1
            
    return mask.T, parent_dict # Transpose to get [num_parents, num_children] for MaskedLinear

mask, parent_dict = build_mask(EDGE_LIST_PATH, feature_columns, PARENT_DICT_PATH)
print(f"Taxonomic mask built. Mask shape: {mask.shape}")

# ---------------------------------------------------------------------
# Model Wrapper for SHAP (Input Feature Level)
# ---------------------------------------------------------------------
class ModelWrapper(torch.nn.Module):
    """Wraps the GAN model to provide a simple (input -> output) interface for SHAP."""
    def __init__(self, model: GAN):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This function encapsulates the full forward pass for disease prediction
        encoded_features = self.model.encoder(x)
        predictions = self.model.disease_classifier(encoded_features)
        return predictions

# ---------------------------------------------------------------------
# Function to Explain Input Features
# ---------------------------------------------------------------------

def explain_input_features(model_paths: list, X_data: np.ndarray, background_size: int = 100) -> None:
    """
    Applies SHAP to explain the contribution of input features (species) to the
    final disease prediction for each model fold, and also produces averaged
    (across folds) bar and beeswarm plots.
    """
    # Use a smaller, random subset of the data as the background for the explainer
    background_size = min(background_size, X_data.shape[0])
    background_indices = np.random.choice(X_data.shape[0], background_size, replace=False)
    background_tensor = torch.tensor(X_data[background_indices], dtype=torch.float32).to(device)
    sample_data_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)

    # Collect per-fold SHAP arrays here (each: [n_samples, n_features])
    per_fold_shap_values = []

    for fold, model_path in enumerate(model_paths):
        print(f"\n--- Explaining input features for Fold {fold + 1} ---")

        # Instantiate model (must match training config)
        base_model = GAN(
            mask=mask,
            input_size=len(feature_columns),
            latent_dim=config["model"]["latent_dim"],
            num_encoder_layers=config["model"]["num_encoder_layers"],
            num_classifier_layers=config["model"]["num_classifier_layers"],
            dropout_rate=config["model"]["dropout_rate"],
            norm=config["model"]["norm"],
            activation=config["model"]["activation"],
            last_activation=config["model"]["last_activation"]
        ).to(device)

        base_model.load_state_dict(torch.load(model_path, map_location=device))
        base_model.eval()

        wrapped_model = ModelWrapper(base_model)
        explainer = shap.DeepExplainer(wrapped_model, background_tensor)
        shap_values = explainer.shap_values(sample_data_tensor, check_additivity=False)

        # Binary-classification handling (use positive class if list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        # Ensure 2D: [n_samples, n_features]
        if len(shap_values.shape) == 3:
            shap_values = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])

        # Keep for averaging later
        per_fold_shap_values.append(shap_values)

        # --------- Per-fold outputs (your existing ones) ----------
        # Bar (mean |SHAP|)
        plt.figure()
        shap.summary_plot(shap_values, X_data, feature_names=feature_columns_named, plot_type="bar", show=False)
        plt.title(f"Feature Importance (Mean SHAP) - Fold {fold + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"shap_bar_features_fold{fold+1}.png"))
        plt.close()

        # Beeswarm
        plt.figure()
        shap.summary_plot(shap_values, X_data, feature_names=feature_columns, show=False)
        plt.title(f"SHAP Summary of Features - Fold {fold + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"shap_beeswarm_features_fold{fold+1}.png"))
        plt.close()

        # Save CSV
        shap_df = pd.DataFrame(shap_values, columns=feature_columns)
        csv_path = os.path.join(RESULTS_DIR, f"shap_values_features_fold{fold+1}.csv")
        shap_df.to_csv(csv_path, index=False)
        print(f"Saved feature-level SHAP values to {csv_path}")

    # ---------------- AFTER LOOP: Averaged plots ----------------
    print("\n--- Building averaged SHAP across folds ---")
    # per_fold_shap_values: list of [n_samples, n_features]
    shap_stack = np.stack(per_fold_shap_values, axis=0)              # [n_folds, n_samples, n_features]

    # 1) Element-wise average across folds keeps per-sample structure
    shap_avg = shap_stack.mean(axis=0)                               # [n_samples, n_features]

    # Replace underscores with spaces for readability in plots
    feature_columns_named_clean = [name.replace("_", " ") for name in feature_columns_named]


    # 1a) Averaged bar plot: mean(|SHAP|) across samples (already averaged across folds)
    plt.figure()
    shap.summary_plot(shap_avg, X_data, feature_names=feature_columns_named_clean, plot_type="bar", show=False)
    plt.title("Feature Importance (Mean SHAP) - Averaged Across Folds")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_bar_features_avg_folds.pdf"))
    plt.close()

    # 1b) Averaged beeswarm: uses per-sample, per-feature SHAP averaged across folds
    plt.figure()
    shap.summary_plot(shap_avg, X_data, feature_names=feature_columns_named_clean, show=False)
    plt.title("SHAP Summary of Features - Averaged Across Folds")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_beeswarm_features_avg_folds.pdf"))
    plt.close()

    # Save averaged SHAP to CSV
    shap_avg_df = pd.DataFrame(shap_avg, columns=feature_columns)
    shap_avg_df.to_csv(os.path.join(RESULTS_DIR, "shap_values_features_avg_folds.csv"), index=False)
    print(f"Saved averaged SHAP values to {os.path.join(RESULTS_DIR, 'shap_values_features_avg_folds.csv')}")

    # 2) (Optional) “All folds concatenated” beeswarm for richer distribution
    shap_concat = np.concatenate(per_fold_shap_values, axis=0)       # [n_folds * n_samples, n_features]
    X_concat = np.tile(X_data, (len(per_fold_shap_values), 1))

    plt.figure()
    shap.summary_plot(shap_concat, X_concat, feature_names=feature_columns_named_clean, show=False)
    plt.title("SHAP Summary of Features - All Folds Concatenated")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_beeswarm_features_all_folds_concat.png"))
    plt.close()

    # --- Top 20 features by mean(|SHAP|) across folds ---
    mean_abs = np.mean(np.abs(shap_avg), axis=0)                  # [n_features]
    top_idx = np.argsort(-mean_abs)[:20]

    top_ids    = [feature_columns[i]        for i in top_idx]
    top_names  = [feature_columns_named[i]  for i in top_idx]
    top_scores = mean_abs[top_idx]

    print("\nTop 20 features (averaged across folds) by mean(|SHAP|):")
    for r, (nm, sc) in enumerate(zip(top_names, top_scores), 1):
        print(f"{r:2d}. {nm}  ({sc:.6f})")

    pd.DataFrame({
        "feature_id": top_ids,
        "feature_name": top_names,
        "mean_abs_shap": top_scores
    }).to_csv(os.path.join(RESULTS_DIR, "top20_features_avg.csv"), index=False)



# ---------------------------------------------------------------------
# Model Wrappers for SHAP (Hidden Layer Level)
# ---------------------------------------------------------------------
def get_first_hidden_activations(model: GAN, x: torch.Tensor) -> torch.Tensor:
    """Passes input through the first block of the encoder (MaskedLinear -> Norm -> Activation)."""
    # The first three layers of the encoder define the first hidden representation
    first_block = torch.nn.Sequential(*list(model.encoder[:3]))
    with torch.no_grad():
        activations = first_block(x)
    return activations

class SubModel(torch.nn.Module):
    """A sub-model that takes first-layer activations and computes the final prediction."""
    def __init__(self, original_model: GAN):
        super().__init__()
        # The rest of the encoder (after the first block)
        self.post_first_layer = torch.nn.Sequential(*list(original_model.encoder[3:]))
        # The final disease classifier
        self.disease_classifier = original_model.disease_classifier

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.post_first_layer(h)
        logits = self.disease_classifier(x)
        return logits

# ---------------------------------------------------------------------
# Function to Explain First Hidden Layer
# ---------------------------------------------------------------------

def explain_first_hidden_layer(model_paths: list, X_data: np.ndarray, parent_dict_path: str, background_size: int = 50) -> None:
    """
    SHAP for FIRST hidden layer -> output, per-fold + averaged-across-folds + (optional) concatenated.
    Saves per-fold bar/beeswarm/CSV, plus averaged bar/beeswarm/CSV, plus concatenated beeswarm.
    """
    # ---- Prepare the human-readable hidden node labels in the SAME order used to build the mask ----
    parent_map_df = pd.read_csv(parent_dict_path)
    # Prefer a readable column if available; fallback to 'Parent'
    for candidate in ["ParentName", "Name", "Label", "Parent"]:
        if candidate in parent_map_df.columns:
            name_col = candidate
            break
    else:
        name_col = parent_map_df.columns[0]  # last resort

    # We'll sanity-check the length later once we see the hidden dimension.
    parent_labels_full = parent_map_df[name_col].astype(str).tolist()

    # ---- Storage for aggregation ----
    per_fold_shap = []      # list of [N, H]
    per_fold_hidden = []    # list of [N, H] (the hidden activations used for coloring in beeswarm)

    for fold, model_path in enumerate(model_paths):
        print(f"\n--- Explaining first hidden layer for Fold {fold + 1} ---")

        # Load model
        base_model = GAN(
            mask=mask,
            input_size=len(feature_columns),
            latent_dim=config["model"]["latent_dim"],
            num_encoder_layers=config["model"]["num_encoder_layers"],
            num_classifier_layers=config["model"]["num_classifier_layers"],
            dropout_rate=config["model"]["dropout_rate"],
            norm=config["model"]["norm"],
            activation=config["model"]["activation"],
            last_activation=config["model"]["last_activation"]
        ).to(device)
        base_model.load_state_dict(torch.load(model_path, map_location=device))
        base_model.eval()

        # Hidden activations for whole dataset (first block of encoder)
        hidden_activations = get_first_hidden_activations(base_model, X_tensor)  # [N, H]
        hidden_activations_np = hidden_activations.detach().cpu().numpy()
        N, H = hidden_activations_np.shape

        # Finalize node labels once we know H
        if len(parent_labels_full) != H:
            print(f"[WARN] parent_dict has {len(parent_labels_full)} parents but hidden dim is {H}. "
                  f"{'Truncating' if len(parent_labels_full) > H else 'Padding'} labels to match.")
        parent_columns_named = (parent_labels_full[:H] if len(parent_labels_full) >= H
                                else parent_labels_full + [f"Hidden_{i}" for i in range(len(parent_labels_full), H)])

        # Sub-model (hidden -> output)
        sub_model = SubModel(base_model).to(device)
        sub_model.eval()

        # Background in hidden space
        background_size_eff = min(background_size, N)
        background_indices = np.random.choice(N, background_size_eff, replace=False)
        background = hidden_activations[background_indices]

        # SHAP on hidden -> output
        explainer = shap.DeepExplainer(sub_model, background)
        shap_values = explainer.shap_values(hidden_activations, check_additivity=False)

        # Binary-class handling
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        # Ensure 2D [N, H]
        if len(shap_values.shape) == 3:
            shap_values = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])

        # --- Save per-fold outputs ---
        # Bar (mean |SHAP|)
        plt.figure()
        shap.summary_plot(shap_values, hidden_activations_np, feature_names=parent_columns_named, plot_type="bar", show=False)
        plt.title(f"Hidden Node Importance (Mean SHAP) - Fold {fold + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"shap_bar_hidden_fold{fold+1}.png"))
        plt.close()

        # Beeswarm
        plt.figure()
        shap.summary_plot(shap_values, hidden_activations_np, feature_names=parent_columns_named, show=False)
        plt.title(f"SHAP Summary of Hidden Nodes - Fold {fold + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"shap_beeswarm_hidden_fold{fold+1}.png"))
        plt.close()

        # CSV per-fold
        pd.DataFrame(shap_values, columns=parent_columns_named).to_csv(
            os.path.join(RESULTS_DIR, f"shap_values_hidden_fold{fold+1}.csv"), index=False
        )

        # Keep for averaging/concatenation
        per_fold_shap.append(shap_values)         # [N, H]
        per_fold_hidden.append(hidden_activations_np)  # [N, H]

    # ---- AFTER LOOP: averaged & concatenated ----
    print("\n--- Building averaged SHAP across folds (hidden layer) ---")
    shap_stack = np.stack(per_fold_shap, axis=0)        # [F, N, H]
    H_stack = np.stack(per_fold_hidden, axis=0)         # [F, N, H]

    shap_avg = shap_stack.mean(axis=0)                  # [N, H]
    H_avg = H_stack.mean(axis=0)                        # [N, H]

    # Use the same labels we finalized in the last fold (dimensions match)
    # If you prefer, rebuild from parent_dict_path here again to be explicit
    parent_columns_named_final = parent_columns_named

    mean_abs_hidden = np.mean(np.abs(shap_avg), axis=0)   # [H]
    top_idx_hidden = np.argsort(-mean_abs_hidden)[:20]    # 20 columns

    custom_top20_names = [
    "Acetate production",
    "Enterocloster",
    "Community 22",
    "Lactate production",
    "D-fructose consumption",
    "Propionate production",
    "Veillonella",
    "Community 17",
    "Lactose consumption",
    "Haemophilus",
    "Cellobiose consumption",
    "Ethanol production",
    "Citrate consumption",
    "H₂O₂ production",
    "Valine consumption",
    "Mannitol consumption",
    "CO₂ production",
    "Glucose consumption",
    "Formate production",
    "Streptococcus",
    ]

    # Averaged bar (mean |SHAP| over samples; SHAP matrix already avg over folds)
    plt.figure()
    shap.summary_plot(shap_avg[:, top_idx_hidden], H_avg[:, top_idx_hidden], feature_names=custom_top20_names, plot_type="bar", show=False)
    plt.title("Hidden Node Importance (Mean SHAP) - Averaged Across Folds")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_bar_hidden_avg_folds.pdf"))
    plt.close()

    # Averaged beeswarm
    plt.figure()
    shap.summary_plot(parent_columns_named_final, H_avg, feature_names=parent_columns_named_final, show=False)
    plt.title("SHAP Summary of Hidden Nodes - Averaged Across Folds")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_beeswarm_hidden_avg_folds.pdf"))
    plt.close()

    # CSV averaged
    pd.DataFrame(shap_avg, columns=parent_columns_named_final).to_csv(
        os.path.join(RESULTS_DIR, "shap_values_hidden_avg_folds.csv"), index=False
    )

    # (Optional) all folds concatenated (shows variability)
    shap_concat = shap_stack.reshape(-1, shap_stack.shape[-1])   # [F*N, H]
    H_concat = H_stack.reshape(-1, H_stack.shape[-1])            # [F*N, H]

    plt.figure()
    shap.summary_plot(shap_concat, H_concat, feature_names=parent_columns_named_final, show=False)
    plt.title("SHAP Summary of Hidden Nodes - All Folds Concatenated")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_beeswarm_hidden_all_folds_concat.png"))
    plt.close()

    # --- Top 20 hidden nodes by mean(|SHAP|) across folds ---
    mean_abs_hidden = np.mean(np.abs(shap_avg), axis=0)   # [H]
    top_idx_hidden = np.argsort(-mean_abs_hidden)[:20]

    top_hidden_names  = [parent_columns_named_final[i].replace("_", " ") for i in top_idx_hidden]
    top_hidden_scores = mean_abs_hidden[top_idx_hidden]

    print("\nTop 20 hidden nodes (averaged across folds) by mean(|SHAP|):")
    for r, (nm, sc) in enumerate(zip(top_hidden_names, top_hidden_scores), 1):
        print(f"{r:2d}. {nm}  ({sc:.6f})")

    pd.DataFrame({
        "hidden_node": top_hidden_names,
        "mean_abs_shap": top_hidden_scores
    }).to_csv(os.path.join(RESULTS_DIR, "top20_hidden_avg.csv"), index=False)


# ---------------------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # List of trained model paths from the 5-fold cross-validation
    model_paths = [
        os.path.join(RESULTS_DIR, f"trained_model_fold{i+1}.pth") for i in range(5)
    ]
    
    # Verify that all model files exist before running
    missing_files = [path for path in model_paths if not os.path.exists(path)]
    if missing_files:
        print("Error: The following model files are missing:")
        for path in missing_files:
            print(f"- {path}")
        sys.exit(1)

    # 1. Generate explanations for the original input features (species)
    explain_input_features(model_paths, X, background_size=100)

    # 2. Generate explanations for the first hidden layer nodes (taxonomic groups)
    explain_first_hidden_layer(model_paths, X, PARENT_DICT_PATH, background_size=100, )
    
    print("\n✅ Explainability analysis complete.")