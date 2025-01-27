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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from src.models import GAN, MaskedLinear
from src.data_utils import get_data

# ---------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Paths (Adjust as needed)
# ---------------------------------------------------------------------
TRAIN_ABUNDANCE_PATH = "MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv"
TRAIN_METADATA_PATH = "MetaCardis_data/train_T2D_metadata.csv"
FEATURE_COLUMNS_PATH = "Results/MicroKPNN_encoder_confounder_free_plots/feature_columns.csv"
MAPPING_FILE_PATH = "Default_Database/species_ids.csv"  # optional mapping file

EDGE_LIST_PATH = "Default_Database/EdgeList.csv"
OUTPUT_DIR = "Results/MicroKPNN_encoder_confounder_free_plots"

# ---------------------------------------------------------------------
# Load merged data and features
# ---------------------------------------------------------------------
# 1) Merged training data
merged_data = get_data(TRAIN_ABUNDANCE_PATH, TRAIN_METADATA_PATH)

# 2) Feature columns
metadata_columns = pd.read_csv(TRAIN_METADATA_PATH).columns.tolist()
feature_columns = (
    pd.read_csv(FEATURE_COLUMNS_PATH, header=None)
    .squeeze("columns")
    .astype(str)
    .tolist()
)

# 3) Optional: Map feature names for visualization
if os.path.exists(MAPPING_FILE_PATH):
    mapping_df = pd.read_csv(MAPPING_FILE_PATH)
    taxon_to_species = dict(zip(mapping_df["taxon_id"].astype(str), mapping_df["species"]))
    visual_feature_columns = [taxon_to_species.get(col, col) for col in feature_columns]
else:
    # Fallback: no mapping file found
    visual_feature_columns = feature_columns

# 4) Ensure feature_columns exist in merged_data
merged_data.columns = merged_data.columns.astype(str)  # unify column types
feature_columns = [col for col in feature_columns if col in merged_data.columns]

# 5) Prepare feature matrix
X = merged_data[feature_columns].values
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# ---------------------------------------------------------------------
# Build mask
# ---------------------------------------------------------------------
def build_mask(edge_list: str, species: list) -> (torch.Tensor, dict):
    """
    Build a hierarchical binary mask from an edge list CSV for a MaskedLinear layer.
    
    Parameters
    ----------
    edge_list : str
        Path to the CSV file containing parent-child relationships.
    species : list
        List of species (child nodes) used to index mask rows.

    Returns
    -------
    (torch.Tensor, dict)
        - mask: shape [num_parents, num_children] with 1/0 indicating connections
        - parent_dict: dictionary mapping parent ID -> column index
    """
    edge_df = pd.read_csv(edge_list)
    edge_df["parent"] = edge_df["parent"].astype(str)
    
    parent_nodes = sorted(set(edge_df["parent"].tolist()))
    mask = torch.zeros(len(species), len(parent_nodes))

    parent_dict = {k: i for i, k in enumerate(parent_nodes)}
    child_dict = {k: i for i, k in enumerate(species)}

    for _, row in edge_df.iterrows():
        if row["child"] != "Unnamed: 0":
            child_idx = child_dict.get(str(row["child"]))
            parent_idx = parent_dict.get(row["parent"])
            if child_idx is not None and parent_idx is not None:
                mask[child_idx][parent_idx] = 1

    return mask.T, parent_dict


# Build mask based on abundance columns (species)
relative_abundance = pd.read_csv(TRAIN_ABUNDANCE_PATH, index_col=0)
species = relative_abundance.columns.tolist()
mask, parent_dict = build_mask(EDGE_LIST_PATH, species)

# Save parent_dict to a CSV for reference
pd.DataFrame(list(parent_dict.items()), columns=["key", "value"]).to_csv(
    os.path.join(OUTPUT_DIR, "parent_dict.csv"), index=False
)

# ---------------------------------------------------------------------
# Model Wrapper (Feature-Level SHAP)
# ---------------------------------------------------------------------
class ModelWrapper(torch.nn.Module):
    """
    Wraps the entire trained model to get final disease predictions
    from input features X. 
    (Used by SHAP to compute attributions wrt. input features.)
    """
    def __init__(self, model: GAN):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_features = self.model.encoder(x)
        predictions = self.model.disease_classifier(encoded_features)
        return predictions

# ---------------------------------------------------------------------
# Explain model (input features)
# ---------------------------------------------------------------------
def explain_model(model_paths: list, X_data: np.ndarray) -> None:
    """
    Applies SHAP to the *input features* for each fold's model.
    Saves bar and bee swarm plots, as well as CSV of SHAP values.

    Parameters
    ----------
    model_paths : list of str
        Paths to trained model .pth files (one per fold or run).
    X_data : np.ndarray
        Input features array of shape (N, D).
    """
    sample_data_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)

    for fold, model_path in enumerate(model_paths):
        print(f"Explaining model (input features) for Fold {fold + 1}")
        
        # Load trained model
        base_model = GAN(mask=mask, latent_dim=64, num_layers=1).to(device)
        base_model.load_state_dict(torch.load(model_path, map_location=device))
        base_model.eval()

        # Wrap model for SHAP
        wrapped_model = ModelWrapper(base_model)

        # Create SHAP DeepExplainer
        explainer = shap.DeepExplainer(wrapped_model, sample_data_tensor)
        shap_values = explainer.shap_values(sample_data_tensor)

        # If DeepExplainer returns a list with single element, unpack it
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]

        # 1) SHAP Bar Summary
        plt.figure(figsize=(15, 10))
        shap.summary_plot(
            shap_values,
            X_data,
            feature_names=visual_feature_columns,
            plot_type="bar",
            show=False
        )
        plt.title(f"SHAP (Features) - Fold {fold + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_summary_fold{fold+1}.png"))
        plt.close()

        # 2) SHAP Bee Swarm
        plt.figure(figsize=(15, 10))
        shap.summary_plot(
            shap_values,
            X_data,
            feature_names=visual_feature_columns,
            plot_type="dot",
            show=False
        )
        plt.title(f"SHAP Bee Swarm (Features) - Fold {fold + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_bee_swarm_fold{fold+1}.png"))
        plt.close()

        # 3) Save SHAP values to CSV
        shap_df = pd.DataFrame(shap_values, columns=visual_feature_columns)
        shap_csv_path = os.path.join(OUTPUT_DIR, f"shap_values_fold{fold+1}.csv")
        shap_df.to_csv(shap_csv_path, index=False)
        
        print(f"SHAP values (features) saved for Fold {fold + 1} -> {shap_csv_path}\n")

# ---------------------------------------------------------------------
# Explain First Hidden Layer
# ---------------------------------------------------------------------
def get_first_hidden_activations(model: GAN, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass x through the *first hidden layer* of the encoder:
      - index 0: MaskedLinear
      - index 1: BatchNorm
      - index 2: ReLU
    Returns activations of shape: [batch_size, hidden_dim].
    """
    out = model.encoder[0](x)   # MaskedLinear
    out = model.encoder[1](out) # BatchNorm
    out = model.encoder[2](out) # ReLU
    return out

class SubModel(torch.nn.Module):
    """
    Sub-model that takes first-layer activations as input,
    then runs the *rest* of the encoder + disease_classifier
    for final disease logits.
    """
    def __init__(self, original_model: GAN):
        super().__init__()
        # everything after the first hidden layer in model.encoder
        self.post_first_layer = torch.nn.Sequential(*list(original_model.encoder[3:]))

        # final disease classifier
        self.disease_classifier = original_model.disease_classifier

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.post_first_layer(h)
        logits = self.disease_classifier(x)
        return logits

def explain_first_hidden_layer(
    model_paths: list,
    X_data: np.ndarray,
    device: torch.device = device,
    background_size: int = 100,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Uses SHAP to explain the *first hidden-layer node importance* for
    final disease output.
    
    Steps:
      1) Run X_data through the first hidden layer (MaskedLinear + BN + ReLU).
      2) Build a SubModel (the rest of encoder + disease_classifier).
      3) Treat those hidden activations as features and apply SHAP.

    Parameters
    ----------
    model_paths : list of str
        List of trained model checkpoints, e.g. ["model_fold1.pth", ...].
    X_data : np.ndarray
        Original input data of shape (N, D).
    device : torch.device, optional
        Device to run computations ("cpu" or "cuda"), by default `device` from above.
    background_size : int, optional
        Number of random samples used as background for SHAP, by default 100.
    output_dir : str, optional
        Directory to store SHAP plots & CSV for each fold, by default OUTPUT_DIR.
    """
    sample_data_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)

    for fold, model_path in enumerate(model_paths):
        print(f"\n=== Explaining FIRST-HIDDEN-LAYER for Fold {fold + 1} ===")

        # 1) Load trained model
        base_model = GAN(mask=mask, latent_dim=64, num_layers=1).to(device)
        base_model.load_state_dict(torch.load(model_path, map_location=device))
        base_model.eval()

        # 2) Get the first-layer activations
        with torch.no_grad():
            hidden_activations = get_first_hidden_activations(base_model, sample_data_tensor)
        # hidden_activations shape: [N, d_hidden]

        # 3) Build sub-model to map hidden_activations -> disease logits
        sub_model = SubModel(base_model).to(device)
        sub_model.eval()

        # 4) Random background set for SHAP
        n_samples = hidden_activations.shape[0]
        idx = np.random.choice(n_samples, size=min(background_size, n_samples), replace=False)
        background = hidden_activations[idx]

        # 5) Create SHAP DeepExplainer
        explainer = shap.DeepExplainer(sub_model, background)

        # 6) Compute SHAP values for entire dataset's hidden_activations
        shap_values = explainer.shap_values(hidden_activations)
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]

        shap_values_np = shap_values
        hidden_activations_np = hidden_activations.cpu().numpy()

        # 7) Define node names
        d_hidden = shap_values_np.shape[1]
        node_names = [f"Node_{i}" for i in range(d_hidden)]

        # 7a) SHAP Bar Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_np,
            hidden_activations_np,
            feature_names=node_names,
            plot_type="bar",
            show=False
        )
        plt.title(f"SHAP Bar (First Hidden Layer) - Fold {fold+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_summary_firstlayer_fold{fold+1}.png"))
        plt.close()

        # 7b) SHAP Bee Swarm Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_np,
            hidden_activations_np,
            feature_names=node_names,
            plot_type="dot",
            show=False
        )
        plt.title(f"SHAP Bee Swarm (First Hidden Layer) - Fold {fold+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_bee_swarm_firstlayer_fold{fold+1}.png"))
        plt.close()

        # 8) Save SHAP values to CSV
        shap_df = pd.DataFrame(shap_values_np, columns=node_names)
        csv_path = os.path.join(output_dir, f"shap_values_first_hidden_fold{fold+1}.csv")
        shap_df.to_csv(csv_path, index=False)

        print(f"Saved first-layer SHAP CSV to: {csv_path}")

# ---------------------------------------------------------------------
# Example usage (uncomment if you want to run directly)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose you have 5 folds of trained models
    model_paths_example = [
        "Results/MicroKPNN_encoder_confounder_free_plots/trained_model1.pth",
        "Results/MicroKPNN_encoder_confounder_free_plots/trained_model2.pth",
        "Results/MicroKPNN_encoder_confounder_free_plots/trained_model3.pth",
        "Results/MicroKPNN_encoder_confounder_free_plots/trained_model4.pth",
        "Results/MicroKPNN_encoder_confounder_free_plots/trained_model5.pth"
    ]
    
    # 1) SHAP for input features
    explain_model(model_paths_example, X)

    # 2) SHAP for first hidden layer
    explain_first_hidden_layer(model_paths_example, X, device=device)
