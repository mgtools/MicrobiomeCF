"""
Main script to train and evaluate the MicroKPNN encoder with confounder control.
"""

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

# Local imports
from src.data_utils import get_data
from src.models import GAN, PearsonCorrelationLoss
from src.train import train_model
from src.utils import create_stratified_dataloader



####################
# Global Functions #
####################

def plot_confusion_matrix(
    conf_matrix: np.ndarray, 
    title: str, 
    save_path: str, 
    class_names=None
):
    """
    Plot and save a confusion matrix figure.
    """
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=class_names
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_metrics_by_fold(
    fold_idx: int,
    metrics_dict: dict,
    save_dir: str
):
    """
    Plots metrics (losses, accuracy, F1, AUCPR, etc.) for a single fold
    over the training epochs.
    """
    train_metrics = metrics_dict['train']
    val_metrics   = metrics_dict['val']
    test_metrics  = metrics_dict['test']

    # Retrieve number of epochs from length of any metric's history
    num_epochs = len(train_metrics['gloss_history'])
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(20, 15))

    # 1. correlation g Loss
    plt.subplot(3, 3, 1)
    plt.plot(epochs, train_metrics['gloss_history'], label='Train - G Loss')
    plt.title("correlation g Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 2. Distance correlation
    plt.subplot(3, 3, 2)
    plt.plot(epochs, train_metrics['dcor_history'], label='Train')
    plt.plot(epochs, test_metrics['dcor_history'], label='Test')
    plt.title("Distance Correlation")
    plt.xlabel("Epoch")
    plt.ylabel("Distance Correlation")
    plt.legend()

    # 3. Disease Loss
    plt.subplot(3, 3, 3)
    plt.plot(epochs, train_metrics['loss_history'], label='Train')
    plt.plot(epochs, test_metrics['loss_history'], label='Test')
    plt.title("Disease Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 4. Accuracy
    plt.subplot(3, 3, 4)
    plt.plot(epochs, train_metrics['accuracy'], label='Train')
    plt.plot(epochs, test_metrics['accuracy'], label='Test')
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # 5. F1 Score
    plt.subplot(3, 3, 5)
    plt.plot(epochs, train_metrics['f1_score'], label='Train')
    plt.plot(epochs, test_metrics['f1_score'], label='Test')
    plt.title("F1 Score History")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    # 6. AUCPR
    plt.subplot(3, 3, 6)
    plt.plot(epochs, train_metrics['auc_pr'], label='Train')
    plt.plot(epochs, test_metrics['auc_pr'], label='Test')
    plt.title("AUCPR Score History")
    plt.xlabel("Epoch")
    plt.ylabel("AUCPR")
    plt.legend()

    # 7. Precision
    plt.subplot(3, 3, 7)
    plt.plot(epochs, train_metrics['precision'], label='Train')
    plt.plot(epochs, test_metrics['precision'], label='Test')
    plt.title("Precision Score History")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    # 8. Recall
    plt.subplot(3, 3, 8)
    plt.plot(epochs, train_metrics['recall'], label='Train')
    plt.plot(epochs, test_metrics['recall'], label='Test')
    plt.title("Recall Score History")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fold_{fold_idx}_metrics.png'))
    plt.close()

def average_and_plot_metrics(
    train_metrics_per_fold: list,
    val_metrics_per_fold: list,
    test_metrics_per_fold: list,
    save_dir: str
):
    """
    Aggregate metrics across folds for train, val, and test. 
    Only 'train' metrics have 'gloss_history', so we skip that key for val/test.
    
    Parameters
    ----------
    train_metrics_per_fold : list
        List of dictionaries containing training metrics for each fold.
    val_metrics_per_fold : list
        List of dictionaries containing validation metrics for each fold.
    test_metrics_per_fold : list
        List of dictionaries containing test metrics for each fold.
    save_dir : str
        Directory to save plots and metrics files.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Number of folds
    n_folds = len(train_metrics_per_fold)
    
    # ---- 1) Identify metric keys for train vs. val/test ----
    # TRAIN includes 'gloss_history' (generator loss)
    # VAL/TEST do NOT include 'gloss_history'.
    train_keys = list(train_metrics_per_fold[0].keys())  # includes 'gloss_history'
    # We separate confusion_matrix from scalar metrics
    train_scalar_keys = [k for k in train_keys if k != "confusion_matrix"]

    # For validation
    val_keys = list(val_metrics_per_fold[0].keys())  # won't have 'gloss_history'
    val_scalar_keys = [k for k in val_keys if k != "confusion_matrix"]
    
    # For test
    test_keys = list(test_metrics_per_fold[0].keys())  # won't have 'gloss_history'
    test_scalar_keys = [k for k in test_keys if k != "confusion_matrix"]

    # Number of epochs (using train's 'gloss_history' as reference)
    num_epochs = len(train_metrics_per_fold[0]['gloss_history'])

    # ---- 2) Initialize accumulators for each subset (train, val, test) ----
    # Train accumulators
    train_avg = {k: np.zeros(num_epochs) for k in train_scalar_keys}
    train_conf_matrix_sum = [
        np.zeros_like(train_metrics_per_fold[0]['confusion_matrix'][0])
        for _ in range(num_epochs)
    ]

    # Val accumulators
    val_avg = {k: np.zeros(num_epochs) for k in val_scalar_keys}
    val_conf_matrix_sum = [
        np.zeros_like(val_metrics_per_fold[0]['confusion_matrix'][0])
        for _ in range(num_epochs)
    ]

    # Test accumulators
    test_avg = {k: np.zeros(num_epochs) for k in test_scalar_keys}
    test_conf_matrix_sum = [
        np.zeros_like(test_metrics_per_fold[0]['confusion_matrix'][0])
        for _ in range(num_epochs)
    ]

    # ---- 3) Accumulate metrics from each fold ----
    for fold_idx in range(n_folds):
        # --- Train ---
        for scalar_k in train_scalar_keys:
            train_avg[scalar_k] += np.array(train_metrics_per_fold[fold_idx][scalar_k])
        for ep_idx, cm in enumerate(train_metrics_per_fold[fold_idx]['confusion_matrix']):
            train_conf_matrix_sum[ep_idx] += cm

        # --- Val ---
        for scalar_k in val_scalar_keys:
            val_avg[scalar_k] += np.array(val_metrics_per_fold[fold_idx][scalar_k])
        for ep_idx, cm in enumerate(val_metrics_per_fold[fold_idx]['confusion_matrix']):
            val_conf_matrix_sum[ep_idx] += cm

        # --- Test ---
        for scalar_k in test_scalar_keys:
            test_avg[scalar_k] += np.array(test_metrics_per_fold[fold_idx][scalar_k])
        for ep_idx, cm in enumerate(test_metrics_per_fold[fold_idx]['confusion_matrix']):
            test_conf_matrix_sum[ep_idx] += cm

    # Divide by number of folds to get average
    for k in train_avg.keys():
        train_avg[k] /= n_folds
    for k in val_avg.keys():
        val_avg[k] /= n_folds
    for k in test_avg.keys():
        test_avg[k] /= n_folds

    # Average confusion matrices
    train_conf_matrix_avg = [cm / n_folds for cm in train_conf_matrix_sum]
    val_conf_matrix_avg   = [cm / n_folds for cm in val_conf_matrix_sum]
    test_conf_matrix_avg  = [cm / n_folds for cm in test_conf_matrix_sum]

    # ---- 4) Plot metrics (averaged) ----
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(20, 15))

    # (a) correlation g Loss (exists only in train)
    if 'gloss_history' in train_avg:
        plt.subplot(3, 3, 1)
        plt.plot(epochs, train_avg['gloss_history'], label='Train - G Loss (Avg)')
        plt.title("Average correlation g Loss (Train Only)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

    # (b) Distance correlation
    if 'dcor_history' in train_avg:
        plt.subplot(3, 3, 2)
        plt.plot(epochs, train_avg['dcor_history'], label='Train')
    if 'dcor_history' in test_avg:
        plt.plot(epochs, test_avg['dcor_history'],  label='Test')
    plt.title("Average Distance Correlation")
    plt.xlabel("Epoch")
    plt.ylabel("Distance Correlation")
    plt.legend()

    # (c) Disease Loss
    if 'loss_history' in train_avg:
        plt.subplot(3, 3, 3)
        plt.plot(epochs, train_avg['loss_history'], label='Train')
    if 'loss_history' in test_avg:
        plt.plot(epochs, test_avg['loss_history'],  label='Test')
    plt.title("Average Disease Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # (d) Accuracy
    if 'accuracy' in train_avg:
        plt.subplot(3, 3, 4)
        plt.plot(epochs, train_avg['accuracy'], label='Train')
    if 'accuracy' in test_avg:
        plt.plot(epochs, test_avg['accuracy'], label='Test')
    plt.title("Average Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # (e) F1 Score
    if 'f1_score' in train_avg:
        plt.subplot(3, 3, 5)
        plt.plot(epochs, train_avg['f1_score'], label='Train')
    if 'f1_score' in test_avg:
        plt.plot(epochs, test_avg['f1_score'], label='Test')
    plt.title("Average F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()

    # (f) AUCPR
    if 'auc_pr' in train_avg:
        plt.subplot(3, 3, 6)
        plt.plot(epochs, train_avg['auc_pr'], label='Train')
    if 'auc_pr' in test_avg:
        plt.plot(epochs, test_avg['auc_pr'],  label='Test')
    plt.title("Average AUCPR")
    plt.xlabel("Epoch")
    plt.ylabel("AUCPR")
    plt.legend()

    # (g) Precision
    if 'precision' in train_avg:
        plt.subplot(3, 3, 7)
        plt.plot(epochs, train_avg['precision'], label='Train')
    if 'precision' in test_avg:
        plt.plot(epochs, test_avg['precision'], label='Test')
    plt.title("Average Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    # (h) Recall
    if 'recall' in train_avg:
        plt.subplot(3, 3, 8)
        plt.plot(epochs, train_avg['recall'], label='Train')
    if 'recall' in test_avg:
        plt.plot(epochs, test_avg['recall'], label='Test')
    plt.title("Average Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'average_metrics.png'))
    plt.close()

    # ---- 5) Plot average confusion matrices (final epoch) ----
    def plot_confusion_matrix(conf_matrix, title, save_path, class_names=None):
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        plt.savefig(save_path)
        plt.close()

    final_epoch_idx = num_epochs - 1
    plot_confusion_matrix(
        train_conf_matrix_avg[final_epoch_idx],
        title='Average Train Confusion Matrix (Final Epoch)',
        save_path=os.path.join(save_dir, 'average_train_conf_matrix.png'),
        class_names=['Class 0', 'Class 1']
    )
    plot_confusion_matrix(
        val_conf_matrix_avg[final_epoch_idx],
        title='Average Validation Confusion Matrix (Final Epoch)',
        save_path=os.path.join(save_dir, 'average_val_conf_matrix.png'),
        class_names=['Class 0', 'Class 1']
    )
    plot_confusion_matrix(
        test_conf_matrix_avg[final_epoch_idx],
        title='Average Test Confusion Matrix (Final Epoch)',
        save_path=os.path.join(save_dir, 'average_test_conf_matrix.png'),
        class_names=['Class 0', 'Class 1']
    )

    # ---- 6) Save a metrics.pkl for further analysis ----
    avg_metrics_pickle = {
        'train_avg_metrics': train_avg,
        'val_avg_metrics': val_avg,
        'test_avg_metrics': test_avg
    }
    with open(os.path.join(save_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(avg_metrics_pickle, f)

    # Print final test accuracy at last epoch
    if 'accuracy' in test_avg:
        print(f"Average Test Accuracy at final epoch = {test_avg['accuracy'][-1]:.4f}")
    else:
        print("Average Test Accuracy not computed because 'accuracy' key not found in test metrics.")

def save_fold_summary_csv(
    n_splits: int,
    train_metrics_per_fold: list,
    val_metrics_per_fold: list,
    test_metrics_per_fold: list,
    save_path: str
):
    """
    Creates and saves a CSV summarizing the final epoch metrics for each fold,
    plus the average row at the bottom.
    """
    columns = [
        'Fold', 
        'Train_Accuracy', 'Val_Accuracy', 'Test_Accuracy',
        'Train_F1',       'Val_F1',       'Test_F1',
        'Train_AUCPR',    'Val_AUCPR',    'Test_AUCPR',
        'Train_precision','Val_precision','Test_precision',
        'Train_recall',   'Val_recall',   'Test_recall'
    ]
    data_rows = []

    # Populate per-fold
    for i in range(n_splits):
        data_rows.append([
            i + 1,
            train_metrics_per_fold[i]['accuracy'][-1],
            val_metrics_per_fold[i]['accuracy'][-1],
            test_metrics_per_fold[i]['accuracy'][-1],

            train_metrics_per_fold[i]['f1_score'][-1],
            val_metrics_per_fold[i]['f1_score'][-1],
            test_metrics_per_fold[i]['f1_score'][-1],

            train_metrics_per_fold[i]['auc_pr'][-1],
            val_metrics_per_fold[i]['auc_pr'][-1],
            test_metrics_per_fold[i]['auc_pr'][-1],

            train_metrics_per_fold[i]['precision'][-1],
            val_metrics_per_fold[i]['precision'][-1],
            test_metrics_per_fold[i]['precision'][-1],

            train_metrics_per_fold[i]['recall'][-1],
            val_metrics_per_fold[i]['recall'][-1],
            test_metrics_per_fold[i]['recall'][-1],
        ])

    # Compute and append "Average" row
    def average_of_last_epoch(metric_list, key):
        # Returns average of last-epoch metric across folds
        return np.mean([m[key][-1] for m in metric_list])

    data_rows.append([
        'Average',
        average_of_last_epoch(train_metrics_per_fold, 'accuracy'),
        average_of_last_epoch(val_metrics_per_fold,   'accuracy'),
        average_of_last_epoch(test_metrics_per_fold,  'accuracy'),
        average_of_last_epoch(train_metrics_per_fold, 'f1_score'),
        average_of_last_epoch(val_metrics_per_fold,   'f1_score'),
        average_of_last_epoch(test_metrics_per_fold,  'f1_score'),
        average_of_last_epoch(train_metrics_per_fold, 'auc_pr'),
        average_of_last_epoch(val_metrics_per_fold,   'auc_pr'),
        average_of_last_epoch(test_metrics_per_fold,  'auc_pr'),
        average_of_last_epoch(train_metrics_per_fold, 'precision'),
        average_of_last_epoch(val_metrics_per_fold,   'precision'),
        average_of_last_epoch(test_metrics_per_fold,  'precision'),
        average_of_last_epoch(train_metrics_per_fold, 'recall'),
        average_of_last_epoch(val_metrics_per_fold,   'recall'),
        average_of_last_epoch(test_metrics_per_fold,  'recall'),
    ])

    metrics_df = pd.DataFrame(data_rows, columns=columns)
    metrics_df.to_csv(save_path, index=False)
    print(f"Metrics summary saved to {save_path}")

################
# Main Function #
################

def main():
    """
    Main function to run cross-validation training, validation, and testing
    for the MicroKPNN encoder model with confounder-free classification.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --------------------- #
    #    Hyperparameters    #
    # --------------------- #
    input_size = 371
    latent_dim = 64
    num_layers = 1
    learning_rate = 0.0001
    num_epochs = 100
    batch_size = 64
    n_splits = 5

    # --------------------- #
    #   Output directory    #
    # --------------------- #
    output_dir = 'Results/MicroKPNN_encoder_confounder_free_plots'
    os.makedirs(output_dir, exist_ok=True)

    # --------------------- #
    #      Data paths       #
    # --------------------- #
    train_abundance_path = 'MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv'
    train_metadata_path   = 'MetaCardis_data/train_T2D_metadata.csv'
    test_abundance_path   = 'MetaCardis_data/new_test_T2D_abundance_with_taxon_ids.csv'
    test_metadata_path    = 'MetaCardis_data/test_T2D_metadata.csv'
    edge_list_path        = "Default_Database/EdgeList.csv"

    # 1) Build mask
    relative_abundance = pd.read_csv(train_abundance_path, index_col=0)
    species = relative_abundance.columns.values.tolist()

    def build_mask(edge_list, species_list):
        """
        Build a binary mask from an edge list CSV for hierarchical connections.
        """
        edge_df = pd.read_csv(edge_list)
        edge_df['parent'] = edge_df['parent'].astype(str)
        parent_nodes = sorted(set(edge_df['parent'].tolist()))
        mask_tensor = torch.zeros(len(species_list), len(parent_nodes))

        parent_dict = {k: i for i, k in enumerate(parent_nodes)}
        child_dict  = {k: i for i, k in enumerate(species_list)}

        for _, row in edge_df.iterrows():
            if row['child'] != 'Unnamed: 0':
                mask_tensor[child_dict[str(row['child'])]][parent_dict[row['parent']]] = 1

        return mask_tensor.T, parent_dict

    mask, parent_dict = build_mask(edge_list_path, species)

    # Save parent dict to CSV
    parent_dict_path = os.path.join(output_dir, "parent_dict_main.csv")
    pd.DataFrame(list(parent_dict.items()), columns=['Parent', 'Index']) \
      .to_csv(parent_dict_path, index=False)

    # --------------------- #
    #   Load & merge data   #
    # --------------------- #
    merged_data_all     = get_data(train_abundance_path, train_metadata_path)
    merged_test_data_all= get_data(test_abundance_path, test_metadata_path)

    metadata_columns = pd.read_csv(train_metadata_path).columns.tolist()
    feature_columns = [
        c for c in merged_data_all.columns 
        if c not in metadata_columns and c != 'SampleID'
    ]

    X = merged_data_all[feature_columns].values
    # Combine labels for stratification
    merged_data_all['combined'] = (
        merged_data_all['PATGROUPFINAL_C'].astype(str) +
        merged_data_all['METFORMIN_C'].astype(str)
    )
    y_all = merged_data_all['combined'].values

    # Test data
    x_test_all = torch.tensor(
        merged_test_data_all[feature_columns].values,
        dtype=torch.float32
    )
    y_test_all = torch.tensor(
        merged_test_data_all['PATGROUPFINAL_C'].values,
        dtype=torch.float32
    ).unsqueeze(1)

    # Disease group only
    test_data_disease = merged_test_data_all[
        merged_test_data_all['PATGROUPFINAL_C'] == 1
    ]
    x_test_disease = torch.tensor(
        test_data_disease[feature_columns].values,
        dtype=torch.float32
    )
    y_test_disease = torch.tensor(
        test_data_disease['METFORMIN_C'].values,
        dtype=torch.float32
    ).unsqueeze(1)

    # --------------------- #
    #   Stratified K-Fold   #
    # --------------------- #
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store metrics for each fold
    train_metrics_per_fold = []
    val_metrics_per_fold   = []
    test_metrics_per_fold  = []

    # --------------------- #
    #        Training       #
    # --------------------- #
    for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y_all)):
        print(f"\n=== Fold {fold_idx + 1} / {n_splits} ===")

        # Split data
        train_data = merged_data_all.iloc[train_index]
        val_data   = merged_data_all.iloc[val_index]

        x_all_train = torch.tensor(
            train_data[feature_columns].values, dtype=torch.float32
        )
        y_all_train = torch.tensor(
            train_data['PATGROUPFINAL_C'].values, dtype=torch.float32
        ).unsqueeze(1)

        x_all_val = torch.tensor(
            val_data[feature_columns].values, dtype=torch.float32
        )
        y_all_val = torch.tensor(
            val_data['PATGROUPFINAL_C'].values, dtype=torch.float32
        ).unsqueeze(1)

        # Disease group only
        train_data_disease = train_data[train_data['PATGROUPFINAL_C'] == 1]
        val_data_disease   = val_data[val_data['PATGROUPFINAL_C'] == 1]

        x_disease_train = torch.tensor(
            train_data_disease[feature_columns].values, dtype=torch.float32
        )
        y_disease_train = torch.tensor(
            train_data_disease['METFORMIN_C'].values, dtype=torch.float32
        ).unsqueeze(1)

        x_disease_val = torch.tensor(
            val_data_disease[feature_columns].values, dtype=torch.float32
        )
        y_disease_val = torch.tensor(
            val_data_disease['METFORMIN_C'].values, dtype=torch.float32
        ).unsqueeze(1)

        # Create DataLoaders
        data_loader      = create_stratified_dataloader(
            x_disease_train, y_disease_train, batch_size
        )
        data_all_loader  = create_stratified_dataloader(
            x_all_train, y_all_train, batch_size
        )
        data_val_loader  = create_stratified_dataloader(
            x_disease_val, y_disease_val, batch_size
        )
        data_all_val_loader = create_stratified_dataloader(
            x_all_val, y_all_val, batch_size
        )
        data_test_loader = create_stratified_dataloader(
            x_test_disease, y_test_disease, batch_size
        )
        data_all_test_loader = create_stratified_dataloader(
            x_test_all, y_test_all, batch_size
        )

        # Compute positive class weights
        num_pos_disease = y_all_train.sum().item()
        num_neg_disease = len(y_all_train) - num_pos_disease
        pos_weight_value_disease = num_neg_disease / num_pos_disease
        pos_weight_disease = torch.tensor(
            [pos_weight_value_disease], dtype=torch.float32
        ).to(device)

        num_pos_drug = y_disease_train.sum().item()
        num_neg_drug = len(y_disease_train) - num_pos_drug
        pos_weight_value_drug = num_neg_drug / num_pos_drug
        pos_weight_drug = torch.tensor(
            [pos_weight_value_drug], dtype=torch.float32
        ).to(device)

        # Initialize model, losses, optimizers
        model = GAN(
            mask=mask, 
            num_layers=num_layers, 
            latent_dim=latent_dim
        ).to(device)

        criterion = PearsonCorrelationLoss().to(device)
        optimizer_g = optim.Adam(model.encoder.parameters(), lr=0.002)

        criterion_classifier = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_drug
        ).to(device)
        optimizer_classifier = optim.Adam(
            model.classifier.parameters(), lr=0.002
        )

        criterion_disease_classifier = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_disease
        ).to(device)
        optimizer_disease_classifier = optim.Adam(
            list(model.encoder.parameters()) + 
            list(model.disease_classifier.parameters()), 
            lr=learning_rate
        )

        # Train model
        results = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer_g,
            data_loader=data_loader,
            data_all_loader=data_all_loader,
            data_val_loader=data_val_loader,
            data_all_val_loader=data_all_val_loader,
            data_test_loader=data_test_loader,
            data_all_test_loader=data_all_test_loader,
            num_epochs=num_epochs,
            criterion_classifier=criterion_classifier,
            optimizer_classifier=optimizer_classifier,
            criterion_disease_classifier=criterion_disease_classifier,
            optimizer_disease_classifier=optimizer_disease_classifier,
            device=device
        )

        # Save the trained model checkpoint
        model_save_path = os.path.join(output_dir, f"trained_model{fold_idx+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved: {model_save_path}")

        # Save feature columns once (only necessary in fold 1, but safe to do each time)
        feature_csv_path = os.path.join(output_dir, "feature_columns.csv")
        pd.Series(feature_columns).to_csv(feature_csv_path, index=False)

        # Store metrics
        train_metrics_per_fold.append(results['train'])
        val_metrics_per_fold.append(results['val'])
        test_metrics_per_fold.append(results['test'])

        # 1) Plot confusion matrices (final epoch)
        plot_confusion_matrix(
            results['train']['confusion_matrix'][-1],
            title=f"Train Confusion Matrix - Fold {fold_idx+1}",
            save_path=os.path.join(output_dir, f"fold_{fold_idx+1}_train_conf_matrix.png"),
            class_names=["Class 0", "Class 1"]
        )
        plot_confusion_matrix(
            results['val']['confusion_matrix'][-1],
            title=f"Validation Confusion Matrix - Fold {fold_idx+1}",
            save_path=os.path.join(output_dir, f"fold_{fold_idx+1}_val_conf_matrix.png"),
            class_names=["Class 0", "Class 1"]
        )
        plot_confusion_matrix(
            results['test']['confusion_matrix'][-1],
            title=f"Test Confusion Matrix - Fold {fold_idx+1}",
            save_path=os.path.join(output_dir, f"fold_{fold_idx+1}_test_conf_matrix.png"),
            class_names=["Class 0", "Class 1"]
        )

        # 2) Plot metrics (train vs. test) across epochs
        plot_metrics_dict = {
            'train': results['train'],
            'val':   results['val'],
            'test':  results['test']
        }
        plot_metrics_by_fold(
            fold_idx=fold_idx+1,
            metrics_dict=plot_metrics_dict,
            save_dir=output_dir
        )

    # End of loop over folds

    # --------------------- #
    #   Average all folds   #
    # --------------------- #
    # Plot averaged metrics, confusion matrices, and save them
    average_and_plot_metrics(
        train_metrics_per_fold=train_metrics_per_fold,
        val_metrics_per_fold=val_metrics_per_fold,
        test_metrics_per_fold=test_metrics_per_fold,
        save_dir=output_dir
    )

    # Save fold summary CSV (final epoch metrics)
    metrics_summary_path = os.path.join(output_dir, "metrics_summary.csv")
    save_fold_summary_csv(
        n_splits=n_splits,
        train_metrics_per_fold=train_metrics_per_fold,
        val_metrics_per_fold=val_metrics_per_fold,
        test_metrics_per_fold=test_metrics_per_fold,
        save_path=metrics_summary_path
    )

    print("=== Training & evaluation completed! ===")

if __name__ == "__main__":
    main()
