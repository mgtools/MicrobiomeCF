import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay

from data_utils import get_data
from bench.MicroKPNN_models import GAN, PearsonCorrelationLoss
from utils import create_stratified_dataloader
from train import train_model

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, title, save_path, class_names=None):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def main():
    # Hyperparameters
    input_size = 371  # 654 features
    latent_dim = 32
    num_layers = 1
    learning_rate = 0.0001
    num_epochs = 100
    batch_size = 64

    os.makedirs('Results/MicroKPNN_plots', exist_ok=True)

    train_abundance_path = 'MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv'
    train_metadata_path = 'MetaCardis_data/train_T2D_metadata.csv'
    test_abundance_path = 'MetaCardis_data/new_test_T2D_abundance_with_taxon_ids.csv'
    test_metadata_path = 'MetaCardis_data/test_T2D_metadata.csv'

    edge_list = "Default_Database/EdgeList.csv"
    relative_abundance = pd.read_csv('MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv', index_col = 0)
    print(relative_abundance.shape)
    species = relative_abundance.columns.values.tolist()
    # Build masks for each pair of consecutive ranks
    def build_mask(edge_list, species):
        # generate the mask
        edge_df = pd.read_csv(edge_list)
        

        edge_df['parent'] = edge_df['parent'].astype(str)
        parent_nodes = list(set(edge_df['parent'].tolist()))
        mask = torch.zeros(len(species), len(parent_nodes))
        child_nodes = species 
		
        # parent_nodes.sort()
        parent_dict = {k: i for i, k in enumerate(parent_nodes)}
        # child_nodes.sort()
        child_dict = {k: i for i, k in enumerate(child_nodes)}
				
        for i, row in edge_df.iterrows():
            if row['child'] != 'Unnamed: 0': 
                mask[child_dict[str(row['child'])]][parent_dict[row['parent']]] = 1

        return mask.T
    

    # Build masks
    
    mask = build_mask(edge_list, species)
    print(mask.shape)
    print(mask)

    # Load merged data
    merged_data_all = get_data(train_abundance_path, train_metadata_path)
    merged_test_data_all = get_data(test_abundance_path, test_metadata_path)

    # Define feature columns
    metadata_columns = pd.read_csv(train_metadata_path).columns.tolist()
    feature_columns = [
        col for col in merged_data_all.columns if col not in metadata_columns and col != 'SampleID'
    ]

    X = merged_data_all[feature_columns].values
    y_all = merged_data_all['PATGROUPFINAL_C'].values  # Labels for disease classification

    # Prepare test data
    x_test_all = torch.tensor(merged_test_data_all[feature_columns].values, dtype=torch.float32)
    y_test_all = torch.tensor(merged_test_data_all['PATGROUPFINAL_C'].values, dtype=torch.float32).unsqueeze(1)

    # Initialize StratifiedKFold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # To store metrics across folds
    train_metrics_per_fold = []
    val_metrics_per_fold = []
    test_metrics_per_fold = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y_all)):
        print(f"Fold {fold+1}")

        # Split data into training and validation sets
        train_data = merged_data_all.iloc[train_index]
        val_data = merged_data_all.iloc[val_index]

        # Prepare training data
        x_all_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
        y_all_train = torch.tensor(train_data['PATGROUPFINAL_C'].values, dtype=torch.float32).unsqueeze(1)

        x_all_val = torch.tensor(val_data[feature_columns].values, dtype=torch.float32)
        y_all_val = torch.tensor(val_data['PATGROUPFINAL_C'].values, dtype=torch.float32).unsqueeze(1)

        # Create stratified DataLoaders
        data_all_loader = create_stratified_dataloader(x_all_train, y_all_train, batch_size)
        data_all_val_loader = create_stratified_dataloader(x_all_val, y_all_val, batch_size)
        data_all_test_loader = create_stratified_dataloader(x_test_all, y_test_all, batch_size)

        # Compute positive class weights
        num_pos_disease = y_all_train.sum().item()
        num_neg_disease = len(y_all_train) - num_pos_disease
        pos_weight_value_disease = num_neg_disease / num_pos_disease
        pos_weight_disease = torch.tensor([pos_weight_value_disease], dtype=torch.float32).to(device)

        # Define model, loss, and optimizer
        model = GAN(mask=mask, num_layers=1).to(device)
        criterion_disease_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight_disease).to(device)
        optimizer_disease_classifier = optim.Adam(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()), lr=learning_rate
        )

        # Train the model
        Results = train_model(
            model, data_all_loader, data_all_val_loader,
            data_all_test_loader, num_epochs,
            criterion_disease_classifier, optimizer_disease_classifier,
            device
        )

        # Store metrics for this fold
        train_metrics_per_fold.append(Results['train'])
        val_metrics_per_fold.append(Results['val'])
        test_metrics_per_fold.append(Results['test'])


        # Plot confusion matrices for the final epoch of this fold
        plot_confusion_matrix(Results['train']['confusion_matrix'][-1], 
                      title=f'Train Confusion Matrix - Fold {fold+1}', 
                      save_path=f'Results/MicroKPNN_plots/fold_{fold+1}_train_conf_matrix.png',
                      class_names=['Class 0', 'Class 1'])

        plot_confusion_matrix(Results['val']['confusion_matrix'][-1], 
                      title=f'Validation Confusion Matrix - Fold {fold+1}', 
                      save_path=f'Results/MicroKPNN_plots/fold_{fold+1}_val_conf_matrix.png',
                      class_names=['Class 0', 'Class 1'])

        plot_confusion_matrix(Results['test']['confusion_matrix'][-1], 
                      title=f'Test Confusion Matrix - Fold {fold+1}', 
                      save_path=f'Results/MicroKPNN_plots/fold_{fold+1}_test_conf_matrix.png',
                      class_names=['Class 0', 'Class 1'])
        
        num_epochs_actual = len(Results['train']['loss_history'])
        epochs = range(1, num_epochs_actual + 1)

        # Plot metrics for this fold
        plt.figure(figsize=(20, 15))



        plt.subplot(2, 3, 1)
        plt.plot(epochs, Results['train']['loss_history'], label=f'Fold train {fold+1}')
        plt.plot(epochs, Results['val']['loss_history'], label=f'Fold val {fold+1}')
        plt.plot(epochs, Results['test']['loss_history'], label=f'Fold test {fold+1}')
        plt.title("Disease Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Disease Loss")
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.plot(epochs, Results['train']['accuracy'], label=f'Fold {fold+1} Train')
        plt.plot(epochs, Results['val']['accuracy'], label=f'Fold {fold+1} Val')
        plt.plot(epochs, Results['test']['accuracy'], label=f'Fold {fold+1} Test')
        plt.title("Accuracy History")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(2, 3, 3)
        plt.plot(epochs, Results['train']['f1_score'], label=f'Fold {fold+1} Train')
        plt.plot(epochs, Results['val']['f1_score'], label=f'Fold {fold+1} Val')
        plt.plot(epochs, Results['test']['f1_score'], label=f'Fold {fold+1} Test')
        plt.title("F1 Score History")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()

        plt.subplot(2, 3, 4)
        plt.plot(epochs, Results['train']['auc_pr'], label=f'Fold {fold+1} Train')
        plt.plot(epochs, Results['val']['auc_pr'], label=f'Fold {fold+1} Val')
        plt.plot(epochs, Results['test']['auc_pr'], label=f'Fold {fold+1} Test')
        plt.title("AUCPR Score History")
        plt.xlabel("Epoch")
        plt.ylabel("AUCPR Score")
        plt.legend()

        plt.subplot(2, 3, 5)
        plt.plot(epochs, Results['train']['precision'], label=f'Fold {fold+1} Train')
        plt.plot(epochs, Results['val']['precision'], label=f'Fold {fold+1} Val')
        plt.plot(epochs, Results['test']['precision'], label=f'Fold {fold+1} Test')
        plt.title("Precisions Score History")
        plt.xlabel("Epoch")
        plt.ylabel("Precisions Score")
        plt.legend()

        plt.subplot(2, 3, 6)
        plt.plot(epochs, Results['train']['recall'], label=f'Fold {fold+1} Train')
        plt.plot(epochs, Results['val']['recall'], label=f'Fold {fold+1} Val')
        plt.plot(epochs, Results['test']['recall'], label=f'Fold {fold+1} Test')
        plt.title("Recalls Score History")
        plt.xlabel("Epoch")
        plt.ylabel("Recalls Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'Results/MicroKPNN_plots/fold_{fold+1}_metrics.png')
        plt.close()

    num_epochs_actual = len(train_metrics_per_fold[0]['loss_history'])
    epochs = range(1, num_epochs_actual + 1)

    train_avg_metrics = {key: np.zeros(num_epochs_actual) for key in train_metrics_per_fold[0].keys() if key != 'confusion_matrix'}
    val_avg_metrics = {key: np.zeros(num_epochs_actual) for key in val_metrics_per_fold[0].keys() if key != 'confusion_matrix'}
    test_avg_metrics = { key: np.zeros(num_epochs_actual) for key in test_metrics_per_fold[0].keys() if key != 'confusion_matrix'}

    # Initialize confusion matrix averages separately
    train_conf_matrix_avg = [np.zeros_like(train_metrics_per_fold[0]['confusion_matrix'][0]) for _ in range(num_epochs_actual)]
    val_conf_matrix_avg = [np.zeros_like(val_metrics_per_fold[0]['confusion_matrix'][0]) for _ in range(num_epochs_actual)]
    test_conf_matrix_avg = [np.zeros_like(test_metrics_per_fold[0]['confusion_matrix'][0]) for _ in range(num_epochs_actual)]

    # Accumulate scalar metrics for averaging
    for train_fold_metrics in train_metrics_per_fold:
        for key in train_avg_metrics.keys():
            train_avg_metrics[key] += np.array(train_fold_metrics[key])
        for epoch_idx, cm in enumerate(train_fold_metrics['confusion_matrix']):
            train_conf_matrix_avg[epoch_idx] += cm

    for val_fold_metrics in val_metrics_per_fold:
        for key in val_avg_metrics.keys():
            val_avg_metrics[key] += np.array(val_fold_metrics[key])
        for epoch_idx, cm in enumerate(val_fold_metrics['confusion_matrix']):
            val_conf_matrix_avg[epoch_idx] += cm

    for test_fold_metrics in test_metrics_per_fold:
        for key in test_avg_metrics.keys():
            test_avg_metrics[key] += np.array(test_fold_metrics[key])
        for epoch_idx, cm in enumerate(test_fold_metrics['confusion_matrix']):
            test_conf_matrix_avg[epoch_idx] += cm

    # Compute averages across folds for scalar metrics
    num_train_folds = len(train_metrics_per_fold)
    num_val_folds = len(val_metrics_per_fold)
    num_test_folds = len(test_metrics_per_fold)

    for key in train_avg_metrics.keys():
        train_avg_metrics[key] /= num_train_folds

    for key in val_avg_metrics.keys():
        val_avg_metrics[key] /= num_val_folds

    for key in test_avg_metrics.keys():
        test_avg_metrics[key] /= num_test_folds

    # Plot average metrics across folds
    plt.figure(figsize=(20, 15))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_avg_metrics['loss_history'], label='Average train')
    plt.plot(epochs, val_avg_metrics['loss_history'], label='Average val')
    plt.plot(epochs, test_avg_metrics['loss_history'], label='Average test')
    plt.title("Average Disease Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Disease Loss")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, train_avg_metrics['accuracy'], label='Train Average')
    plt.plot(epochs, val_avg_metrics['accuracy'], label='Validation Average')
    plt.plot(epochs, test_avg_metrics['accuracy'], label='Test Average')
    plt.title("Average Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs, train_avg_metrics['f1_score'], label='Train Average')
    plt.plot(epochs, val_avg_metrics['f1_score'], label='Validation Average')
    plt.plot(epochs, test_avg_metrics['f1_score'], label='Test Average')
    plt.title("Average F1 Score History")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, train_avg_metrics['auc_pr'], label='Train Average')
    plt.plot(epochs, val_avg_metrics['auc_pr'], label='Validation Average')
    plt.plot(epochs, test_avg_metrics['auc_pr'], label='Test Average')
    plt.title("Average AUCPR Score History")
    plt.xlabel("Epoch")
    plt.ylabel("AUCPR Score")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs, train_avg_metrics['precision'], label=f'Fold {fold+1} Train')
    plt.plot(epochs, val_avg_metrics['precision'], label=f'Fold {fold+1} Val')
    plt.plot(epochs, test_avg_metrics['precision'], label=f'Fold {fold+1} Test')
    plt.title("Average Precisions Score History")
    plt.xlabel("Epoch")
    plt.ylabel("Precision Score")
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(epochs, train_avg_metrics['recall'], label=f'Fold {fold+1} Train')
    plt.plot(epochs, val_avg_metrics['recall'], label=f'Fold {fold+1} Val')
    plt.plot(epochs, test_avg_metrics['recall'], label=f'Fold {fold+1} Test')
    plt.title("Average Recalls Score History")
    plt.xlabel("Epoch")
    plt.ylabel("Recalls Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig('Results/MicroKPNN_plots/average_metrics.png')
    plt.close()

    # Print average of the final epoch's test accuracy across folds

    # Compute averages for confusion matrices
    train_conf_matrix_avg = [cm / num_train_folds for cm in train_conf_matrix_avg]
    val_conf_matrix_avg = [cm / num_val_folds for cm in val_conf_matrix_avg]
    test_conf_matrix_avg = [cm / num_test_folds for cm in test_conf_matrix_avg]

    # Add the averaged confusion matrices back to the metrics dictionaries
    train_avg_metrics['confusion_matrix'] = train_conf_matrix_avg
    val_avg_metrics['confusion_matrix'] = val_conf_matrix_avg
    test_avg_metrics['confusion_matrix'] = test_conf_matrix_avg

    # Plot average confusion matrices
    plot_confusion_matrix(train_avg_metrics['confusion_matrix'][-1], 
                        title='Average Train Confusion Matrix', 
                        save_path='Results/MicroKPNN_plots/average_train_conf_matrix.png',
                        class_names=['Class 0', 'Class 1'])

    plot_confusion_matrix(val_avg_metrics['confusion_matrix'][-1], 
                        title='Average Validation Confusion Matrix', 
                        save_path='Results/MicroKPNN_plots/average_val_conf_matrix.png',
                        class_names=['Class 0', 'Class 1'])

    plot_confusion_matrix(test_avg_metrics['confusion_matrix'][-1], 
                        title='Average Test Confusion Matrix', 
                        save_path='Results/MicroKPNN_plots/average_test_conf_matrix.png',
                        class_names=['Class 0', 'Class 1'])
    
    avg_test_accs = test_avg_metrics['accuracy'][-1]
    print(f"Average Test Accuracy over {n_splits} folds: {avg_test_accs:.4f}")

     # Create DataFrame to store metrics for each fold
    metrics_columns = ['Fold', 'Train_Accuracy', 'Val_Accuracy', 'Test_Accuracy',
                    'Train_F1', 'Val_F1', 'Test_F1', 
                    'Train_AUCPR', 'Val_AUCPR', 'Test_AUCPR',
                    'Train_precision', 'Val_precision', 'Test_precision',
                    'Train_recall', 'Val_recall', 'Test_recall']

    metrics_data = []

    for i in range(n_splits):
        metrics_data.append([
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

    # Add average metrics to the table
    metrics_data.append([
        'Average',
        train_avg_metrics['accuracy'][-1],
        val_avg_metrics['accuracy'][-1],
        test_avg_metrics['accuracy'][-1],
        train_avg_metrics['f1_score'][-1],
        val_avg_metrics['f1_score'][-1],
        test_avg_metrics['f1_score'][-1],
        train_avg_metrics['auc_pr'][-1],
        val_avg_metrics['auc_pr'][-1],
        test_avg_metrics['auc_pr'][-1],
        train_avg_metrics['precision'][-1],
        val_avg_metrics['precision'][-1],
        test_avg_metrics['precision'][-1],
        train_avg_metrics['recall'][-1],
        val_avg_metrics['recall'][-1],
        test_avg_metrics['recall'][-1],
    ])

    # Create a DataFrame
    metrics_df = pd.DataFrame(metrics_data, columns=metrics_columns)

    # Save to CSV
    metrics_df.to_csv('Results/MicroKPNN_plots/metrics_summary.csv', index=False)

    print("Metrics for each fold and their average have been saved to 'Results/MicroKPNN_plots/metrics_summary.csv'.")


if __name__ == "__main__":
    main()