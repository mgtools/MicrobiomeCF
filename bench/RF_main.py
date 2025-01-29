import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_recall_curve, auc,
    precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier

from data_utils import get_data

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, title, save_path, class_names=None):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def main():
    # Ensure plots directory exists
    os.makedirs('Results/RF_plots', exist_ok=True)
    
    # Parameters
    n_splits = 5

    train_abundance_path = 'MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv'
    train_metadata_path = 'MetaCardis_data/train_T2D_metadata.csv'
    test_abundance_path = 'MetaCardis_data/new_test_T2D_abundance_with_taxon_ids.csv'
    test_metadata_path = 'MetaCardis_data/test_T2D_metadata.csv'

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

    X_test = merged_test_data_all[feature_columns].values
    y_test = merged_test_data_all['PATGROUPFINAL_C'].values

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # To store metrics across folds
    train_metrics_per_fold = []
    val_metrics_per_fold = []
    test_metrics_per_fold = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y_all)):
        print(f"Fold {fold+1}")

        # Split data into training and validation sets
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y_all[train_index], y_all[val_index]

        # Define and train the Random Forest classifier
        clf = RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_train, y_train)

        # Evaluate on train set
        y_train_prob = clf.predict_proba(X_train)[:, 1]
        y_train_pred = clf.predict(X_train)
        train_acc = balanced_accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        precision_vals, recall_vals, _ = precision_recall_curve(y_train, y_train_prob)
        train_auc_pr = auc(recall_vals, precision_vals)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_conf_matrix = confusion_matrix(y_train, y_train_pred)

        train_metrics = {
            'accuracy': [train_acc],
            'f1_score': [train_f1],
            'auc_pr': [train_auc_pr],
            'precision': [train_precision],
            'recall': [train_recall],
            'confusion_matrix': [train_conf_matrix]
        }

        # Evaluate on validation set
        y_val_prob = clf.predict_proba(X_val)[:, 1]
        y_val_pred = clf.predict(X_val)
        val_acc = balanced_accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_val_prob)
        val_auc_pr = auc(recall_vals, precision_vals)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_conf_matrix = confusion_matrix(y_val, y_val_pred)

        val_metrics = {
            'accuracy': [val_acc],
            'f1_score': [val_f1],
            'auc_pr': [val_auc_pr],
            'precision': [val_precision],
            'recall': [val_recall],
            'confusion_matrix': [val_conf_matrix]
        }

        # Evaluate on test set
        y_test_prob = clf.predict_proba(X_test)[:, 1]
        y_test_pred = clf.predict(X_test)
        test_acc = balanced_accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_prob)
        test_auc_pr = auc(recall_vals, precision_vals)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_conf_matrix = confusion_matrix(y_test, y_test_pred)

        test_metrics = {
            'accuracy': [test_acc],
            'f1_score': [test_f1],
            'auc_pr': [test_auc_pr],
            'precision': [test_precision],
            'recall': [test_recall],
            'confusion_matrix': [test_conf_matrix]
        }

        train_metrics_per_fold.append(train_metrics)
        val_metrics_per_fold.append(val_metrics)
        test_metrics_per_fold.append(test_metrics)

        # Plot confusion matrices for this fold
        plot_confusion_matrix(train_conf_matrix, 
                              title=f'Train Confusion Matrix - Fold {fold+1}', 
                              save_path=f'Results/RF_plots/fold_{fold+1}_train_conf_matrix.png',
                              class_names=['Class 0', 'Class 1'])
        plot_confusion_matrix(val_conf_matrix, 
                              title=f'Validation Confusion Matrix - Fold {fold+1}', 
                              save_path=f'Results/RF_plots/fold_{fold+1}_val_conf_matrix.png',
                              class_names=['Class 0', 'Class 1'])
        plot_confusion_matrix(test_conf_matrix, 
                              title=f'Test Confusion Matrix - Fold {fold+1}', 
                              save_path=f'Results/RF_plots/fold_{fold+1}_test_conf_matrix.png',
                              class_names=['Class 0', 'Class 1'])

    # Compute average metrics across folds
    def average_metric(metrics_list, key):
        return np.mean([m[key][0] for m in metrics_list])

    def average_confusion_matrix(metrics_list):
        sum_cm = np.zeros_like(metrics_list[0]['confusion_matrix'][0], dtype=float)
        for m in metrics_list:
            sum_cm += m['confusion_matrix'][0]
        avg_cm = sum_cm / len(metrics_list)
        return avg_cm

    avg_train_acc = average_metric(train_metrics_per_fold, 'accuracy')
    avg_train_f1 = average_metric(train_metrics_per_fold, 'f1_score')
    avg_train_auc_pr = average_metric(train_metrics_per_fold, 'auc_pr')
    avg_train_precision = average_metric(train_metrics_per_fold, 'precision')
    avg_train_recall = average_metric(train_metrics_per_fold, 'recall')
    avg_train_cm = average_confusion_matrix(train_metrics_per_fold)

    avg_val_acc = average_metric(val_metrics_per_fold, 'accuracy')
    avg_val_f1 = average_metric(val_metrics_per_fold, 'f1_score')
    avg_val_auc_pr = average_metric(val_metrics_per_fold, 'auc_pr')
    avg_val_precision = average_metric(val_metrics_per_fold, 'precision')
    avg_val_recall = average_metric(val_metrics_per_fold, 'recall')
    avg_val_cm = average_confusion_matrix(val_metrics_per_fold)

    avg_test_acc = average_metric(test_metrics_per_fold, 'accuracy')
    avg_test_f1 = average_metric(test_metrics_per_fold, 'f1_score')
    avg_test_auc_pr = average_metric(test_metrics_per_fold, 'auc_pr')
    avg_test_precision = average_metric(test_metrics_per_fold, 'precision')
    avg_test_recall = average_metric(test_metrics_per_fold, 'recall')
    avg_test_cm = average_confusion_matrix(test_metrics_per_fold)

    print(f"Average Test Accuracy over {n_splits} folds: {avg_test_acc:.4f}")

    # Plot average confusion matrices
    plot_confusion_matrix(avg_train_cm, 
                          title='Average Train Confusion Matrix', 
                          save_path='Results/RF_plots/average_train_conf_matrix.png',
                          class_names=['Class 0', 'Class 1'])

    plot_confusion_matrix(avg_val_cm, 
                          title='Average Validation Confusion Matrix', 
                          save_path='Results/RF_plots/average_val_conf_matrix.png',
                          class_names=['Class 0', 'Class 1'])

    plot_confusion_matrix(avg_test_cm, 
                          title='Average Test Confusion Matrix', 
                          save_path='Results/RF_plots/average_test_conf_matrix.png',
                          class_names=['Class 0', 'Class 1'])

    # Save all metrics to a CSV file
    fold_data = []
    for i in range(n_splits):
        fold_data.append({
            'fold': i+1,
            'train_accuracy': train_metrics_per_fold[i]['accuracy'][0],
            'train_f1_score': train_metrics_per_fold[i]['f1_score'][0],
            'train_auc_pr': train_metrics_per_fold[i]['auc_pr'][0],
            'train_precision': train_metrics_per_fold[i]['precision'][0],
            'train_recall': train_metrics_per_fold[i]['recall'][0],

            'val_accuracy': val_metrics_per_fold[i]['accuracy'][0],
            'val_f1_score': val_metrics_per_fold[i]['f1_score'][0],
            'val_auc_pr': val_metrics_per_fold[i]['auc_pr'][0],
            'val_precision': val_metrics_per_fold[i]['precision'][0],
            'val_recall': val_metrics_per_fold[i]['recall'][0],

            'test_accuracy': test_metrics_per_fold[i]['accuracy'][0],
            'test_f1_score': test_metrics_per_fold[i]['f1_score'][0],
            'test_auc_pr': test_metrics_per_fold[i]['auc_pr'][0],
            'test_precision': test_metrics_per_fold[i]['precision'][0],
            'test_recall': test_metrics_per_fold[i]['recall'][0]
        })

    # Add averaged row
    fold_data.append({
        'fold': 'average',
        'train_accuracy': avg_train_acc,
        'train_f1_score': avg_train_f1,
        'train_auc_pr': avg_train_auc_pr,
        'train_precision': avg_train_precision,
        'train_recall': avg_train_recall,

        'val_accuracy': avg_val_acc,
        'val_f1_score': avg_val_f1,
        'val_auc_pr': avg_val_auc_pr,
        'val_precision': avg_val_precision,
        'val_recall': avg_val_recall,

        'test_accuracy': avg_test_acc,
        'test_f1_score': avg_test_f1,
        'test_auc_pr': avg_test_auc_pr,
        'test_precision': avg_test_precision,
        'test_recall': avg_test_recall
    })

    df_metrics = pd.DataFrame(fold_data)
    df_metrics.to_csv('Results/RF_plots/fold_metrics.csv', index=False)

if __name__ == "__main__":
    main()
