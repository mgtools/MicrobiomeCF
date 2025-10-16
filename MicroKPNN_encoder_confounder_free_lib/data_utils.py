import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clr_transform(df, pseudocount=1e-6):
    """
    Apply centered log-ratio (CLR) transformation to compositional data.
    """
    df = df + pseudocount
    gm = np.exp(np.log(df).mean(axis=1))
    clr = np.log(df).subtract(np.log(gm), axis=0)
    return clr

def load_and_transform_data(train_file_path, test_file_path, pseudocount=1e-6):
    """
    Load train/test data, apply CLR transformation, and standard-scale using only train stats.
    Returns:
        - train_df_transformed: scaled CLR-transformed train data with SampleID
        - test_df_transformed: scaled CLR-transformed test data with SampleID
    """
    # Load CSVs
    train_raw = pd.read_csv(train_file_path)
    test_raw = pd.read_csv(test_file_path)

    # Save SampleIDs
    train_uid = train_raw['SampleID']
    test_uid = test_raw['SampleID']

    # Drop SampleID column for feature matrix
    train_features = train_raw.drop(columns=['SampleID'])
    test_features = test_raw.drop(columns=['SampleID'])

    # Normalize to proportions
    train_normalized = train_features.div(train_features.sum(axis=1), axis=0)
    test_normalized = test_features.div(test_features.sum(axis=1), axis=0)

    # CLR transform
    train_clr = clr_transform(train_normalized, pseudocount=pseudocount)
    test_clr = clr_transform(test_normalized, pseudocount=pseudocount)

    # Standard scale using stats from training data only
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_clr)
    test_scaled = scaler.transform(test_clr)

    # Reconstruct DataFrames with SampleID
    train_df_transformed = pd.DataFrame(train_scaled, columns=train_clr.columns)
    test_df_transformed = pd.DataFrame(test_scaled, columns=test_clr.columns)
    train_df_transformed.insert(0, 'SampleID', train_uid)
    test_df_transformed.insert(0, 'SampleID', test_uid)

    return train_df_transformed, test_df_transformed

def get_data(train_file_path, train_metadata_file_path, test_file_path, test_metadata_file_path):
    """
    Load CLR-transformed + scaled train/test abundance and merge with metadata.
    """
    train_abundance, test_abundance = load_and_transform_data(train_file_path, test_file_path)
    train_metadata = pd.read_csv(train_metadata_file_path)
    test_metadata = pd.read_csv(test_metadata_file_path)

    train_merged = pd.merge(train_metadata, train_abundance, on='SampleID')
    test_merged = pd.merge(test_metadata, test_abundance, on='SampleID')
    return train_merged, test_merged
