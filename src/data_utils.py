"""
data_utils.py

Module for loading and preprocessing CSV data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_transform_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file, apply log transformation, standard scaling,
    and return a processed DataFrame with 'SampleID' as a column.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the features and 'SampleID'.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the first column as 'SampleID' and the rest are
        log-transformed, scaled features.
    """
    data = pd.read_csv(file_path)
    uid = data['SampleID']

    # Drop SampleID for transformation
    X = data.drop(columns=['SampleID']).values

    # Log transform
    X_log = np.log(X + 1)

    # Standard scaling
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(X_log)

    # Construct output DataFrame
    transformed_df = pd.DataFrame(
        features_normalized,
        columns=data.columns[1:]  # all columns except SampleID
    )
    transformed_df['SampleID'] = uid

    # Reorder columns: SampleID first
    cols = ['SampleID'] + list(transformed_df.columns[:-1])
    return transformed_df[cols]


def get_data(file_path: str, metadata_file_path: str) -> pd.DataFrame:
    """
    Load and merge metadata and relative abundance data for training.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing abundance features.
    metadata_file_path : str
        Path to the CSV file containing metadata.

    Returns
    -------
    pd.DataFrame
        A merged DataFrame of metadata and abundance features on 'SampleID'.
    """
    # Load transformed abundance
    relative_abundance = load_and_transform_data(file_path)

    # Load metadata
    metadata = pd.read_csv(metadata_file_path)

    # Merge
    merged_data = pd.merge(metadata, relative_abundance, on='SampleID')
    return merged_data



