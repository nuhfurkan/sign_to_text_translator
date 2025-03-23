import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from typing import Optional
from sklearn.impute import KNNImputer

path = "./landmarks_output.csv"

# Read data
def read_data(path):
    data = pd.read_csv(path)
    
    # remove the first column - frame
    data = data.drop(data.columns[0], axis=1)
    
    return data

# Normalize
def normalize(data):
    # Identify unique coordinate types (x, y, z)
    coordinate_types = ['x', 'y', 'z']

    # Normalize each coordinate type independently
    for coord in coordinate_types:
        # Find all columns that belong to this coordinate type
        cols = [col for col in data.columns if col.endswith(f'_{coord}')]

        # Apply MinMaxScaler to those columns
        # minmax scales all the values between 0-1
        data[cols] = pd.DataFrame(
            preprocessing.MinMaxScaler().fit_transform(data[cols]),
        )        

    return data

# Rotate
def rotate(data):
    
    pass

# Impute the NaN values / empty data
def impute_missing_entries(df, k_neighbors=3, save=False, name="./data/completed_data.csv") -> Optional[pd.DataFrame]:
    """
    Automatically detects and imputes missing values in all columns based on gap length.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing missing values.
    k_neighbors (int): Number of neighbors for KNN Imputation (for long gaps).
    
    Returns:
    pd.DataFrame: DataFrame with missing values imputed.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue  # Skip columns without missing values

        # Identify missing segments
        mask = df[col].isna()
        df['gap_group'] = mask.ne(mask.shift()).cumsum() * mask
        gap_lengths = df.groupby('gap_group').size()

        # Apply different methods based on gap length
        for gap_id, length in gap_lengths.items():
            if gap_id == 0:  # Skip if no missing data
                continue

            if length <= 2:
                # Short gaps: Linear interpolation
                df[col] = df[col].interpolate(method='linear')

            elif 3 <= length <= 5:
                # Medium gaps: Spline interpolation
                df[col] = df[col].interpolate(method='spline', order=3)

        # Long gaps: Use KNN Imputation
        if gap_lengths.max() > 5:
            knn_imputer = KNNImputer(n_neighbors=k_neighbors)
            df.iloc[:, :-1] = knn_imputer.fit_transform(df.iloc[:, :-1])  # Exclude 'gap_group'

    # Drop temporary column
    df.drop(columns=['gap_group'], inplace=True, errors='ignore')

    if save:
        df.to_csv(name, index=False)

    return df

# Smooth data  
def smooth(data) -> Optional[pd.DataFrame]:

    pass

# PCA
def pca(data, n_components=0.95, save=False, name="./data/landmarks_output_pca.csv") -> Optional[pd.DataFrame]:
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)

    # Retain as many original column names as possible
    original_columns = list(data.columns)
    num_components = data_pca.shape[1]

    if num_components <= len(original_columns):
        column_names = original_columns[:num_components]  # Keep original names
    else:
        column_names = original_columns + [f'PC{i+1}' for i in range(len(original_columns), num_components)]

    # Create DataFrame with original index
    pca_df = pd.DataFrame(data_pca, columns=column_names, index=data.index)

    # Save transformed data if requested
    if save:
        pca_df.to_csv(name, index=False)

    return pca_df

data = read_data("completed_data.csv")
# data = impute_missing_entries(data, save=True, name="completed_data.csv")

print(data.head())

pca_data = pca(data, save=True)
print(pca_data.head())
# print(normalize(read_data(path)))