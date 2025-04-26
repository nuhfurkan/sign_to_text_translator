import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from typing import Optional
from sklearn.impute import KNNImputer
import argparse

path = "./completed_data.csv"

# Read data
def read_data(path):
    data = pd.read_csv(path)

    # remove the first column - frame
    # data = data.drop(data.columns[0], axis=1)

    return data

def read_landmarks(data) -> pd.DataFrame:
    pure_data = read_data(data)

    landmarks = sorted(set(col.rsplit('_', 1)[0] for col in pure_data.columns))

    df_tuples = pd.DataFrame({
        landmark: list(zip(pure_data[f"{landmark}_x"], pure_data[f"{landmark}_y"], pure_data[f"{landmark}_z"])) for landmark in landmarks
    })

    return df_tuples

# Normalize
def normalize(data: pd.DataFrame, save=False, name="./data/normalized_landmarks.csv") -> Optional[pd.DataFrame]:
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

    if save:
        data.to_csv(name, index=False)

    return data

# Rotate
def rotate(data: pd.DataFrame, frame_idx=None, save=False, name="./data/rotated_landmarks.csv") -> Optional[pd.DataFrame]:
    """
    Rotates landmarks to align with a canonical coordinate system based on body pose.

    Parameters:
    data: DataFrame containing landmark data
    frame_idx: If provided, process only this frame; otherwise process all frames

    Returns:
    DataFrame or np.array of rotated landmarks
    """

    or_data = data.copy()
    data = data.drop(columns=["frame"])
    
    if frame_idx is not None:
        # Process a single frame
        frame = data.iloc[frame_idx]
        return _rotate_single_frame(frame)
    
    else:
        # Process all frames sequentially
        rotated_data = []
        for i in range(len(data)):
            rotated_data.append(_rotate_single_frame(data.iloc[i]).values.flatten().T)

        res = pd.DataFrame(rotated_data, columns=data.columns)
        res.insert(0, "frame", or_data["frame"].values)

        if save:
            res.to_csv(name, index=False)

        return res


def _rotate_single_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # Extract key points (shoulder and hip landmarks)
    p11 = np.array([frame['pose_11_x'], frame['pose_11_y'], frame['pose_11_z']])  # Left shoulder
    p12 = np.array([frame['pose_12_x'], frame['pose_12_y'], frame['pose_12_z']])  # Right shoulder
    p23 = np.array([frame['pose_23_x'], frame['pose_23_y'], frame['pose_23_z']])  # Left hip
    p24 = np.array([frame['pose_24_x'], frame['pose_24_y'], frame['pose_24_z']])  # Right hip

    # Calculate center of torso
    C = (p11 + p12 + p23 + p24) / 4

    # Define coordinate axes
    x_axis = p12 - p11  # Right to left shoulder
    x_axis /= np.linalg.norm(x_axis)

    mid_shoulder = (p11 + p12) / 2
    y_axis = mid_shoulder - C
    y_axis /= np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Rotation matrix
    R = np.vstack([x_axis, y_axis, z_axis]).T

    # Extract all landmarks from the frame
    landmarks = []
    for col in frame.index:
        if col.endswith('_x'):
            base = col[:-2]
            point = np.array([
                frame[f"{base}_x"],
                frame[f"{base}_y"],
                frame[f"{base}_z"]
            ])
            landmarks.append(point)

    # Rotate landmarks
    rotated_landmarks = []
    for point in landmarks:
        p = point - C  # Translate to torso center
        p_rotated = np.dot(R.T, p)  # Rotate
        rotated_landmarks.append(p_rotated)

    return pd.DataFrame(rotated_landmarks)

# Impute the NaN values / empty data
# TODO: imputing missing entries should be applied only for a limited number of consequtive NaN values
"""
 Explanation: Some gestures requires only single hand movements
 so the other hand might be visible and recorded as NaN,
 for such cases we should not impute the missing data. 
 This could be achieved by putting a limit for consequtive Nan value imputing.
"""
def impute_missing_entries(df: pd.DataFrame, k_neighbors=3, save=False, name="./data/completed_data.csv") -> Optional[pd.DataFrame]:
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
                pass  # Placeholder to indicate no further action here

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


# Smooth
# TODO: Imrove the smoothing function performance-wise
def smooth(data: pd.DataFrame, alpha=0.4, columns=None) -> pd.DataFrame:
    """
    Apply exponential moving average smoothing to a DataFrame of landmark data.

    Parameters:
    data: DataFrame containing landmark data
    alpha: Smoothing factor (0-1), higher values give more weight to new data
    columns: Optional list of columns to smooth (if None, all columns are smoothed)

    Returns:
    Smoothed DataFrame
    """
    if columns is None:
        columns = data.columns
        columns = columns.drop('frame')  # Exclude non-numeric columns

    smoothed_data = data.copy()
    prev_frame = data.iloc[0].copy()

    for i in range(1, len(data)):
        current_frame = data.iloc[i].copy()

        # Apply smoothing only to columns with valid data
        for col in columns:
            if not pd.isna(current_frame[col]):
                smoothed_data.loc[data.index[i], col] = prev_frame[col] * (1 - alpha) + current_frame[col] * alpha

        # Update previous frame for next iteration
        prev_frame = smoothed_data.iloc[i].copy()

    return smoothed_data

# Find outlier detection method
def detect_outliers(data: pd.DataFrame, threshold=3) -> pd.DataFrame:
    """
    Detect outliers in a DataFrame of landmark data.

    Parameters:
    data: DataFrame containing landmark data
    threshold: Z-score threshold for outlier detection

    Returns:
    DataFrame with outliers replaced by NaN
    """
    # Calculate Z-scores
    z_scores = np.abs((data - data.mean()) / data.std())

    # Replace outliers with NaN
    data[z_scores > threshold] = np.nan

    return data

def translate_hands(data: pd.DataFrame, save=False, name="./data/hands_translated.csv") -> Optional[pd.DataFrame]:
    # Calculate distance between right_hand_0 and pose_16
    x_distance = data['right_hand_0_x'] - data['pose_16_x']
    y_distance = data['right_hand_0_y'] - data['pose_16_y']
    z_distance = data['right_hand_0_z'] - data['pose_16_z']

    # Translate right hand to pose_16
    for i in range(21):
        data[f'right_hand_{i}_x'] -= x_distance
        data[f'right_hand_{i}_y'] -= y_distance
        data[f'right_hand_{i}_z'] -= z_distance

    # Calculate distance between left_hand_0 and pose_15
    x_distance = data['left_hand_0_x'] - data['pose_15_x']
    y_distance = data['left_hand_0_y'] - data['pose_15_y']
    z_distance = data['left_hand_0_z'] - data['pose_15_z']

    # Translate left hand to pose_15
    for i in range(21):
        data[f'left_hand_{i}_x'] -= x_distance
        data[f'left_hand_{i}_y'] -= y_distance
        data[f'left_hand_{i}_z'] -= z_distance

    if save:
        data.to_csv(name, index=False)

    return data

# PCA
def pca(data: pd.DataFrame, n_components=0.9999, save=False, name="./data/landmarks_output_pca.csv") -> Optional[pd.DataFrame]:
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)

    # Retain as many original column names as possible
    original_columns = list(data.columns)
    num_components = data_pca.shape[1]

    if num_components <= len(original_columns):
        column_names = original_columns[:num_components]  # Keep original names
    else:
        column_names = original_columns + [f'PC{i + 1}' for i in range(len(original_columns), num_components)]

    # Create DataFrame with original index
    pca_df = pd.DataFrame(data_pca, columns=column_names, index=data.index)

    # Save transformed data if requested
    if save:
        pca_df.to_csv(name, index=False)

    return pca_df

# Skip initial blank rows - Some videos have initial blank frames where no body parts is detected
def skip_initial_blank(data: pd.DataFrame) -> pd.DataFrame:
    while data.iloc[0].fillna(0).eq(0).all() == True:
        data = data.drop(0, axis=0)
        data = data.reset_index(drop=True)

    return data

def skip_blank(data: pd.DataFrame, save: bool = False, name: str = "./data/trimmed_data.csv") -> pd.DataFrame:
    # Remove rows where all values are NaN
    data = data.dropna(thresh=2)
    # print(data.shape)
    data = data.reset_index(drop=True)
    if save:
        data.to_csv(name, index=False)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess landmark data")
    parser.add_argument("--data", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--save_imputed", type=bool, default=True, help="Save imputed data to a file.")
    parser.add_argument("--save_translated_hands", type=bool, default=True, help="Save translated hands to a file.")
    parser.add_argument("--save_rotated", type=bool, default=True, help="Save rotated data to a file.")
    parser.add_argument("--save_normalized", type=bool, default=True, help="Save normalized data to a file.")
    parser.add_argument("--save_trimmed", type=bool, default=True, help="Save trimmed data to a file.")
    parser.add_argument("--out_folder", type=str, default="./data/", help="Output folder for processed data.")

    parser.add_argument("--hands_data", default=False, action="store_true", help="Process only hands data.")
    parser.add_argument("--pose_data", default=False, action="store_true", help="Process only pose data.")

    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    data = read_data(args.data)
    print("Data was read")
    
    data = skip_blank(data, save=args.save_trimmed, name=args.out_folder + "trimmed_data.csv")
    print("Removed blank frames")
    
    data = impute_missing_entries(data)
    print("Imputed missing entries")
    
    data = detect_outliers(data)
    print("Detected outliers")
    
    data = impute_missing_entries(data, save=args.save_imputed, name=args.out_folder + "completed_data.csv")
    print("Imputed missing entries again")
    
    if not args.hands_data and not args.pose_data:    
        data = translate_hands(data, save=args.save_translated_hands, name=args.out_folder + "translated_hands.csv")
        print("Translated hands")
    
    # TODO: data smooting should be imporved
    data = smooth(data)
    print("Smoothing done")
    
    if not args.hands_data and not args.pose_data:
        data = rotate(data, save=args.save_rotated, name=args.out_folder + "rotated_landmarks.csv")
        print("Rotated data")
    
    data = normalize(data, save=args.save_normalized, name=args.out_folder + "normalized_landmarks.csv")
    print("Normalized data")
# print(data.head())

# print(normalize(read_data(path)))