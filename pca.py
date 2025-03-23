import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple
from typing import Optional

# Reads data and groups the x,y,z coordinates of each landmark together
def read_group_data(path) -> pd.DataFrame:
    pure_data = pd.read_csv(path)
    # pure_data = pure_data.drop(pure_data.columns[0], axis=1)  # remove the first column - frame

    landmarks = sorted(set(col.rsplit('_', 1)[0] for col in pure_data.columns))

    # Create a new DataFrame with tuples
    df_tuples = pd.DataFrame({
        landmark: list(zip(pure_data[f"{landmark}_x"], pure_data[f"{landmark}_y"], pure_data[f"{landmark}_z"])) for
        landmark in landmarks
    })

    return df_tuples


def pca_importance_scores(data: pd.DataFrame, n_components: Optional[int] = None, save=False,
                          name="./data/pca_importance_list.csv") -> Optional[pd.DataFrame]:
    """
    Performs PCA and calculates importance scores for each (x, y, z) landmark.

    Parameters:
    - data (pd.DataFrame): Input data where columns represent coordinates (x_1, y_1, z_1, x_2, y_2, z_2, ...).
    - n_components (int, optional): Number of PCA components to retain.

    Returns:
    - pd.DataFrame: Importance scores per landmark.
    """
    num_landmarks = data.shape[1] // 3  # Total landmarks in data

    if n_components is None:
        n_components = min(data.shape[0], data.shape[1])  # Use all possible components if not specified

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(data)

    # Explained variance per component
    explained_variance = pca.explained_variance_ratio_  # Importance of each PC

    # Aggregate scores per landmark (each has 3 coordinates)
    landmark_scores = []
    for i in range(num_landmarks):
        score = sum(explained_variance[i * 3: (i + 1) * 3])  # Sum importance of x, y, z for each landmark
        landmark_scores.append(score)

    # Create DataFrame for scores
    landmark_names = [f'Landmark_{i + 1}' for i in range(num_landmarks)]
    scores_df = pd.DataFrame({'Landmark': landmark_names, 'Importance Score': landmark_scores})

    return scores_df.sort_values(by="Importance Score", ascending=False)  # Sort by importance

# Example Usage

data = read_group_data("./data/normalized_landmarks.csv")

# importance_df = pca_importance_scores(data=data, n_components=0.95)
# print(importance_df)

data.to_csv("./data/landmarks_grouped.csv", index=False)