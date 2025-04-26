import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple
from typing import Optional
import argparse
from sklearn.preprocessing import StandardScaler

# Reads data and groups the x,y,z coordinates of each landmark together
def read_group_data(path) -> pd.DataFrame:
    pure_data = pd.read_csv(path)
    pure_data = pure_data.drop(pure_data.columns[0], axis=1)  # remove the first column - frame

    # Create a new DataFrame with lists
    # landmarks = sorted(set(col.rsplit('_', 1)[0] for col in pure_data.columns))
    # df_tuples = pd.DataFrame({
    #     landmark: list(list(elem) for elem in  zip(pure_data[f"{landmark}_x"], pure_data[f"{landmark}_y"], pure_data[f"{landmark}_z"])) for
    #     landmark in landmarks
    # })

    return pure_data

def pca_importance_scores(data: pd.DataFrame, normalised: True) -> Optional[pd.DataFrame]:
    num_landmarks = int(data.shape[1] // 3)  # Total landmarks in data

    # Apply PCA
    if not normalised:
        X = np.array(data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        data = pd.DataFrame(X_scaled, columns=data.columns)

    pca = PCA()
    pca.fit_transform(data)

    components = pca.components_
    components_df = pd.DataFrame(components, columns=data.columns)
    components_df_abs = components_df.abs()
    feature_scores = components_df_abs.sum(axis=0)
    sorted_features = feature_scores.sort_values(ascending=False)

    # Aggregate scores per landmark (each has 3 coordinates)
    landmark_scores = []
    for i in range(feature_scores.shape[0] // 3):
        score = sum(feature_scores[i * 3: (i + 1) * 3])  # Sum importance of x, y, z for each landmark
        landmark_scores.append(score)

    # Create DataFrame for scores
    landmark_names = [f'Landmark_{i}' for i in range(num_landmarks)]
    scores_df = pd.DataFrame({'Landmark': landmark_names, 'Importance Score': landmark_scores}).sort_values(by='Importance Score', ascending=False)

    return scores_df, sorted_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA Importance Scores")
    parser.add_argument("--data", type=str, default="./data/normalized_landmarks.csv", help="Path to the data file.")
    parser.add_argument("--save", type=str, default="", help="Path to save the importance scores.")
    parser.add_argument("--normalised", default=True, action="store_true", help="Use normalised data.")

    args = parser.parse_args()

    # Example usage
    data = read_group_data(args.data)
    print("Data loaded successfully.")

    importance_df, feature_impact = pca_importance_scores(data=data, normalised=args.normalised)
    print("PCA importance scores calculated successfully.")

    if args.save != "":
        importance_df.to_csv(args.save + "_aggragted_values.csv", index=False)
        feature_impact.to_csv(args.save + "_values.csv", index=True, header="Score")   
        print(f"Importance scores saved to {args.save}")