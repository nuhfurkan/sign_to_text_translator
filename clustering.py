import pandas as pd
import glob
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt
import argparse

def process_clustering(feature_csv_path: str, output_path: str):
    # ---------------------------
    # Step 1: Load CSV Files
    # ---------------------------

    csv_files = glob.glob(feature_csv_path)

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the specified directory.")

    # List to store valid (non-empty) feature DataFrames
    valid_features = []

    for file in csv_files:
        if os.path.getsize(file) == 0:
            # Skip truly empty files (no bytes at all)
            continue
        try:
            df = pd.read_csv(file)
            if not df.empty:
                valid_features.append(df)
        except pd.errors.EmptyDataError:
            # Skip files that can't be parsed due to missing headers
            continue


    # If no valid files were found, raise an error
    if not valid_features:
        raise ValueError("No valid (non-empty) CSV files found.")


    # Concatenate all valid DataFrames
    all_features = pd.concat(valid_features, ignore_index=True)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(all_features)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # PCA
    pca = PCA(n_components=20)
    X_reduced = pca.fit_transform(X_scaled)

    # ---------------------------
    # Step 4: Clustering with HDBSCAN
    # ---------------------------

    # HDBSCAN does not require specifying the number of clusters
    clusterer = hdbscan.HDBSCAN(min_cluster_size=80)
    cluster_labels = clusterer.fit_predict(X_reduced)

    # Add cluster labels to the original features
    all_features['cluster'] = cluster_labels

    # ---------------------------
    # Step 5: Save Clustered Results
    # ---------------------------

    all_features.to_csv(output_path, index=False)
    print(f"Clustered motif features saved to: {output_path}")

    # ---------------------------
    # Step 6: Visualize Clusters (Optional)
    # ---------------------------
    # Plot first 2 PCA components colored by cluster label
    # plt.figure(figsize=(10, 7))
    # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap='Spectral', s=50)
    # plt.title("Motif Cluste

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster motif features using HDBSCAN.")
    parser.add_argument('--input', type=str, required=True, help='Path to the directory containing CSV files.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the clustered results.')
    args = parser.parse_args()

    # Update paths based on command line arguments
    right_hands = os.path.join(args.input, "*_right_hand.csv")
    left_hands = os.path.join(args.input, "*_left_hand.csv")
    pose = os.path.join(args.input, "*_pose.csv")
    output_path = args.output


    process_clustering(feature_csv_path=right_hands, output_path=output_path+"_right_hand.csv")
    process_clustering(feature_csv_path=left_hands, output_path=output_path+"_left_hand.csv")
    process_clustering(feature_csv_path=pose, output_path=output_path+"_pose.csv")