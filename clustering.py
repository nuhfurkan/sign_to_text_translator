import pandas as pd
import glob
import os
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

    # Loop through each CSV file and check if it is empty
    for file in csv_files:
        df = pd.read_csv(file)
        
        if df.empty:
            print(f"Skipping empty file: {file}")
        else:
            valid_features.append(df)

    # If no valid files were found, raise an error
    if not valid_features:
        raise ValueError("No valid (non-empty) CSV files found.")

    # Concatenate all valid DataFrames
    all_features = pd.concat(valid_features, ignore_index=True)

    # ---------------------------
    # Step 2: Normalize the Features
    # ---------------------------

    # Assuming all columns are numeric from motif_to_feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_features)

    # ---------------------------
    # Step 3: Dimensionality Reduction (Optional)
    # ---------------------------

    # Reduce dimensionality to 10 for better clustering performance
    pca = PCA(n_components=10)
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
    # plt.title("Motif Clusters")
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.colorbar(label="Cluster")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster motif features using HDBSCAN.")
    parser.add_argument('--input', type=str, required=True, help='Path to the directory containing CSV files.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the clustered results.')
    args = parser.parse_args()

    # Update paths based on command line arguments
    feature_csv_path = os.path.join(args.input, "*.csv")
    print(feature_csv_path)
    output_path = args.output

    process_clustering(feature_csv_path=feature_csv_path, output_path=output_path)
