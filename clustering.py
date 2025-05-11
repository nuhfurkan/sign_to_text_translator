import pandas as pd
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: Load CSV Files
# ---------------------------

# Change this path to your directory with motif feature CSVs
feature_csv_path = "./data/motifs_features/*.csv"

csv_files = glob.glob(feature_csv_path)

if not csv_files:
    raise FileNotFoundError("No CSV files found in the specified directory.")

# Load and concatenate all feature CSVs
all_features = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

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

output_path = "./data/motif_clusters.csv"
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
