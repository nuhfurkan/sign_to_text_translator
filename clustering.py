import numpy as np
import pandas as pd
import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read multi-dimensional time-series data from a pandas DataFrame
def read_data_from_dataframe(df):
    return [df.values]  # Treats all columns as value columns, assuming sequential entries

# Compute DTW distance matrix
def compute_dtw_distance_matrix(sequences):
    num_series = len(sequences)
    distance_matrix = np.zeros((num_series, num_series))
    
    for i in range(num_series):
        for j in range(i + 1, num_series):
            dist, _ = fastdtw.fastdtw(sequences[i], sequences[j], dist=euclidean)
            distance_matrix[i, j] = distance_matrix[j, i] = dist
    
    return distance_matrix

# Perform clustering
def cluster_sequences(distance_matrix, num_clusters=3):
    scaler = StandardScaler()
    dist_scaled = scaler.fit_transform(distance_matrix)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(dist_scaled)


df = pd.read_csv("./data/final_data.csv")

sequences = read_data_from_dataframe(df)

distance_matrix = compute_dtw_distance_matrix(sequences)

print(distance_matrix)
# clusters = cluster_sequences(distance_matrix)

# print("Cluster assignments:", clusters)
