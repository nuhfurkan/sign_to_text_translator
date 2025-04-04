import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

data = pd.read_csv("./data/final_data.csv")
df = pd.DataFrame(data)
df = df.iloc[:, 1:4]  # Select only the first three columns
print(df.shape)

motifs = pd.read_csv("./data/motifs.csv")
motif_idx = motifs["motifs_idx"]
nn_idx = motifs["nn_idx"]
# print(motif_idx)
# print(nn_idx)
motifs = motifs.iloc[:, :-2]  # Last two columns are not needed - indexes of the motifs
motifs = motifs.transpose()
print(motifs.shape)

def extract_subsequences(df, window_size) -> list[pd.DataFrame]:
    subsequences = []
    
    for i in range(len(df) - window_size + 1):
        subsequences.append(pd.DataFrame(df.iloc[i:i+window_size]))
    
    return subsequences

subsequences = extract_subsequences(df, 30)

# distances = []

# for i, seq in enumerate(subsequences):    
#     distances.append( cdist(
#             seq,
#             motifs,
#             metric="euclidean",
#         ).flatten()
#     )

sorted_distances = []

# Loop over each subsequence
for i, seq in enumerate(subsequences):
    # Calculate pairwise distances between the subsequence and motifs
    dist_matrix = cdist(seq, motifs, metric="euclidean")
    
    # Flatten the distance matrix
    flat_distances = dist_matrix.flatten()

    # Get the indices of the sorted distances
    rows, cols = np.unravel_index(np.argsort(flat_distances), dist_matrix.shape)
    
    # Store the sorted distances with their corresponding subsequence and motif indices
    sorted_pairs_with_distances = list(zip(zip([i] * len(flat_distances), rows, cols), flat_distances))
    
    # Append the sorted pairs for the current subsequence
    sorted_distances.extend(sorted_pairs_with_distances)

# Print the sorted distances and their corresponding subsequences and motifs
for (subseq_idx, motif_idx_row, motif_idx_col), dist in sorted_distances:
    print(f"Subsequence {subseq_idx} (Seq[{motif_idx_row}]) and Motif {motif_idx_col} -> Distance: {dist}")