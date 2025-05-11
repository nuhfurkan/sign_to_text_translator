import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt
import argparse
import os

# df = pd.read_csv('./data/normalized_landmarks.csv')

# Uncomment to work with only pose_0
# df = df.iloc[:, 1:4]

def process_motifs(data: pd.DataFrame, mps, m):
    """
    Process the motifs and nearest neighbors.
    """
    mps, indices = stumpy.mstump(data, m)

    motif_distances, motif_indices, motif_subspaces, motif_mdls = stumpy.mmotifs(data, mps, indices)

    '''
    Starting with motifs_idx and nn_idx go m steps forward
    Save the indices of the motifs and the nearest neighbors
    Then implement the logic to find the motifs and nearest neighbors in the original read
    '''
    # motifs_idx = np.argmin(mps, axis=1)
    # nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]

    # print(motifs_idx)
    # print(nn_idx)
    '''---------------------------'''

    return mps, indices, motif_distances, motif_indices, motif_subspaces, motif_mdls

def motif_to_feature(distances, indices, subspace, mdl, total_dims):
    feature = {
        "mean_distance": np.mean(distances),
        "std_distance": np.std(distances),
        "min_distance": np.min(distances),
        "max_distance": np.max(distances),
        "mdl_mean": np.mean(mdl),
        "mdl_std": np.std(mdl),
        "subspace_size": len(subspace),
    }

    # One-hot encode the subspace (dimensions involved)
    for dim in range(total_dims):
        feature[f"dim_{dim}"] = 1 if dim in subspace else 0

    return feature

def generate_feature_matrix(motif_distances, motif_indices, motif_subspaces, motif_mdls, data):
    features = []
    total_dims = data.shape[0]

    # Ensure all lists are of the same length
    n_motifs = min(len(motif_distances), len(motif_indices), len(motif_subspaces), len(motif_mdls))

    for i in range(n_motifs):  # for each motif, ensuring we don't go out of range
        f = motif_to_feature(
            motif_distances[i],
            motif_indices[i],
            motif_subspaces[i],
            motif_mdls[i],
            total_dims
        )
        features.append(f)

    import pandas as pd
    df = pd.DataFrame(features)
    return df

def plot_data(data: pd.DataFrame, mps, motifs_idx, nn_idx, m):
    fig, axs = plt.subplots(mps.shape[0] * 2, sharex=True, gridspec_kw={'hspace': 0})

    for k, dim_name in enumerate(data.columns):
        axs[k].set_ylabel(dim_name, fontsize='20')
        axs[k].plot(data[dim_name])
        axs[k].set_xlabel('Time', fontsize ='20')

        axs[k + mps.shape[0]].set_ylabel(dim_name.replace('T', 'P'), fontsize='20')
        axs[k + mps.shape[0]].plot(mps[k], c='orange')
        axs[k + mps.shape[0]].set_xlabel('Time', fontsize ='20')

        axs[k].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
        axs[k].axvline(x=nn_idx[1], linestyle="dashed", c='black')
        axs[k + mps.shape[0]].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
        axs[k + mps.shape[0]].axvline(x=nn_idx[1], linestyle="dashed", c='black')

        if dim_name != 'T3':
            axs[k].plot(range(motifs_idx[k], motifs_idx[k] + m), data[dim_name].iloc[motifs_idx[k] : motifs_idx[k] + m], c='red', linewidth=4)
            axs[k].plot(range(nn_idx[k], nn_idx[k] + m), data[dim_name].iloc[nn_idx[k] : nn_idx[k] + m], c='red', linewidth=4)
            axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10, color='red')
            axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='red')
        else:
            axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10, color='black')
            axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='black')

    plt.show()

def save_motifs(mps, motifs_idx, nn_idx) -> pd.DataFrame:
    """
    Save the motifs and nearest neighbors in a DataFrame.
    """
    motifs = pd.DataFrame(mps)
    motifs['motifs_idx'] = motifs_idx
    motifs['nn_idx'] = nn_idx
    return motifs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find motifs in time series data.')
    parser.add_argument('--m', type=int, default=30, help='Length of the motif.')
    parser.add_argument('--data', type=str, default='./data/normalized_landmarks.csv', help='Path to the data file.')
    # parser.add_argument("--plot", action="store_true", help="Plot the motifs and nearest neighbors.")
    parser.add_argument("--save", type=str, default="./data/motifs.csv", help="Save the motifs and nearest neighbors to a file.")
    args = parser.parse_args()

    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    df = pd.read_csv(args.data)    
    
    pose_columns = [col for col in df.columns if col.startswith('pose')]
    right_hand_columns = [col for col in df.columns if col.startswith('right_hand')]
    left_hand_columns = [col for col in df.columns if col.startswith('left_hand')]

    # Create separate DataFrames
    pose_df = df[pose_columns]
    right_hand_df = df[right_hand_columns]
    left_hand_df = df[left_hand_columns]
    

    mps, indices, motif_distances, motif_indices, motif_subspaces, motif_mdls = process_motifs(pose_df, mps={}, m=args.m)
    print(
        mps.shape,
        indices.shape,
        motif_distances.shape,
        motif_indices.shape,
        motif_subspaces,
        motif_mdls,
        pose_df.shape,
    )
    df = generate_feature_matrix(motif_distances, motif_indices, motif_subspaces, motif_mdls, pose_df)
    df.to_csv(args.save + "_pose.csv", index=False)
    
    mps, indices, motif_distances, motif_indices, motif_subspaces, motif_mdls = process_motifs(right_hand_df, mps={}, m=args.m)
    print(
        mps.shape,
        indices.shape,
        motif_distances.shape,
        motif_indices.shape,
        motif_subspaces,
        motif_mdls,
        right_hand_df.shape,
    )
    df = generate_feature_matrix(motif_distances, motif_indices, motif_subspaces, motif_mdls, right_hand_df)
    df.to_csv(args.save + "_right_hand.csv", index=False)

    mps, indices, motif_distances, motif_indices, motif_subspaces, motif_mdls = process_motifs(left_hand_df, mps={}, m=args.m)
    print(
        mps.shape,
        indices.shape,
        motif_distances.shape,
        motif_indices.shape,
        motif_subspaces,
        motif_mdls,
        left_hand_df.shape,
    )
    df = generate_feature_matrix(motif_distances, motif_indices, motif_subspaces, motif_mdls, left_hand_df)
    df.to_csv(args.save + "_left_hand.csv", index=False)



    # if args.plot:
        # plot_data(mps, motifs_idx, nn_idx, m=args.m)
    

# TODO: not all motifs are detected