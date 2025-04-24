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

    '''
    Starting with motifs_idx and nn_idx go m steps forward
    Save the indices of the motifs and the nearest neighbors
    Then implement the logic to find the motifs and nearest neighbors in the original read
    '''
    motifs_idx = np.argmin(mps, axis=1)
    nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]

    print(motifs_idx)
    print(nn_idx)
    '''---------------------------'''

    return mps, motifs_idx, nn_idx

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
    parser.add_argument("--plot", action="store_true", help="Plot the motifs and nearest neighbors.")
    parser.add_argument("--save", type=str, default="./data/motifs.csv", help="Save the motifs and nearest neighbors to a file.")
    args = parser.parse_args()

    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    data = pd.read_csv(args.data)    
    
    mps, motifs_idx, nn_idx = process_motifs(data, mps={}, m=args.m)

    if args.plot:
        plot_data(mps, motifs_idx, nn_idx, m=args.m)
    
    data = save_motifs(mps, motifs_idx, nn_idx)
    print(data.head())
    
    data.to_csv(args.save, index=False)

# TODO: not all motifs are detected