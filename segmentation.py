import numpy as np
import stumpy
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def find_segments(m: int = 30, save: bool = False) -> pd.DataFrame:
    X = pd.read_csv("./data/greet_right_hand/normalized_landmarks.csv")
    X_reduced = np.linalg.norm(X, axis=1)
    m = 30  # subsequence length
    mp = stumpy.stump(X_reduced, m)

    cac, regime_locations = stumpy.fluss(mp[:, 1], L=10, n_regimes=2, excl_factor=1)
    res = pd.DataFrame(cac, columns=["CAC"], index=X["frame"].iloc[:len(X["frame"]) - m + 1])
    
    if save:
        res.to_csv("./data/cac.csv")

    return res

# 5. Find segmentation points
# Low CAC values suggest change points
# change_points = np.argsort(cac)[:5]  # top 5 change points

# # 6. Plot
# plt.figure(figsize=(15,5))
# plt.plot(X_reduced, label='Reduced Time Series')
# for cp in change_points:
#     plt.axvline(cp, color='red', linestyle='--', alpha=0.7)
# plt.title('Detected Change Points with FLUSS')
# plt.legend()
# plt.show()


def find_local_minima(df: pd.DataFrame, column_name: str) -> list[float]:
    local_minima = []
    values = df[column_name].values
    indexes = df.index
    
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            local_minima.append(indexes[i])
    
    return local_minima


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find segments in time series data.")
    parser.add_argument("--save", action="store_true", help="Save the results to a CSV file.")
    parser.add_argument("--data", type=str, default="./data/greet_right_hand/normalized_landmarks.csv", help="Path to the data file.")
    parser.add_argument("--m", type=int, default=30, help="Length of the motif.")

    args = parser.parse_args()

    res = find_segments(m=args.m, save=args.save)
    local_min = find_local_minima(res, "CAC")
    print("No of local minima:", len(local_min))
    print("Local minima:", local_min)