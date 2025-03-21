import numpy as np
import pandas as pd
from sklearn import preprocessing

path = "./landmarks_output.csv"

# Read data
def read_data(path):
    data = pd.read_csv(path)
    
    # remove the first column - frame
    data = data.drop(data.columns[0], axis=1)
    
    return data

# Normalize
def normalize(data):
    # Identify unique coordinate types (x, y, z)
    coordinate_types = ['x', 'y', 'z']

    # Normalize each coordinate type independently
    for coord in coordinate_types:
        # Find all columns that belong to this coordinate type
        cols = [col for col in data.columns if col.endswith(f'_{coord}')]

        # Apply MinMaxScaler to those columns
        # minmax scales all the values between 0-1
        data[cols] = pd.DataFrame(
            preprocessing.MinMaxScaler().fit_transform(data[cols]),
        )        

    return data

# Rotate
def rotate(data):
    
    pass

# Scale
def scale(data):

    pass

# PCA
def pca(data, n_components):

    pass

print(normalize(read_data(path)))