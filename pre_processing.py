import numpy as np

path = "../data/landmarks_output.csv"

# Read data
def read_data(path):
    data = np.genfromtxt(path, delimiter=',')
    return data

# Normalize
def normalize(data):

    pass

# Rotate
def rotate(data):

    pass

# Scale
def scale(data):

    pass

# PCA
def pca(data, n_components):

    pass
