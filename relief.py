import numpy as np
import random

def euclidean(X, i, j):
    sum = 0
    num_features = len(X.keys())
    for idx in range(num_features):
        sum = sum + (X.iloc[i, idx] - X.iloc[j, idx]) ** 2
    return np.sqrt(sum)

def get_nearhit(X, y, p):
    min_dist = np.inf
    min_idx = np.inf

    num_features = len(X.keys())
    # Iterate the features.
    for i in range(num_features):
        # Check if it's a miss.
        if y.iloc[i] != y.iloc[p]:
            cur_dist = euclidean(X, i, p)
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_idx = i

    return min_idx if min_idx != np.inf else p

def get_nearmiss(X, y, p):
    min_dist = np.inf
    min_idx = np.inf

    num_features = len(X.keys())
    # Iterate the features.
    for i in range(num_features):
        # Check if it's a hit.
        if y.iloc[i] == y.iloc[p]:
            cur_dist = euclidean(X, i, p)
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_idx = i

    return min_idx if min_idx != np.inf else p

def run_algorithm(X, y, threshold, num_iter=20):
    # Init an empty weigths vector.
    weights = np.zeros(len(X.keys()))
    features = set([])

    # Algorithm iterations:
    for t in range(num_iter):
        # Pick a random sample.
        p = random.randint(0, X.shape[0])

        nearhit = get_nearhit(X, y, p)
        nearmiss = get_nearmiss(X, y, p)


        # Iterating the features and updating the weights.
        for i in range(len(X.keys())):
            weights[i] = weights[i] + (X.iloc[p, i] - X.iloc[nearmiss, i]) ** 2 - (X.iloc[p, i] - X.iloc[nearhit, i]) ** 2

    # Returns a set of the best features.
    idx = 0
    for feature in X.keys():
        if weights[idx] > threshold:
            features.add(feature)
        idx = idx + 1

    return features
