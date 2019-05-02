import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats import normaltest
from scipy.stats import kurtosis
from sklearn.neighbors import LocalOutlierFactor
from matplotlib import pyplot
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, chi2
import random


################################################################################
# 1. Load the Election Challenge data from the ElectionsData.csv file
################################################################################
def load_data(datapath):
    data = pd.read_csv(datapath, header=0)
    data = data.replace([np.inf, -np.inf], np.nan)

    return data


################################################################################
# 3.c. Data Cleansing - Type/Value modification
################################################################################
def categorical_to_int(data, category_name, temp_name):
    data[temp_name] = data[category_name].astype("category")
    data[category_name] = data[temp_name].cat.rename_categories(range(data[temp_name].nunique())).astype(int)

    samples_with_missing_values = np.where(data[temp_name].isnull())[0]
    for i in samples_with_missing_values:
        data.ix[i, category_name] = np.nan

    return data


def modify_types(data):
    temp_name = 'temp'
    print("\nModifying types for:")

    for feature in data.columns:
        for i in range(len(data[feature])):
            if data[feature][i] != np.nan:
                if isinstance(data[feature][i], str):
                    categorical_to_int(data, feature, temp_name)
                    print(feature, end=', ')

                break

    data = data.drop(['temp'], axis=1)
    print("\n*** Done modifying types ***")
    return data


################################################################################
# 2. Identify and set the correct type of each attribute
################################################################################
def remove_negative_values_from_feature(data, feature):
    negative_values_indices = np.where(data[feature] < 0)[0]

    for negative_value_index in negative_values_indices:
        data.ix[negative_value_index, feature] = np.nan

    return data[feature], negative_values_indices.size


def remove_negative_values_from_data(data):
    sum_removed = 0
    print("\nRemoving negative values for:")
    for feature in data.columns:
        for i in range(len(data[feature])):
            if data[feature][i] != np.nan:
                if isinstance(data[feature][i], float):
                    data[feature], removed = remove_negative_values_from_feature(data, feature)
                    sum_removed += removed

                    print(feature, end=", ")

                break

    print("\n*** Removed ", sum_removed, " negative values ***")
    return data


################################################################################
# 3.a. Imputation
################################################################################
def fill_missing_values(data):
    sum_filled = 0
    print("\nFilling missing values for:")

    for i, feature in enumerate(data.columns):
        samples_with_missing_values = np.where(data[feature].isnull())[0]
        if len(samples_with_missing_values):
            for j in range(len(data[feature])):
                if data[feature][j] != np.nan:
                    if isinstance(data[feature][j], float):
                        imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
                    else:
                        imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

                    filled = imp_mean.fit_transform(data)

                    print(feature, end=", ")
                    break

        for missing_value in samples_with_missing_values:
            data.ix[missing_value, feature] = filled[missing_value][i]
            sum_filled += 1

    print("\n*** Filled ", sum_filled, " values ***")
    return data


################################################################################
# 3.b. Data Cleansing - Outlier Detection
################################################################################
def add_outliers_indicator_vector(data):
    print("\nAdding outliers indicator vector")

    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    outliers_vector = clf.fit_predict(data) # inlier == 1 , outlier == -1

    data['outlier_indicator'] = outliers_vector
    data['outlier_indicator'] = data['outlier_indicator'].map({1: 0, -1: 1}).astype(int)

    print("\n*** Marked ", np.where(outliers_vector < 0)[0].size, " as outliers***")
    return data


################################################################################
# 3.d. Normalization (scaling)
################################################################################
def z_score_normalization(feature, feature_mean, feature_std):
    return (feature - feature_mean) / feature_std


def min_max_normalization(feature, feature_min, feature_max):
    return 2 * ((feature - feature_min) / (feature_max - feature_min)) - 1


def normalize_data(data):
    print("\nNormalizing feature number:")

    stat, p = normaltest(data)  # might need to turn train_set to np array
    alpha = 1e-200
    is_normal = (p > alpha)

    for i, feature in enumerate(data):
        # print("before: ", stat[i], p[i], is_normal[i])
        # pyplot.hist(data[feature])
        # pyplot.show()

        print(i, end=' ')
        if feature == 'Vote':
            continue

        # k = kurtosis(data[feature])

        # is_binary = (data[feature].nunique() == 2)
        # if is_binary:
        #     continue

        if is_normal[i]:
            feature_mean = data[feature].mean()
            feature_std = data[feature].std()
            data[feature] = min_max_normalization(data[feature], feature_mean, feature_std)

        else:
            feature_min = data[feature].min()
            feature_max = data[feature].max()
            data[feature] = z_score_normalization(data[feature], feature_min, feature_max)

        # feature_min = data[feature].min()
        # feature_max = data[feature].max()
        # data[feature] = z_score_normalization(data[feature], feature_min, feature_max)

        # print("after")
        # pyplot.hist(data[feature])
        # pyplot.show()

    print("\n*** Done normalizing ***")
    return data


################################################################################
# 4. Feature Selection
################################################################################
def variance_threshold_filter(data, p):
    print("Using variance threshold for feature selection with p = ", p)
    size_before = data.columns.size

    selector = VarianceThreshold(threshold=p)
    selector.fit(data)

    filtered = np.where(selector.variances_ < p)[0]
    for i, feature_index in enumerate(filtered):
        if feature_index == 0:
            continue

        data = data.drop([data.columns[feature_index - i]], axis=1)

    print("\n*** Done using variance threshold. Filtered ", size_before - data.columns.size, " features ***")
    return data


def mutual_info_k_best(data):
    print("Using mutual info k-best for feature selection:")
    size_before = data.columns.size

    train_data_X = data.drop(['Vote'], axis=1).values
    train_data_Y = data.Vote.values

    univariate_filter = SelectKBest(mutual_info_classif, k=23).fit(train_data_X, train_data_Y)
    # print(univariate_filter.scores_)
    deleted = 0
    for i, is_chosen in enumerate(univariate_filter.get_support(False)):
        if is_chosen:
            continue

        data = data.drop([data.columns[i + 1 - deleted]], axis=1)
        deleted += 1

    # train_data_X = data.drop(['Vote'], axis=1).values
    # train_data_Y = data.Vote.values
    #
    # univariate_filter = SelectKBest(mutual_info_classif, k=23).fit(train_data_X, train_data_Y)
    # print(univariate_filter.scores_)

    print("\n*** Done using mutual info k-best. Filtered ", size_before - data.columns.size, " features ***")
    return data

################################################################################
# Non-Mandatory(Bonus) B. Relief
################################################################################
# def euclidean(X, i, j):
#     sum = 0
#     num_features = len(X[0])
#     for idx in range(num_features):
#         sum = sum + (X[i][idx] - X[j][idx]) ** 2
#     return np.sqrt(sum)
#
# def get_nearhit(X, y, p):
#     min_dist = np.inf
#     min_idx = np.inf
#
#     num_features = len(X[0])
#     # Iterate the features.
#     for i in range(num_features):
#         # Check if it's a miss.
#         if y[i] != y[p]:
#             cur_dist = euclidean(X, i, p)
#             if cur_dist < min_dist:
#                 min_dist = cur_dist
#                 min_idx = i
#
#     return min_idx if min_idx != np.inf else p
#
# def get_nearmiss(X, y, p):
#     min_dist = np.inf
#     min_idx = np.inf
#
#     num_features = len(X[0])
#     # Iterate the features.
#     for i in range(num_features):
#         # Check if it's a hit.
#         if y[i] == y[p]:
#             cur_dist = euclidean(X, i, p)
#             if cur_dist < min_dist:
#                 min_dist = cur_dist
#                 min_idx = i
#
#     return min_idx if min_idx != np.inf else p
#
# def relief(X, y, threshold, num_iter=20):
#     # Init an empty weigths vector.
#     weights = np.zeros(len(X[0]))
#     features = set([])
#
#     # Algorithm iterations:
#     for t in range(num_iter):
#         # Pick a random sample.
#         p = random.randint(0, X.shape[0])
#
#         nearhit = get_nearhit(X, y, p)
#         nearmiss = get_nearmiss(X, y, p)
#
#
#         # Iterating the features and updating the weights.
#         for i in range(len(X[0])):
#             weights[i] = weights[i] + (X[p][i] - X[nearmiss][i]) ** 2 - (X[p][i] - X[nearhit][i]) ** 2
#
#     # Returns a set of the best features.
#     idx = 0
#     for i in range(len(X[0])):
#         if weights[idx] > threshold:
#             features.add(i + 1)
#         idx = idx + 1
#
#     return features
#
# def call_relief(data):
#     print("Using relief for feature selection:")
#     size_before = data.columns.size
#
#     train_data_X = data.drop(['Vote'], axis=1).values
#     train_data_Y = data.Vote.values
#
#     print(relief(train_data_X, train_data_Y, 0, num_iter=1000))
#
#     print("\n*** Done using relief. Filtered ", size_before - data.columns.size, " features ***")
#     return data


################################################################################
# Main
################################################################################
def main():
    data = load_data('ElectionsData_orig.csv')

    # print(data.columns.size)
    data = modify_types(data)
    data = remove_negative_values_from_data(data)

    data = fill_missing_values(data)

    # call_relief(data)
    data = variance_threshold_filter(data, 0.1)
    data = mutual_info_k_best(data)
    data = mutual_info_k_best(data)

    normalize_data(data)

    # mutual_info_k_best(data)
    # variance_threshold_filter(data, 0.1)

    # print(data.columns.size)
    # print("\n", data.head())
    # print("\n", data['Num_of_kids_born_last_10_years'].head())
    return


if __name__ == '__main__':
    main()