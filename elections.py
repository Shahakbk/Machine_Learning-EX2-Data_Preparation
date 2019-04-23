# TODO clear redundant imports when done.
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
# from sklearn.impute import IterativeImputer //TODO check why this doesn't work.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Loading the data.
data_url = 'https://grades.cs.technion.ac.il/grades.cgi?dbbceceead9ba0238f47df22b94b49+2+236756+Spring2019+hw/WCFiles/ElectionsData.csv+7366'
# dataset = pd.read_csv(data_url, header=0)
dataset = pd.read_csv('ElectionsData.csv', header=0)


# Y will be the 'Vote' column (the label) and X will be the rest of the columns.
X = dataset.drop(['Vote'], axis=1)
y = dataset['Vote'].values

# Splitting the data into train, validation and test sets divided randomly to 60% : 20% : 20%.
# Since train_test_split only splits into two sets, the procedure will be done twice.

# First step - splitting the whole set into train set & test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Second step - splitting the test set into test set & validation set.
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

# Imputing the missing data.
# imp = IterativeImputer(estimator=impute_estimator) // TODO check why this imputer is not found.
imp = SimpleImputer(strategy="most_frequent")
X_train = imp.fit_transform(X_train)

# Detecting outliers.
# TODO in order for this to work, we first need to convert non-numerical features into numbers.
# For that, we will visualize the data on a graph using PCA with the below function.
def plot_2d_space(X, y, label):
    """
    Plots the data labels as a two-dimensional graph to observe the distribution.
    """
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    colors = ['#228B22', '#8A2BE2']
    markers = ['o', 's']
    size = 25
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            Z[y==l, 0],
            Z[y==l, 1],
            size, c=c, label=l, marker=m, alpha=0.6
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

# Plotting the data.
plot_2d_space(X_test, y_train, 'Train data - Labels distribution')
