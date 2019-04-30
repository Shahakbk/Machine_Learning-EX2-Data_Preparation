# TODO clear redundant imports when done.
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
# from sklearn.impute import IterativeImputer //TODO check why this doesn't work.
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Loading the data.
# data_url = 'https://grades.cs.technion.ac.il/grades.cgi?dbbceceead9ba0238f47df22b94b49+2+236756+Spring2019+hw/WCFiles/ElectionsData.csv+7366'
# dataset = pd.read_csv(data_url, header=0)
dataset = pd.read_csv('ElectionsData.csv', header=0)

# Identify which of the features are objects.
ObjFeat = dataset.keys()[dataset.dtypes.map(lambda x: x=='object')]

# Converting binary features to 0 / 1 representation.
dataset['Looking_at_poles_results_Binary'] = dataset['Looking_at_poles_results'].map( {'No':0, 'Yes':1, 'nan':-1}).astype(int)
dataset.drop(['Looking_at_poles_results'], axis=1)

dataset['Married_Binary'] = dataset['Married'].map( {'No':0, 'Yes':1, 'nan':-1}).astype(int)
dataset.drop(['Married'], axis=1)

dataset['Gender_Binary'] = dataset['Gender'].map( {'Male':0, 'Female':1, 'nan':-1}).astype(int)
dataset.drop(['Gender'], axis=1)

dataset['Voting_Time_Binary'] = dataset['Voting_Time'].map( {'By_16:00':0, 'After_16:00':1, 'nan':-1}).astype(int)
dataset.drop(['Voting_Time'], axis=1)

dataset['Financial_agenda_matters_Binary'] = dataset['Financial_agenda_matters'].map( {'No':0, 'Yes':1, 'nan':-1}).astype(int)
dataset.drop(['Financial_agenda_matters'], axis=1)

# TODO this creates OneHot for all categorial feautres. before that, i wanted to convert binary categories seperately because OneHot is redundant but nan values need to be handled.
# dataset = pd.get_dummies(dataset)

# Y will be the 'Vote' column (the label) and X will be the rest of the columns.
X = dataset.drop(['Vote'], axis=1)
y = dataset['Vote'].values

# Splitting the data into train, validation and test sets divided randomly to 60% : 20% : 20%.
# Since train_test_split only splits into two sets, the procedure will be done twice.

# First step - splitting the whole set into train set & test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Second step - splitting the test set into test set & validation set.
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

# TODO this should probably be changed based on correlation
# Imputing the missing data.
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

"""
# Plotting the data.
plot_2d_space(X_test, y_train, 'Train data - Labels distribution')

"""