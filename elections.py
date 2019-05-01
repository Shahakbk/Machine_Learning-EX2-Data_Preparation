# TODO clear redundant imports when done.
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import utils

# Loading the data.
# data_url = 'https://grades.cs.technion.ac.il/grades.cgi?dbbceceead9ba0238f47df22b94b49+2+236756+Spring2019+hw/WCFiles/ElectionsData.csv+7366'
# dataset = pd.read_csv(data_url, header=0)
dataset = pd.read_csv('ElectionsData_orig.csv', header=0)
dataset = dataset.replace([np.inf, -np.inf], np.nan)

"""
Marking missing values.
"""
# Create a DF to hold True/False for indicating whether or not values were initially empty.
dataset_nan = dataset.isnull()
# Temporarily fill the missing values with '0'.
dataset = dataset.fillna(0)

# Y will be the 'Vote' column (the label) and X will be the rest of the columns.
y = dataset['Vote']
X = dataset.drop(['Vote'], axis=1)

"""
Splitting the data.
"""

# Splitting the data into train, validation and test sets divided randomly to 60% : 20% : 20%.
# Since train_test_split only splits into two sets, the procedure will be done twice.

# First step - splitting the whole set into train set & a temporary set.
indices = range(X.shape[0])
X_train, X_tmp, y_train, y_tmp, indices_train, indices_tmp = train_test_split(X, y, indices, test_size=0.4, random_state=1)

# Second step - splitting the the temporary set into test set & validation set.
indices = range(X_tmp.shape[0])
X_test, X_val, y_test, y_val, indices_test, indices_val = train_test_split(X_tmp, y_tmp, indices, test_size=0.5, random_state=1)

# Correcting the indices of the train & validation sets.
for i in range(len(indices_test)):
    indices_test[i] = indices_tmp[indices_test[i]]

for i in range(len(indices_val)):
    indices_val[i] = indices_tmp[indices_val[i]]

"""
Converting non-numeric data.
"""

train_data = pd.concat([X_train, y_train], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data = utils.convert_discrete_features(train_data)
val_data = utils.convert_discrete_features(val_data)
test_data = utils.convert_discrete_features(test_data)

X_train = train_data.drop(['Vote'], axis=1)
y_train = train_data['Vote']

X_val = val_data.drop(['Vote'], axis=1)
y_val = val_data['Vote']

X_test = test_data.drop(['Vote'], axis=1)
y_test = test_data['Vote']

"""
Imputing missing values with regression/ classification decision trees.
"""

# Create a 'clean' train dataset without any NaN values.
train_data_no_nan = pd.concat([X_train, y_train], axis=1)
train_data_no_nan = train_data_no_nan.dropna()
y_train_no_nan = train_data_no_nan['Vote']
X_train_no_nan = train_data_no_nan.drop(['Vote'], axis=1)

pd.DataFrame.to_csv(X_train, 'X_train.csv')
pd.DataFrame.to_csv(X_val, 'X_val.csv')
pd.DataFrame.to_csv(X_test, 'X_test.csv')
pd.DataFrame.to_csv(y_train, 'y_train.csv')


discrete_features = ['Occupation_Satisfaction', 'Most_Important_Issue', 'Looking_at_poles_results', 'Married', 'Gender',
                     'Voting_Time', 'Will_vote_only_large_party', 'Last_school_grades', 'Age_group',
                     'Number_of_differnt_parties_voted_for', 'Number_of_valued_Kneset_members', 'Main_transportation',
                     'Occupation', 'Num_of_kids_born_last_10_years', 'Financial_agenda_matters']
continuous_features = ['Avg_monthly_expense_when_under_age_21', 'AVG_lottary_expanses',
                       'Avg_monthly_expense_on_pets_or_plants', 'Avg_environmental_importance',
                       'Financial_balance_score_(0-1)', '%Of_Household_Income', 'Yearly_IncomeK',
                       'Overall_happiness_score', 'Garden_sqr_meter_per_person_in_residancy_area',
                       'Avg_Residancy_Altitude', 'Yearly_ExpensesK', '%Time_invested_in_work',
                       'Avg_government_satisfaction', 'Avg_Satisfaction_with_previous_vote',
                       'Avg_monthly_household_cost', 'Phone_minutes_10_years', 'Avg_size_per_room',
                       'Weighted_education_rank', '%_satisfaction_financial_policy', 'Avg_monthly_income_all_years',
                       'Political_interest_Total_Score', 'Avg_education_importance']

for feature in continuous_features:
    clf = DecisionTreeRegressor(criterion="mse")
    clf.fit(X_train_no_nan, y_train_no_nan)

    for idx in range(len(indices_train)):
        # This means the sample originally had a NaN value in this feature.
        if dataset_nan[feature][indices_train[idx]] == True:
            sample = X_train.iloc[idx, :]
            X_train[feature][indices_train[idx]] = clf.predict(X_train.iloc[idx, :])


for feature in discrete_features:
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train_no_nan, y_train_no_nan)

    # TODO check 'Numerical' columns
    for idx in range(len(indices_train)):
        # This means the sample originally had a NaN value in this feature.
        if dataset_nan[feature][indices_train[idx]] == True:
            sample = X_train.iloc[idx, :]
            X_train[feature][indices_train[idx]] = clf.predict(X_train.iloc[idx, :])



"""
# First of all, we try to find correlation between features in order for the imputing to be as accurate as possible.
corr_matrix = dataset.corr(method='pearson', min_periods=1)

# TODO this should probably be changed based on correlation
# Imputing the missing data.
imp = SimpleImputer(strategy="most_frequent")
X_train = imp.fit_transform(X_train)

"""

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
# plot_2d_space(X_train, y_train, 'Train data - Labels distribution')
"""

"""
# Performing features selection.
# Starting with ReliefF algorithm as a filter method.

knn = KNeighborsClassifier(n_neighbors=4)
sfs = SFS(knn, k_features=3, forward=True, floating=False, scoring='accuracy', cv=0)
sfs = sfs.fit(X_train, y_train)

# TODO set threshold based on the biggest categorical feature.
# 'discrete_threshold' is set to 4 so that any feature with over 13 unique values will be considered continuous.
# 'n_jobs' is set to -1 so that all available cores will be used for the algorithm.
relieff = ReliefF(n_features_to_select=3, n_neighbors=100, discrete_threshold=13)
X_train_reduced = relieff.fit_transform(X_train, y_train)

"""