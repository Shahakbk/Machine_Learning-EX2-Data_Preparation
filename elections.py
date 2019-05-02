# TODO clear redundant imports when done.
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import utils, relief


# Loading the data.

dataset = pd.read_csv('ElectionsData_orig.csv', header=0)
dataset = dataset.replace([np.inf, -np.inf], np.nan)

# The label will be 'Vote' column and X will be the rest of the columns.
y = dataset['Vote']
X = dataset.drop(['Vote'], axis=1)

"""
Splitting the data.
"""

# Splitting the data into train, validation and test sets divided randomly to 60% : 20% : 20%.
# Since train_test_split only splits into two sets, the procedure will be done twice.
# The indices lists will hold the original indices of the samples according to the full data set.

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

# Create a 'clean' train dataset without any NaN values.
train_data_no_nan = pd.concat([X_train, y_train], axis=1)
train_data_no_nan = train_data_no_nan.dropna()
X_train_no_nan = train_data_no_nan.drop(['Vote'], axis=1)
y_train_no_nan = train_data_no_nan['Vote']

"""
Marking missing values.
"""

# Create a DF to hold True/False for indicating whether or not values were initially empty.
dataset_nan = dataset.isnull()
# Temporarily fill the missing values with '0'.
dataset = dataset.fillna(0)
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
X_test = X_test.fillna(0)

"""
Converting non-numeric data.
"""

train_data = pd.concat([X_train, y_train], axis=1)
train_data_no_nan = pd.concat([X_train_no_nan, y_train_no_nan], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data = utils.convert_discrete_features(train_data)
train_data_no_nan = utils.convert_discrete_features(train_data_no_nan)
val_data = utils.convert_discrete_features(val_data)
test_data = utils.convert_discrete_features(test_data)

X_train = train_data.drop(['Vote'], axis=1)
y_train = train_data['Vote']

X_train_no_nan = train_data_no_nan.drop(['Vote'], axis=1)
y_train_no_nan = train_data_no_nan['Vote']

X_val = val_data.drop(['Vote'], axis=1)
y_val = val_data['Vote']

X_test = test_data.drop(['Vote'], axis=1)
y_test = test_data['Vote']

"""
Imputing missing values with regression/ classification decision trees.
"""

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

# Filling missing values.
# TODO removing line this might cause an error.
X_train_new = X_train.copy()

for feature in continuous_features:
    clf = DecisionTreeRegressor(criterion="mse")
    clf.fit(train_data_no_nan.drop(['Vote'], axis=1), train_data_no_nan[feature])

    for idx in range(len(indices_train)):
        # This means the sample originally had a NaN value in this feature.
        if dataset_nan[feature][indices_train[idx]] == True:
            sample = X_train.iloc[idx, :]
            [prediction] = (clf.predict([sample]))
            X_train_new.ix[indices_train[idx], feature] = prediction


for feature in discrete_features:
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train_no_nan, y_train_no_nan)

    for idx in range(len(indices_train)):
        # This means the sample originally had a NaN value in this feature.
        if dataset_nan[feature][indices_train[idx]] == True:
            sample = X_train.iloc[idx, :]
            X_train[feature][indices_train[idx]] = clf.predict(sample)

X_train = X_train_new