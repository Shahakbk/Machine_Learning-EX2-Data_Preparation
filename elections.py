import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedShuffleSplit
import matplotlib.pyplot as plt

# Loading the data.
data_url = 'https://grades.cs.technion.ac.il/grades.cgi?dbbceceead9ba0238f47df22b94b49+2+236756+Spring2019+hw/WCFiles/ElectionsData.csv+7366'
dataset = pd.read_csv(data_url, header=0)

# Y will be the 'Vote' column (the label) and X will be the rest of the columns.
X = dataset.drop(['Vote'], axis=1)
y = dataset['Vote'].values

# Splitting the data into train, validation and test sets divided randomly to 60% : 20% : 20%.
# Since train_test_split only splits into two sets, the procedure will be done twice.

# First step - splitting the whole set into train set & test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Second step - splitting the test set into test set & validation set.
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

