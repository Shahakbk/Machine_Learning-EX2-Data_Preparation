import pandas as pd

"""
Converting non-numeric data.
"""


def convert_discrete_features(dataset):
    # Converting the 'Vote' column to numerical.
    dataset['Vote'] = dataset['Vote'].map({'Blues': 0, 'Browns': 1, 'Greens': 2, 'Greys': 3, 'Khakis': 4,
                                           'Oranges': 5, 'Pinks': 6, 'Purples': 7, 'Reds': 8, 'Turquoises': 9,
                                           'Violets': 10, 'Whites': 11, 'Yellows': 12})

    # Converting binary features to 0 / 1 representation.
    dataset['Looking_at_poles_results_Numeric'] = dataset['Looking_at_poles_results'].map({'No': 0, 'Yes': 1})
    dataset['Married_Numeric'] = dataset['Married'].map({'No': 0, 'Yes': 1})
    dataset['Gender_Numeric'] = dataset['Gender'].map({'Male': 0, 'Female': 1})
    dataset['Voting_Time_Numeric'] = dataset['Voting_Time'].map({'By_16:00': 0, 'After_16:00': 1})
    dataset['Financial_agenda_matters_Numeric'] = dataset['Financial_agenda_matters'].map({'No': 0, 'Yes': 1})
    dataset['Will_vote_only_large_party_Numeric'] = dataset['Will_vote_only_large_party'].map({'No': 0, 'Yes': 1, 'Maybe': 0.5})
    dataset['Age_group_Numeric'] = dataset['Age_group'].map({'Below_30': 0, '30-45': 1, '45_and_up': 2})

    # Dropping the previous non-numerical columns.
    dataset = dataset.drop(['Looking_at_poles_results', 'Married', 'Gender', 'Voting_Time', 'Financial_agenda_matters',
                            'Will_vote_only_large_party', 'Age_group'], axis=1)

    # Converting categorical features to OneHot representation.
    tmp_dummies_1 = pd.get_dummies(dataset['Most_Important_Issue'])
    tmp_dummies_2 = pd.get_dummies(dataset['Main_transportation'])
    tmp_dummies_3 = pd.get_dummies(dataset['Occupation'])
    dataset = pd.concat([dataset, tmp_dummies_1, tmp_dummies_2, tmp_dummies_3], axis=1)

    # Dropping the previous non-numerical columns.
    dataset = dataset.drop(['Most_Important_Issue', 'Main_transportation', 'Occupation'], axis=1)

    return dataset
