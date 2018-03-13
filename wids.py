# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 22:14:57 2018

@author: Vinothini Pushparaja
"""
# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('train.csv')
data_types = dataset.dtypes

# Moving the prediction variable to last column for ease of access
header = list(dataset)
dataset = dataset[[header for header in dataset if header not in ['is_female']] + ['is_female']]

# Getting the percentage of missing data from each column
# Removing the column that has more than 30% of NaN values
null_percent = dataset.isnull().mean()
missing_columns = null_percent[null_percent > 0.30].index
dataset.drop(missing_columns, axis=1, inplace=True)

# new variable to check the missing percentage for remaining features
missing_percentage = dataset.isnull().mean().sort_values(ascending = False)

# Fill NaNs with the most frequent value from each column
dataset = dataset.apply(lambda x:x.fillna(x.value_counts().index[0]))

# Converting all features to category data type
# previously they were in int64 and float64 type
# for col_name in list(dataset):
#    dataset[col_name] = dataset[col_name].astype('category', copy=False)
    
# dataset.dtypes - check the changes with this instruction

#colnames = list(dataset)
#dataset['AA7'].value_counts()

############################################################
# DATA PREPROCESSING
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# df = dataset.apply(labelencoder.fit_transform, axis=1)
# df2 = df.get_dummies(df[df.columns[1:-1]], drop_first = True)
df = dataset.apply(labelencoder_X.fit_transform, axis = 1)

# Splitting the Independent and Dependent features
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

onehotencoder = OneHotEncoder(categorical_features='all')
X = onehotencoder.fit_transform(X).toarray()
############################################################


# Splitting the dataset into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Applying Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 30)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression to Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set for Logistic Regression
y_pred = classifier.predict(X_test)

# Confusion Matrix for Logistic Regression
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

# Fitting Random Forest to Training set
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=600, random_state=0)
forest_model = forest.fit(X_train, y_train)

y_pred_forest = forest_model.predict(X_test)

# Confusion Matrix & Accuracy for Random Forest
cm_forest = confusion_matrix(y_test, y_pred_forest)

accuracy_forest = accuracy_score(y_test, y_pred_forest)
################################################################
# TESTING DATA

testing_set = pd.read_csv('test.csv')

# Getting the percentage of missing data from each column
# Removing the column that has more than 30% of NaN values
test_null_percent = testing_set.isnull().mean()
test_missing_columns = test_null_percent[test_null_percent > 0.30].index
testing_set.drop(test_missing_columns, axis=1, inplace=True)

# new variable to check the missing percentage for remaining features
# missing_percentage = dataset.isnull().mean().sort_values(ascending = False)

# Fill NaNs with the most frequent value from each column
testing_set = testing_set.apply(lambda x:x.fillna(x.value_counts().index[0]))

############################################################
# Comparing columns in both dataframes
df1 = df.iloc[:, :-1].reindex_axis(sorted(df.iloc[:, :-1].columns), axis = 1)
df2 = testing_set.reindex_axis(sorted(testing_set.columns), axis = 1)

set(df1.iloc[:, :-1].columns) == set(df2.iloc[:, :-1].columns)
############################################################
# DATA PREPROCESSING
# df = dataset.apply(labelencoder.fit_transform, axis=1)
# df2 = df.get_dummies(df[df.columns[1:-1]], drop_first = True)
labelencoder_test = LabelEncoder()
new_testing_set = testing_set.apply(labelencoder_test.fit_transform, axis = 1)

# Splitting the Independent and Dependent features
test_X = new_testing_set.iloc[:, 1:].values

onehotencoder_test = OneHotEncoder(categorical_features='all')
test_X = onehotencoder_test.fit_transform(test_X).toarray()
############################################################

pca2 = PCA(n_components = 30)
test_X = pca2.fit_transform(test_X)
explained_variance_test = pca2.explained_variance_ratio_

test_pred = classifier.predict(test_X)

testing_set['is_female'] = test_pred

### Random Forest
test_pred_forest = forest.predict(test_X)
testing_set['is_female_forest'] = test_pred_forest

############################################################
# Writing the result to a file

import csv

testing_set[['test_id', 'is_female_forest']].to_csv('submission_file_2.csv', columns = ['test_id', 'is_female_forest'])
    











