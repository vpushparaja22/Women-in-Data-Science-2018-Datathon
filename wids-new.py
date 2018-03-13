# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 22:14:57 2018

@author: Vinothini Pushparaja
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('train.csv')

print(dataset.shape)

# Moving the prediction variable to last column for ease of access
header = list(dataset)
dataset = dataset[[header for header in dataset if header not in ['is_female']] + ['is_female']]

dataset2 = dataset.dropna(axis = 1)
print(dataset2.shape)

X = dataset2.drop(['train_id', 'is_female'], axis = 1)
y = dataset2.is_female

test_dataset = pd.read_csv('test.csv')

test_dataset2 = test_dataset.reindex(columns=X.columns, fill_value = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

N_tree = 500
classifier = RandomForestClassifier(n_estimators = N_tree, random_state=123)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

y_prob = classifier.predict_proba(X_test)
from sklearn import metrics
y_prob
metrics.roc_auc_score(y_test, y_prob[:,1])



Ntree = 500
classifier2 = RandomForestClassifier(n_estimators = Ntree, random_state = 123)
classifier2.fit(X, y)
y_submit = classifier2.predict_proba(test_dataset2)[:,1]
test_dataset['is_female'] = y_submit
ans = test_dataset[['test_id','is_female']]
ans.to_csv('submit.csv', index=None)












