# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
path = "C:/Users/mhabayeb/Documents/courses_cen/COMP309 new/data/"
filename = 'boston.csv'
fullpath = os.path.join(path,filename)
#load
boston_mayy = pd.read_csv(fullpath)
# check types and missing
print(boston_mayy.info())
print(boston_mayy.isnull().sum())
#Visulize distribution
boston_mayy.hist()
plt.hist(boston_mayy['age'])
plt.plot(boston_mayy['age'])
#assign the first 13 variables of the datast as predictor variables and
# the last one (MEDV) as the target variable
colnames=boston_mayy.columns.values.tolist()
predictors=colnames[:13]
target=colnames[13]
x=boston_mayy[predictors]
y=boston_mayy[target]
# build the tree using the whole data
regression_tree = DecisionTreeRegressor(min_samples_split=30,min_samples_leaf=10,random_state=0)
regression_tree.fit(x,y)
# predict
reg_tree_pred=regression_tree.predict(boston_mayy[predictors])
boston_mayy['pred']=reg_tree_pred
cols=['pred','medv']
boston_mayy[cols]
#cross vlidate
crossvalidation = KFold(n_splits=x.shape[0], shuffle=True, random_state=1)
score = np.mean(cross_val_score(regression_tree, x, y,scoring='neg_mean_absolute_error', cv=crossvalidation,n_jobs=1))
print(score)
#
crossvalidation = KFold(n_splits=x.shape[0], shuffle=True, random_state=1)
score_mse = np.mean(cross_val_score(regression_tree, x, y,scoring='neg_mean_squared_error', cv=crossvalidation,n_jobs=1))
print(score_mse)
#regression tree feature importance
regression_tree.feature_importances_