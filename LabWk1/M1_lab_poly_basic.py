# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:08:48 2020

@author: mhabayeb
"""


import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#help(PolynomialFeatures)

X = np.arange(6).reshape(3, 2)
print(X)
# degree default is 2
# interaction only default False. If true no power(degree) features
# include_bias default is true then include a bias column, the feature in which all polynomial
# powers are zero (i.e. a column of ones - acts as an intercept term in a linear model).
poly = PolynomialFeatures(degree=1, interaction_only=False)
transformed = poly.fit_transform(X)

print(poly.fit_transform(X))

print(transformed.shape[1])

print(poly.get_feature_names(input_features=None))