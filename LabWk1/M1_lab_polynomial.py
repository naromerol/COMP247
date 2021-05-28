# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 08:56:25 2020

@author: mhabayeb
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.scatter(X, y,  color='red')
plt.title('Scatter plot non-linear')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
#print (X_poly)

print (X[0])      # one feature
print (X_poly[0]) #two features

#Now we can fit a linear regerssion
lin_reg = LinearRegression()
m= lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

#Predict


X_plot=np.linspace(0,3,10).reshape(-1,1)
X_plot_poly=poly_features.fit_transform(X_plot)
plt.plot(X,y,"b.")
plt.plot(X_plot_poly[:,1],lin_reg.predict(X_plot_poly),'-r')
plt.title('Scatter plot non-linear')
plt.xlabel('X')
plt.ylabel('y')
plt.show()