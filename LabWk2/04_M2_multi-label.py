# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:09:38 2020

Module 2 multi label
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#Load the data
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
X, y = mnist["data"].to_numpy(), mnist["target"]
X.shape
print(type(X))
print(X[2])
#plot using imshow
some_digit = X[2]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()
print(y[2])
#
y = y.astype(np.uint8)
#
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_large = y_train >= 6
y_train_odd = y_train % 2 == 1
y_multi_label = np.c_[y_train_large, y_train_odd] 
print(y[2])
print(y_multi_label[2])
###
#Let us try a K nearest neighbor model using multi label
some_digit = X[0]
print(y_multi_label[0])
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multi_label)
#predict
print(knn_clf.predict([some_digit]))

###
