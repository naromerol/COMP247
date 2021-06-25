# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:58:35 2020
Module 2 multi class
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
#Load the data
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
X, y = mnist["data"].to_numpy(), mnist["target"]
X.shape
print(type(X))
#plot using imshow
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()
y = y.astype(np.uint8)
#
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

from sklearn.naive_bayes import MultinomialNB
NB_clf = MultinomialNB()
NB_clf.fit(X_train, y_train)
NB_clf.predict([some_digit])
NB_clf.predict_proba([some_digit])
NB_clf.predict_log_proba([some_digit])
NB_clf.score(X_test,y_test)
NB_clf.classes_

cross_val_score(NB_clf, X_train, y_train, cv=3, scoring="accuracy")
