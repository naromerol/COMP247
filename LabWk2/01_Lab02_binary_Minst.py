# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 08:08:00 2020
Module 2 MINST
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#Set the seed
np.random.seed(42)
#Load the dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

# split the data to attributes and output
X, y = mnist["data"].to_numpy(), mnist["target"]
#Check the data shape 
X.shape
y.shape
# Each hand writing image is a 28 by 28 pixel
28*28
#Plot an image use the method imshow 
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()
# Check the coresponding label
y[0]
# change output to integer
y = y.astype(np.uint8)
#Define a function to plot a group of images 10 per row
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
# Call the function to print the first 100 images
plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images, images_per_row=10)
#save_fig("more_digits_plot")
plt.show()
# The set is already shufled so split into train test 60K for training and 10 k for testing
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
"""
Binary classification
"""
# Use a binary classifier i.e. only classifiy if 5 is true and the rest are not true
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])
#cross validate 3-folds
from sklearn.model_selection import cross_val_score
scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print ('All scores:', scores)
print  ('Average score:', scores.mean())
#Wow 95% 
#Let us look at a dump classifier the base which is 50% -50%
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#That’s right, it has over 90% accuracy! This is simply because only about 10% of the images are 5s,
# so if you always guess that an image is not a 5, you will be right about 90% of the time.


