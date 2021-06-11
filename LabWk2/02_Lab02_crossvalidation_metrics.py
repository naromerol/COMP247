# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:23:57 2020
"""
"""
module 2 MINST metrics
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
#Plot an image use the method imshow 
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()
# Check the coresponding lable
y[0]
# change output to integer
y = y.astype(np.uint8)

"""
Binary classification
"""
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
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
##Build your own cross validation use Stratified KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))


# print the confusion matirix need cross validate
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)
# pretend we reached perfection
y_train_perfect_predictions = y_train_5  
confusion_matrix(y_train_5, y_train_perfect_predictions)

# peecision recall f1
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)
"""
Percision re-call tradeoff


####
"""
"""
Scikit-Learn does not let you set the threshold directly,
 but it does give you access to the decision scores that it uses to make predictions.
 Instead of calling the classifier’s predict() method, you can call its decision_function()
 method, which returns a score for each instance,
 and then use any threshold you want to make predictions based on those scores:
The SGDClassifier uses a threshold equal to 0, so the below code returns the same result
 as the predict() method (i.e., True). Let’s raise the threshold:
"""
y_scores = sgd_clf.decision_function([some_digit])
y_scores
threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
#Change the threshold
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
"""
the above This confirms that raising the threshold decreases recall. 
The image actually represents a 5, and the classifier detects it when the threshold is 0, 
but it misses it when the threshold is increased to 8,000.
"""
"""
How to decide on threshhold ?? 
The signed distance to the hyperplane (computed as the dot product 
between the coefficients and the input sample, plus the intercept)
 is given by SGDClassifier.decision_function:
"""
y_scores = sgd_clf.decision_function([some_digit])
y_scores
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    [...] # highlight the threshold and add the legend, axis label, and grid

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

"""
ROC
"""
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)        

plot_roc_curve(fpr, tpr)
plt.show()