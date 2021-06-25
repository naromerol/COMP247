# -*- coding: utf-8 -*-
"""
COMP247 - Assignment 1 - Classification
@author: Nestor Romero Leon - 301133331
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#Set the seed
np.random.seed(3011)
#Load the dataset
from sklearn.datasets import fetch_openml
MNIST_Nestor = fetch_openml('mnist_784', version=1)
print(f'MNIST Keys: {MNIST_Nestor.keys()}')

X_Nestor, y_Nestor = MNIST_Nestor['data'].to_numpy(),MNIST_Nestor['target']
print(f'Data Types X_Nestor, y_Nestor: {type(X_Nestor)} {type(y_Nestor)}')
print(f'Data Shapes X_Nestor, y_Nestor: {X_Nestor.shape} {y_Nestor.shape}')


some_digit1, some_digit2, some_digit3 = X_Nestor[7], X_Nestor[5], X_Nestor[0]

y_Nestor[7], y_Nestor[5],y_Nestor[0]

some_digit1_image = some_digit1.reshape(28,28)
some_digit2_image = some_digit2.reshape(28,28)
some_digit3_image = some_digit3.reshape(28,28)

plt.imshow(some_digit1_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()
plt.imshow(some_digit2_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()
plt.imshow(some_digit3_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()

#Section2  - Preprocess the data
y_Nestor = y_Nestor.to_numpy().astype(np.uint8)

y_Nestor = np.where(np.logical_and(y_Nestor > 0, y_Nestor <= 3), 0, y_Nestor)
y_Nestor = np.where(np.logical_and(y_Nestor > 3, y_Nestor <= 6), 1, y_Nestor)
y_Nestor = np.where(np.logical_and(y_Nestor > 6, y_Nestor <= 9), 9, y_Nestor)

print(f'Class 0 Values Frequency\t{len(y_Nestor[y_Nestor == 0])}')
print(f'Class 1 Values Frequency\t{len(y_Nestor[y_Nestor == 1])}')
print(f'Class 9 Values Frequency\t{len(y_Nestor[y_Nestor == 9])}')
plt.hist(y_Nestor)
plt.title('Frequency Histogram')
plt.show()

MNIST_Nestor['target'].hist()
plt.title('Original Class Frequencies')
plt.show()

#Build classification models
X_train, X_test, y_train, y_test = X_Nestor[:60000], X_Nestor[60000:], y_Nestor[:60000], y_Nestor[60000:]

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

NB_clf_Nestor = MultinomialNB()
NB_clf_Nestor.fit(X_train, y_train)


pred_d1, actual_d1 = NB_clf_Nestor.predict([some_digit1]), y_Nestor[7]
pred_d2, actual_d2  = NB_clf_Nestor.predict([some_digit2]), y_Nestor[5]
pred_d3, actual_d3 = NB_clf_Nestor.predict([some_digit3]), y_Nestor[0]

print(f'Digit1 Results Predicted:{pred_d1} Actual: {actual_d1}')
print(f'Digit2 Results Predicted:{pred_d2} Actual: {actual_d2}')
print(f'Digit3 Results Predicted:{pred_d3} Actual: {actual_d3}')


accuracy_score = NB_clf_Nestor.score(X_test,y_test)
print(f'NBClassifier Accuracy: {accuracy_score}')
#NB_clf_Nestor.classes_

scores = cross_val_score(NB_clf_Nestor, X_train, y_train, cv=3, scoring="accuracy")
print(f'NBClassifier 3Fold Cross validation scores: {scores} mean: {scores.mean()}')

y_pred = NB_clf_Nestor.predict(X_test)
cm = confusion_matrix(y_pred,y_test)
print('Confusion Matrix')
print(cm)


#Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

scaler_train = MinMaxScaler().fit(X_train, y_train)
X_train_scaled = scaler_train.transform(X_train)
scaler_test = MinMaxScaler().fit(X_test,y_test)
X_test_scaled = scaler_test.transform(X_test)

#LBFGS CLASSIFIER DATA
LR1_clf_Nestor = LogisticRegression(solver="lbfgs", multi_class='multinomial', 
                                    tol=0.1, max_iter=1000,n_jobs=4, random_state=3011)

LR1_clf_Nestor.fit(X_train_scaled, y_train)

lr1_scores_train = cross_val_score(LR1_clf_Nestor, X_train_scaled, y_train, cv=3, scoring="accuracy")
print(f'Multinomial Classifier <lbfgs>\n\
>>Train Data - 3Fold Cross validation scores: {lr1_scores_train} mean: {lr1_scores_train.mean()}')
      
lr1_scores_test = cross_val_score(LR1_clf_Nestor, X_test_scaled, y_test, cv=3, scoring="accuracy")
print(f'Multinomial Classifier <lbfgs>\n\
>>Test Data - 3Fold Cross validation scores: {lr1_scores_test} mean: {lr1_scores_test.mean()}')


#SAGA CLASSIFIER DATA
LR2_clf_Nestor = LogisticRegression(solver="saga", multi_class='multinomial', 
                                    tol=0.1, max_iter=1000,n_jobs=4, random_state=3011)

LR2_clf_Nestor.fit(X_train_scaled, y_train)

lr2_scores_train = cross_val_score(LR2_clf_Nestor, X_train_scaled, y_train, cv=3, scoring="accuracy")
print(f'Multinomial Classifier <saga>\n\
>>Train Data - 3Fold Cross validation scores: {lr2_scores_train} mean: {lr2_scores_train.mean()}')
      
lr2_scores_test = cross_val_score(LR2_clf_Nestor, X_test_scaled, y_test, cv=3, scoring="accuracy")
print(f'Multinomial Classifier <saga>\n\
>>Test Data - 3Fold Cross validation scores: {lr2_scores_test} mean: {lr2_scores_test.mean()}')

lr1_pred_dig1 = LR1_clf_Nestor.predict([some_digit1])
lr1_pred_dig2 = LR1_clf_Nestor.predict([some_digit2])
lr1_pred_dig3 = LR1_clf_Nestor.predict([some_digit3])

print(f'lbfgs prediction digit1: {lr1_pred_dig1} actual: {y_Nestor[7]}')
print(f'lbfgs prediction digit2: {lr1_pred_dig2} actual: {y_Nestor[5]}')
print(f'lbfgs prediction digit3: {lr1_pred_dig3} actual: {y_Nestor[0]}')

lr2_pred_dig1 = LR2_clf_Nestor.predict([some_digit1])
lr2_pred_dig2 = LR2_clf_Nestor.predict([some_digit2])
lr2_pred_dig3 = LR2_clf_Nestor.predict([some_digit3])

print(f'saga prediction digit1: {lr2_pred_dig1} actual: {y_Nestor[7]}')
print(f'saga prediction digit2: {lr2_pred_dig2} actual: {y_Nestor[5]}')
print(f'saga prediction digit3: {lr2_pred_dig3} actual: {y_Nestor[0]}')


y_predict_lbfgs = LR1_clf_Nestor.predict(X_test_scaled)
cm_lbfgs = confusion_matrix(y_predict_lbfgs, y_test)
print('Confusion Matrix LBFGS')
print(cm_lbfgs)

from sklearn.metrics import precision_score, recall_score
precision_lbfgs = precision_score(y_predict_lbfgs, y_test, average='weighted')
recall_lbfgs = recall_score(y_predict_lbfgs, y_test, average='weighted')

print(f'LBFGS Precision Score: {precision_lbfgs}')
print(f'LBFGS Recall Score: {recall_lbfgs}')
