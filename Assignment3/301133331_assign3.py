"""
Assignment 3 - Decision Trees - COMP247
@author: Nestor Romero - 301133331
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

path = os.path.join(os.getcwd(),'student-por.csv');
data_nestor = pd.read_csv(path, sep=';')

#Data Exploration
data_nestor.info()
data_nestor.head(3)
data_nestor.isnull().sum()
obj_cols = data_nestor.select_dtypes('object')
obj_cols_stats = obj_cols.describe()
num_cols = data_nestor.select_dtypes('number')
num_col_stats = num_cols.describe()
#Data plotting for initial analysis
#data_nestor.hist(figsize=(20,15))
#data_nestor.boxplot(figsize=(20,15))

#Create new column > label
data_nestor['pass_nestor'] = 1
data_nestor['pass_nestor'].where(data_nestor['G1'] + data_nestor['G2'] + data_nestor['G3']>=35,0,inplace=True)

#Remove grade columns
data_nestor.drop(['G1','G2','G3'], axis=1, inplace=True)

#Split dataset - create target variable
features_nestor = data_nestor[data_nestor.columns[:-1]]
target_variable_nestor = data_nestor[data_nestor.columns[-1]]
print(target_variable_nestor.value_counts())

#Numeric and categorical features
numeric_features_nestor = features_nestor.select_dtypes('number')
cat_features_nestor = features_nestor.select_dtypes('object')

#Column Transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print(cat_features_nestor.columns)
transformer_nestor = ColumnTransformer([('encoder',OneHotEncoder(),
                                        cat_features_nestor.columns)],
                                       remainder='passthrough')

#Decission Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf_nestor = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=31)

#Pipeline creation
from sklearn.pipeline import Pipeline
pipeline_nestor = Pipeline([
    ('transformer', transformer_nestor),
    ('classifier', clf_nestor)
    ])

#Dataset split
from sklearn.model_selection import train_test_split
X_train_nestor, X_test_nestor, y_train_nestor, y_test_nestor = train_test_split(features_nestor, target_variable_nestor, 
                                                 test_size=0.2, random_state=31)

#10-Fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

pipeline_nestor.fit(X_train_nestor, y_train_nestor)
cross_val = KFold(n_splits=10, shuffle=True, random_state=31)

scores = cross_val_score(pipeline_nestor, X_train_nestor, y_train_nestor,
                         scoring='accuracy', cv=cross_val, n_jobs=1)

scores_mean = scores.mean()
print(scores)
print(scores_mean)


#graphviz tree visualization
import graphviz 
from sklearn import tree
#Could not find a clean way to construct this feature name list
names = np.array(transformer_nestor.named_transformers_['encoder'].get_feature_names()).tolist()
names2 = numeric_features_nestor.columns.tolist()
names = names + names2

tree_graph = tree.export_graphviz(clf_nestor, 
                      out_file=None, 
                      feature_names=names,  
                      class_names=['Fail','Pass'],  
                      filled=True, rounded=True,  
                      special_characters=True) 
graph = graphviz.Source(tree_graph) 
graph.render("tree")
graph 

#Accuracy scores test and train
train_accuracy = cross_val_score(pipeline_nestor, X_train_nestor, y_train_nestor,
                         scoring='accuracy', cv=cross_val, n_jobs=1).mean()

test_accuracy = cross_val_score(pipeline_nestor, X_test_nestor, y_test_nestor,
                         scoring='accuracy', cv=cross_val, n_jobs=1).mean()

print('Training Accuracy: ', train_accuracy)
print('Test Accuracy: ', test_accuracy)

#Predict test data
y_predict = pipeline_nestor.predict(X_test_nestor)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

conf_matrix = confusion_matrix(y_test_nestor, y_predict)
accuracy = accuracy_score(y_test_nestor, y_predict)
precision = precision_score(y_test_nestor, y_predict)
recall = recall_score(y_test_nestor, y_predict) 

print('Confusion Matrix:', conf_matrix)
print('Accuracy', accuracy)
print('Precision', precision)
print('Recall', recall)

#Fine-tune the model
from sklearn.model_selection import RandomizedSearchCV

params = {'classifier__min_samples_split' : range(10,300,20),
             'classifier__max_depth': range(1,30,2),
             'classifier__min_samples_leaf':range(1,15,3)}
 

rand_grid_search = RandomizedSearchCV(estimator = pipeline_nestor,
                                               scoring='accuracy',param_distributions = params,
                                               cv=5, n_iter = 7, refit = True, verbose = 3, 
                                               random_state=31)

rand_grid_search.fit(X_train_nestor, y_train_nestor)

#Print Scores from grid search
print()
print("Best Parameters")
print(rand_grid_search.best_params_)
print("Best Score")
print(rand_grid_search.best_score_)
print("Best Estimator")
print(rand_grid_search.best_estimator_)

#Fitting test data
fine_tuned_model = rand_grid_search.best_estimator_
y_grid_predict = fine_tuned_model.predict(X_test_nestor)

conf_matrix = confusion_matrix(y_test_nestor, y_grid_predict)
accuracy = accuracy_score(y_test_nestor, y_grid_predict)
precision = precision_score(y_test_nestor, y_grid_predict)
recall = recall_score(y_test_nestor, y_grid_predict) 

print('Confusion Matrix:', conf_matrix)
print('Accuracy', accuracy)
print('Precision', precision)
print('Recall', recall)


#Save model and pipeline
import joblib
joblib.dump(fine_tuned_model, "decisiontree_assign3.pkl")
joblib.dump(pipeline_nestor, "pipeline_assign3.pkl")