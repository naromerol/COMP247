# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 01:57:37 2020

@author: mhabayeb
"""
# Load libraries
import pandas as pd
from time import time
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import os
#Load the data
path = os.getcwd()
filename = 'pima-indians-diabetes.csv'
fullpath = os.path.join(path,filename)
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv(fullpath, header=None, names=col_names)
pima.info()
pima.head(2)
#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable
for i in pima:
    print(i,':',pima[i].unique(),'\n')
#check the dirstributions
pima.hist()   
# Chek correlation using pd matrix
pd.plotting.scatter_matrix(pima, alpha = 0.4)  
# Check correlation between fieleds using Seaborn
corr = pima.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


#Splitting Data
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=3, criterion = 'entropy', random_state=42)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Validate using 10 cross fold
print(" 10-fold cross-validation ")
clf = DecisionTreeClassifier(min_samples_split=20,criterion = 'entropy',
                                random_state=42)
clf.fit(X_train, y_train)
scores= cross_val_score(\
   clf, X_train, y_train, cv=10, scoring='f1_macro')

print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                          scores.std()),
                                         end="\n\n" )
#Predict using the test set
y_pred = clf.predict(X_test)
#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

"""
fine tune the model using grid search

"""
# set of parameters to test
param_grid = {
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }
print("-- Grid Parameter Search via 10-fold CV")
dt = DecisionTreeClassifier(criterion = 'entropy')
grid_search = GridSearchCV(dt,
                               param_grid=param_grid,
                               cv=10)
start = time()
grid_search.fit(X_train, y_train)

print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.cv_results_)))

print('Best Parameters are:',grid_search.best_params_)

#Predict the response for test dataset using the best parameters
dt = DecisionTreeClassifier(max_depth =5,min_samples_split= 2, criterion = 'entropy', min_samples_leaf= 10, random_state=42 )
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

### print the tree
import graphviz 
dot_data = tree.export_graphviz(dt, max_depth =5,out_file=None, filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1']) 
graph = graphviz.Source(dot_data) 
graph.render("pima")
graph 





