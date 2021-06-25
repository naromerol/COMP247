# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:23:57 2021

@author: Nestor Romero leon - 301133331
MidTerm COMP 247
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_nestor = pd.read_csv('C:/Dev/CENTENNIAL/S2-COMP247/COMP247/MidTerm/voice.csv')
print('*** Initial Analysis per column\n')
#Basic shape of the dataset and column names
print(df_nestor.shape)
print(df_nestor.columns.values)

#Columns, data types and missing values >> Seems complete for most columns except Q75, IQR
print(df_nestor.info())

incomplete = df_nestor[df_nestor.isnull().any(axis=1)]
#6 incomplete rows

#Exploration of few samples
print(df_nestor.head(10))
#Range per feature, min max and basic statistics informations
print(df_nestor.describe())

#label analysis
print('Class values (label)')
print(df_nestor["label"].unique())
print('Class values (Counts)')
#balanced classes 1584 males and 1584 females
df_nestor["label"].value_counts()

#Histogram
df_nestor.hist(bins=50, figsize=(30,30))
plt.savefig(r'C:/Dev/CENTENNIAL/S2-COMP247/COMP247/MidTerm/histogram.png')

####DATA PREPARATION
df_nestor_features = df_nestor.drop('label', axis=1)
df_nestor_target = df_nestor['label'].copy()

print(len(df_nestor_features))
print(len(df_nestor_target))

from sklearn.model_selection import train_test_split
X_train_nestor, X_test_nestor, y_train_nestor, y_test_nestor  = train_test_split(df_nestor_features, df_nestor_target, 
                                                     test_size=0.25, random_state=31)
print(X_train_nestor.shape)
print(X_test_nestor.shape)
print(y_train_nestor.shape)
print(y_test_nestor.shape)

##STEP 5 - ENCODER
from sklearn.preprocessing import OrdinalEncoder
LR_nestor = OrdinalEncoder()
##STEP6 - TRANSFORM labels for training set
y_train_nestor_transformed = LR_nestor.fit_transform(np.array(y_train_nestor).reshape(-1,1))

###PIPELINE
##STEP 7
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline_nestor = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

X_train_nestor_transformed = num_pipeline_nestor.fit_transform(X_train_nestor)

###BUILD AND TEST THE MODEL
from sklearn.svm import SVC
clf_svm_nestor = SVC(kernel="rbf", gamma='auto', random_state=31)

pipe = Pipeline([
            ('num_pipeline', num_pipeline_nestor),
            ("svm_nestor", clf_svm_nestor)
        ])

pipe.fit(X_train_nestor_transformed, y_train_nestor_transformed.ravel())

##Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X_train_nestor_transformed, y_train_nestor_transformed.ravel(), cv=3, scoring="accuracy")
print ('All scores:', scores)
print  ('Average score:', scores.mean())

y_test_nestor_transformed = LR_nestor.fit_transform(np.array(y_test_nestor).reshape(-1,1))
X_test_nestor_transformed = num_pipeline_nestor
