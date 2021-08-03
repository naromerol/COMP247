# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 22:16:51 2021

@author: diwak
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
dataset = pd.read_csv("Bicycle_Thefts.csv")
pd.set_option('display.max_columns',80)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 100)
print(dataset.columns.values)
print(type(dataset))
print(len(dataset))
print(dataset.dtypes)
print(dataset.describe())
print(dataset.info())
print("Bikes Details : \n ",dataset['Status'].value_counts())
sns.set_theme(style="darkgrid")
ax = sns.catplot(x = "Status",kind="count", palette="ch:.30",data = dataset)

ax = sns.catplot(y = "Location_Type",kind="count" ,data = dataset)
ax = sns.catplot(y = "Occurrence_Year",kind="count" ,data = dataset)
ax = sns.catplot(y = "Occurrence_Month",kind="count" ,data = dataset)

df_dataset = dataset.drop(columns = ['X','Y','OBJECTID','event_unique_id','ObjectId2'])
print(df_dataset.columns.values)

list_categories_dataset = []
for col, col_type in df_dataset.dtypes.iteritems():
    if col_type == 'O':
        list_categories_dataset.append(col)
    else:
        df_dataset[col].fillna(0,inplace=True)
print("Data Set -  Categories")
print(list_categories_dataset)
print(df_dataset.isnull().any())
print(df_dataset.isnull().sum())








