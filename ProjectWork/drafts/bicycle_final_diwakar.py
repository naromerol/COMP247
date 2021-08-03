# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:26:23 2021

@author: diwak
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.utils import resample
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
dataset_bicycle = pd.read_csv("Bicycle_Thefts.csv")
pd.set_option('display.max_columns',80)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 100)


#to get the column names
print(dataset_bicycle.columns.values)
#description
# data type of column in a a DataFrame
print(dataset_bicycle.dtypes)
# to get the count mean std min iqr max of each column
print(dataset_bicycle.describe())
# to get non null count and dtypes
print(dataset_bicycle.info())
#to see categorical column
list_categories_dataset = []
for col, col_type in dataset_bicycle.dtypes.iteritems():
    if col_type == 'O':
        list_categories_dataset.append(col)
        
    else:
        dataset_bicycle[col].fillna(0,inplace=True)
print("Data Set -  Categories \n",list_categories_dataset)
#range 
dataset_bicycle.describe().max() - dataset_bicycle.describe().min()
#to see the missing values
print(dataset_bicycle.isnull().any())
print(dataset_bicycle.isnull().sum())
#To check unique values
print(dataset_bicycle['Primary_Offence'].unique())
print(len(dataset_bicycle['Primary_Offence'].unique()))
print(dataset_bicycle['Division'].unique())
print(len(dataset_bicycle['Division'].unique()))
print(dataset_bicycle['City'].unique())
print(len(dataset_bicycle['City'].unique()))
print(dataset_bicycle['Hood_ID'].unique())
print(len(dataset_bicycle['Hood_ID'].unique()))
print(dataset_bicycle['NeighbourhoodName'].unique())
print(len(dataset_bicycle['NeighbourhoodName'].unique()))
print(dataset_bicycle['Location_Type'].unique())
print(len(dataset_bicycle['Location_Type'].unique()))
print(dataset_bicycle['Premises_Type'].unique())
print(len(dataset_bicycle['Premises_Type'].unique()))
print(dataset_bicycle['Bike_Make'].unique())
print(len(dataset_bicycle['Bike_Make'].unique()))
print(dataset_bicycle['Bike_Model'].unique())
print(len(dataset_bicycle['Bike_Model'].unique()))
print(dataset_bicycle['Bike_Type'].unique())
print(len(dataset_bicycle['Bike_Type'].unique()))
print(dataset_bicycle['Bike_Speed'].unique())
print(len(dataset_bicycle['Bike_Speed'].unique()))
print(dataset_bicycle['Bike_Colour'].unique())
print(len(dataset_bicycle['Bike_Colour'].unique()))
print(dataset_bicycle['Cost_of_Bike'].unique())
print(len(dataset_bicycle['Cost_of_Bike'].unique()))


#Feature correlation


#Graphs
dataset_bicycle.hist(figsize=(10,12))
plt.show

sns.set_theme(style="darkgrid")
ax = sns.catplot(x = "Status",kind="count", palette="ch:.30",data = dataset_bicycle)
ax = sns.catplot(y = "Location_Type",kind="count" ,data = dataset_bicycle)
ax = sns.catplot(y = "Occurrence_Year",kind="count" ,data = dataset_bicycle)
ax = sns.catplot(y = "Occurrence_Month",kind="count" ,data = dataset_bicycle)

#sns.pairplot(dataset_bicycle)
#sns.catplot(x="Occurrence_Year",y = "Occurrence_Month",data = dataset_bicycle)
df_dataset = dataset_bicycle.drop(columns = ['X','Y','OBJECTID','event_unique_id','ObjectId2','Occurrence_Date'])
print(df_dataset.columns.values)
#removing rows with unknown values in Status
df_dataset.drop(df_dataset.index[df_dataset['Status'] == 'UNKNOWN'],inplace=True)



from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
label_encoder = LabelEncoder()

df_dataset = df_dataset.apply(lambda col: label_encoder.fit_transform(col.astype(str)),axis= 0 ,result_type = 'expand')

bd_major = df_dataset[df_dataset.Status ==0]
bd_minor = df_dataset[df_dataset.Status == 1]
bd_major_downsampled = resample(bd_major,replace =True,n_samples = len(bd_minor),random_state = 0)
bd_downsampled = pd.concat([bd_major_downsampled,bd_minor])
print(bd_downsampled.Status.value_counts())
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
model = LogisticRegression()
rfe = RFE(model,12)
df_target = df_dataset['Status']
df_features = df_dataset[df_dataset.columns.difference(['Status'])]
model = LogisticRegression(solver='lbfgs',max_iter = 10000)
rfe = RFE(model,12)
rfe = rfe.fit(df_features,df_target.values.ravel().astype('int'))
print(rfe.support_)
print(rfe.ranking_)

