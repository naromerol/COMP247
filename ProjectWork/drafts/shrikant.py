# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 12:30:15 2021

@author: Shrikant Kale
"""

#install tabulate

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


#importing dataset into pandas dataframe
path = "E:/Centennial Semester 2/Supervised Learning/Assignments/Final Project"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename) 

df_bicycle = pd.read_csv(fullpath, low_memory=False)


print(df_bicycle.head())
print(df_bicycle.info())
print(df_bicycle.columns)

#value counts for Status feature
df_bicycle['Status'].value_counts()

#filtering columns with numeric dtype
df_bicycle._get_numeric_data().columns

#filtering columns with object dtype
df_bicycle.select_dtypes(exclude=["number"]).columns


#statistical analysis of dataset
df_numeric = df_bicycle._get_numeric_data()
df_numeric.describe()
df_numeric.corr()

df_numeric = df_numeric.drop(['X', 'Y', 'ObjectId2'], axis = 1)

#missing values for bike_make(121), bike_model(9646), bike_colour(2061), cost_of_bike(1744)
df_bicycle.isnull().sum()

#Frequency of each Status
#create figure and axis
fig, ax = plt.subplots()
# plot histogram
ax.hist(df_bicycle['Status'])
# set title and labels
ax.set_title('Status')
ax.set_xlabel('Status')
ax.set_ylabel('Frequency')

#Occurence Year and number of crimes
sns.distplot(df_bicycle['Occurrence_Year'], bins=2, kde=True)


#Occurence Month countplot
ax = sns.countplot(df_bicycle['Occurrence_Month'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Crimes')
ax.set_ylabel('Months')
plt.tight_layout()
plt.show()


#Primes types according to number of Crimes
ax = sns.countplot(df_bicycle['Premises_Type'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Crimes')
ax.set_ylabel('Premise Types')
plt.tight_layout()
plt.show()


#top 10 neighbourhoods with most number of crimes
df_bicycle['NeighbourhoodName'].value_counts().head(10).plot.barh(title="Top 10 Neighbourhoods with most number of Crimes")

#Top 10 Primary_Offences
df_bicycle['Primary_Offence'].value_counts().head(10).plot.bar()


#cost of Bike and Status
sns.catplot(x="Status", y="Cost_of_Bike", data=df_bicycle)


#Day of week Crimes
k = df_bicycle['Occurrence_DayOfWeek'].value_counts()
x = df_bicycle['Occurrence_DayOfWeek'].unique()
plt.barh(x, k)
for index, value in enumerate(k):
    plt.text(value, index,
             str(value)) 
plt.show()


#---------------Feature selection, handling missing data

#Handling missing value in Bike Make
df_missing = df_bicycle
df_missing = df_missing.replace(to_replace = "?", value = np.nan)
df_missing = df_missing.replace(to_replace = "-", value = np.nan)

df_missing['Bike_Model'].fillna(df_missing['Bike_Model'].mode()[0], inplace= True)

df_missing.isnull().sum()

#handling missing nan values in Bike Make

#replacing nan values with median of column
df_missing["Bike_Make"].fillna(df_missing["Bike_Make"].mode()[0], inplace=True)

#--------Categorical data management-----------------------------

#encoded_data = pd.get_dummies(df_bicycle, columns =['Primary_Offence', 'Occurrence_Month','Occurrence_DayOfWeek','Report_Month', 'Report_DayOfWeek',
#                                                   'Division','NeighbourhoodName','Location_Type','Premises_Type','Bike_Make','Bike_Model','Bike_Type'])





#-------------------------Handling imbalanced dataset---------------
df_missing['Status'].value_counts()

df_missing['Status'] = [1 if b=='RECOVERED' else 0 for b in df_missing.Status]

# Separate majority and minority classes
df_majority = df_missing[df_missing.Status==0]
df_minority = df_missing[df_missing.Status==1]
 
# Upsample minority class
from sklearn.utils import resample
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=25261,    # to match majority class
                                 random_state=42) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.Status.value_counts()

