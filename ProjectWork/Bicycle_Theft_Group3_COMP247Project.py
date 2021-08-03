'''
COMP247 - SUPERVISED LEARNING
Bicycle Theft Classifier
Group 3 - Section 002
TEAM MEMBERS
Diwakar Singh - 301185459
Hitesh Dharmadhikari - 301150694
Jefil Tasna John Moran - 301149710
Shrikant Kale - 301150258
Nestor Romero Leon - 301133331
'''

import os
import pandas as pd
import numpy as np

#pd.set_option("display.max_columns",35)

project_folder = os.getcwd();
dataset_file = 'Bicycle_Thefts.csv'
dataset_full_path = os.path.join(project_folder, dataset_file)
print('Reading dataset file from: ', dataset_full_path)
bicycle_df = pd.read_csv(dataset_full_path,low_memory=False)

#INITIAL DATA EXPLORATION
print(bicycle_df.shape)
print(bicycle_df.head())
print(bicycle_df.info())
print(bicycle_df.describe())
print(bicycle_df.dtypes)
print(bicycle_df.columns)

#Status variable selected as label
bicycle_df['Status'].value_counts()
bicycle_df['Status'].count()

#Missing and null values
isna_df = bicycle_df.isna().sum()
isna_df[isna_df>0]
isnull_df = bicycle_df.isnull().sum()
isnull_df[isnull_df>0]

#REMOVE COLUMNS
bicycle_df.drop(columns=['X', 'Y','OBJECTID','event_unique_id','ObjectId2'], axis=1, inplace=True)

#MISSING VALUES
# Numerical  > Mean, Categorical > Mode
'''
bicycle_df['Cost_of_Bike'].fillna(bicycle_df['Cost_of_Bike'].mean(), inplace=True)
bicycle_df['Bike_Model'].fillna(bicycle_df['Bike_Model'].mode()[0], inplace= True)
bicycle_df["Bike_Make"].fillna(bicycle_df["Bike_Make"].mode()[0], inplace=True)
bicycle_df["Bike_Colour"].fillna(bicycle_df["Bike_Make"].mode()[0], inplace=True)
'''

bicycle_df.isnull().sum()

#FULL DATA CORRELATION
#corr_matrix = bicycle_df_encoded.corr()

#FEATURE SELECTION

#Alternative 1
'''
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
df_target = bicycle_df_encoded['Status']
df_features = bicycle_df_encoded[bicycle_df_encoded.columns.difference(['Status'])]
model = LogisticRegression(solver='lbfgs',max_iter = 10000)
rfe = RFE(model,n_features_to_select=15)
rfe = rfe.fit(df_features,df_target.values.ravel().astype('int'))
rfe.support_
rfe.ranking_
df_features.columns[rfe.support_]
Index(['Bike_Type', 'City', 'Division', 'Occurrence_Date', 'Occurrence_Hour',
       'Occurrence_Month', 'Occurrence_Year', 'Premises_Type',
       'Primary_Offence', 'Report_Date', 'Report_DayOfMonth', 'Report_Year'],
      dtype='object')
Index(['Bike_Speed', 'Bike_Type', 'City', 'Division', 'Occurrence_Date',
       'Occurrence_Hour', 'Occurrence_Month', 'Occurrence_Year',
       'Premises_Type', 'Primary_Offence', 'Report_Date', 'Report_DayOfMonth',
       'Report_Hour', 'Report_Month', 'Report_Year'],
      dtype='object')
'''

#Alternative 2
feature_df = bicycle_df
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
feature_df = feature_df.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')


X = feature_df.drop(columns=['Status'], axis=1)
Y = feature_df['Status']

#RFE Feature Selection
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
rfe_selector = RFE(estimator=DecisionTreeClassifier())
rfe_selector.fit(X,Y)
X.columns[rfe_selector.get_support()]


#OVERSAMPLING TO HANDLE IMBALANCED DATA

bicycle_df['Status'].value_counts()

bicycle_df['Status'] = [1 if b=='RECOVERED' else 0 for b in bicycle_df.Status]

df_majority = bicycle_df[bicycle_df.Status==0]
df_minority = bicycle_df[bicycle_df.Status==1]
 
# Upsample minority class
from sklearn.utils import resample
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=25261,   # to match majority class
                                 random_state=42) # reproducible results
 
# Combine majority class with upsampled minority class
df_oversampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_oversampled.Status.value_counts()

#PIPELINE

#Pipelines and data preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

numeric_df = list(df_oversampled.select_dtypes(include=[np.number]).columns)
cat_df = list(df_oversampled.select_dtypes(exclude=[np.number]))


num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
    ])

#bicycle_df_tr = num_pipeline.fit_transform(numeric_df)

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
     ("encoder", OneHotEncoder())
    ])

#cat_pipeline_tr = cat_pipeline.fit_transform(cat_df)


full_pipeline = ColumnTransformer([
        ("num", num_pipeline, numeric_df),
        ("cat", cat_pipeline, cat_df),
        
    ])

bicycle_df_tr = full_pipeline.fit_transform(df_oversampled)