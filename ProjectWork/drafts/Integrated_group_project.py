'''
COMP247 - SUPERVISED LEARNING
Bicycle Theft Classifier
Group 3 - Section 002
TEAM MEMBERS
Diwakar Singh - 
Hitesh Dharmadhikari - 
Jefil Tasna John Moran - 
Shrikant Kale - 301150258
Nestor Romero Leon - 301133331
'''

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
print(bicycle_df.describe().transpose())
print(bicycle_df.dtypes)
print(bicycle_df.columns)

#Status variable selected as label
print(bicycle_df['Status'].value_counts())
print(bicycle_df['Status'].count())

#Missing and null values
isna_df = bicycle_df.isna().sum()
print(isna_df[isna_df>0])
isnull_df = bicycle_df.isnull().sum()
print(isnull_df[isnull_df>0])

#REMOVE COLUMNS
bicycle_df.drop(columns=['X', 'Y','OBJECTID','event_unique_id','ObjectId2','Occurrence_Date'], axis=1, inplace=True)

#DATA PLOTS
#Frequency of each Status
#create figure and axis
fig, ax = plt.subplots()
# plot histogram
ax.hist(bicycle_df['Status'])
# set title and labels
ax.set_title('Status')
ax.set_xlabel('Status')
ax.set_ylabel('Frequency')

#Occurence Year and number of crimes
sns.distplot(bicycle_df['Occurrence_Year'], bins=2, kde=True)


#Occurence Month countplot
ax = sns.countplot(bicycle_df['Occurrence_Month'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Crimes')
ax.set_ylabel('Months')
plt.tight_layout()
plt.show()


#Primes types according to number of Crimes
ax = sns.countplot(bicycle_df['Premises_Type'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Crimes')
ax.set_ylabel('Premise Types')
plt.tight_layout()
plt.show()


#Top 10 neighbourhoods with most number of crimes
bicycle_df['NeighbourhoodName'].value_counts().head(10).plot.barh(title="Top 10 Neighbourhoods with most number of Crimes")

#Top 10 Primary_Offences
bicycle_df['Primary_Offence'].value_counts().head(10).plot.bar()


#cost of Bike and Status
sns.catplot(x="Status", y="Cost_of_Bike", data=bicycle_df)


#Day of week Crimes
k = bicycle_df['Occurrence_DayOfWeek'].value_counts()
x = bicycle_df['Occurrence_DayOfWeek'].unique()
plt.barh(x, k)
for index, value in enumerate(k):
    plt.text(value, index,
             str(value)) 
plt.show()

ax = sns.catplot(x = "Status",kind="count", palette="ch:.30",data = bicycle_df)
ax = sns.catplot(y = "Location_Type",kind="count" ,data = bicycle_df)
ax = sns.catplot(y = "Occurrence_Year",kind="count" ,data = bicycle_df)
ax = sns.catplot(y = "Occurrence_Month",kind="count" ,data = bicycle_df)

#MISSING VALUES   <-- CHANGED

bicycle_df['Cost_of_Bike'] = bicycle_df['Cost_of_Bike'].fillna(bicycle_df['Cost_of_Bike'].mean())
bicycle_df = bicycle_df.replace(to_replace = "?", value = np.nan)
bicycle_df = bicycle_df.replace(to_replace = "-", value = np.nan)
bicycle_df = bicycle_df.replace(to_replace = "+", value = np.nan)
bicycle_df['Bike_Model'].fillna(bicycle_df['Bike_Model'].mode()[0], inplace= True)
bicycle_df["Bike_Make"].fillna(bicycle_df["Bike_Make"].mode()[0], inplace=True)
bicycle_df["Bike_Colour"].fillna(bicycle_df["Bike_Colour"].mode()[0], inplace=True)


#CATEGORICAL VARIABLES ENCODING-----------
'''
Questions: 
    What happens if we need the original labels for reports or results?
    Case HoodID vs NeighbourhoodName - Diluted ID
'''
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
bicycle_df = bicycle_df.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
bicycle_df


#FEATURE SELECTION-------

X = bicycle_df.drop(columns=['Status'], axis=1)
Y = bicycle_df['Status']

#SelectKBest 
from sklearn.feature_selection import SelectKBest, chi2
df_new = SelectKBest(chi2, k=10).fit_transform(X,Y)

#Following are the top 10 features that are selected by SelectedKBest
#Primary_Offence, Occurence_DayOfYear, Report_Date, Report_DateOfYear, hoodID, Bike_Make, Bike_Model, Cost_Of_Bike, Longitude, Latitude

#RFE Feature Selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
rfe_selector = RFE(estimator=DecisionTreeClassifier())
rfe_selector.fit(X,Y)
X.columns[rfe_selector.get_support()]


#DATA SCALING---------
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(bicycle_df)
bicycle_df_scaled = pd.DataFrame(x_scaled)
bicycle_df_scaled



