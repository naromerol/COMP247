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

#pd.set_option("display.max_columns",35)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
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

#MISING VALUES
# Numerical  > Mean, Categorical > Mode
bicycle_df['Cost_of_Bike'].fillna(bicycle_df['Cost_of_Bike'].mean(), inplace=True)
bicycle_df['Bike_Model'].fillna(bicycle_df['Bike_Model'].mode()[0], inplace= True)
bicycle_df["Bike_Make"].fillna(bicycle_df["Bike_Make"].mode()[0], inplace=True)
bicycle_df["Bike_Colour"].fillna(bicycle_df["Bike_Make"].mode()[0], inplace=True)

bicycle_df.isnull().sum()

#HANDLING UNBALANCED CLASSES
#-------------------------Handling imbalanced dataset---------------
df_missing = bicycle_df
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


#CATEGORICAL VARIABLES ENCODING
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
bicycle_df_encoded = bicycle_df.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
bicycle_df_encoded

bicycle_df.select_dtypes(include=['object']).columns

#DATA SCALING
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(bicycle_df_encoded)
bicycle_df_scaled = pd.DataFrame(x_scaled)
bicycle_df_scaled

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
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
rfe_selector = RFE(estimator=DecisionTreeClassifier())
X = bicycle_df_encoded.drop(columns=['Status'])
y = bicycle_df_encoded['Status']
rfe_selector.fit(X,y)
X.columns[rfe_selector.get_support()]

'''
Index(['Primary_Offence', 'Occurrence_DayOfMonth', 'Occurrence_DayOfYear',
       'Report_Date', 'Report_DayOfMonth', 'Report_DayOfYear', 'Report_Hour',
       'NeighbourhoodName', 'Bike_Make', 'Bike_Model', 'Bike_Colour',
       'Cost_of_Bike', 'Longitude', 'Latitude'],
      dtype='object')

Index(['Primary_Offence', 'Occurrence_Date', 'Occurrence_DayOfMonth',
       'Report_Date', 'Report_DayOfMonth', 'Report_DayOfYear', 'Report_Hour',
       'NeighbourhoodName', 'Location_Type', 'Bike_Model', 'Bike_Speed',
       'Cost_of_Bike', 'Longitude', 'Latitude'],
      dtype='object')
'''

#PIPELINE DEFINITION
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#SPLIT CATEGORICAL AND NUMERIC DATA FOR EACH PIPELINE
date_columns = ['Occurrence_Date','Report_Date']

bicycle_df_cat = bicycle_df.drop(columns=date_columns).select_dtypes(include=['object'])
bicycle_df_num = bicycle_df.select_dtypes(include=['int64', 'float64'])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
    ])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder())
    ])

bicycle_df_cat_tr = cat_pipeline.fit_transform(bicycle_df_cat)
bicycle_df_num_tr = num_pipeline.fit_transform(bicycle_df_num)

#DATA SPLIT
from sklearn.model_selection import train_test_split
X = df_upsampled.drop(columns=['Status'])
y = df_upsampled['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)