import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sn


#pd.get_option("display.max_columns")
#pd.set_option("display.max_columns",35)

project_folder = os.getcwd();
project_data = 'Bicycle_Thefts.csv'
#print(os.path.join(project_folder, project_data))
bike_ds = pd.read_csv(os.path.join(project_folder, project_data),sep=',')

bike_ds.shape
bike_ds.head()
bike_ds.describe()
bike_ds.dtypes
bike_ds.columns
bike_ds.count() 
#missing Bike_Make(25448), Bike_Model(15923), 
#Bike_Colour(23508), Cost_of_Bike(23825)

bike_ds['Occurrence_Date'].describe()
bike_ds['Occurrence_Month'].describe()
bike_ds['Occurrence_Year'].describe() #2009-2020 span
bike_ds['Occurrence_DayOfWeek'].describe()
bike_ds['Occurrence_DayOfMonth'].describe()
bike_ds['Occurrence_DayOfYear'].describe()
bike_ds['Occurrence_Hour'].describe()

bike_ds['Report_Date'].describe()
bike_ds['Report_Month'].describe()
bike_ds['Report_Year'].describe()
bike_ds['Report_DayOfWeek'].describe()
bike_ds['Report_DayOfMonth'].describe()
bike_ds['Report_DayOfYear'].describe()
bike_ds['Report_Hour'].describe()

bike_ds['Primary_Offence'].describe()
bike_ds['Division'].describe()
bike_ds['City'].describe()
bike_ds['Hood_ID'].describe()
bike_ds['NeighbourhoodName'].describe()

bike_ds[['Hood_ID','NeighbourhoodName']].head() #id + label required?

bike_ds['Location_Type'].describe()
bike_ds['Premises_Type'].describe()
bike_ds['Bike_Make'].describe()
bike_ds['Bike_Model'].describe()
bike_ds['Bike_Type'].describe()
bike_ds['Bike_Speed'].describe()
bike_ds['Bike_Colour'].describe()

bike_ds['Cost_of_Bike'].describe() #missing values 23825
bike_ds['Status'].describe()
bike_ds['Longitude'].describe()
bike_ds['Latitude'].describe()

#candidates to remove
bike_ds['X'].describe()
bike_ds['Y'].describe()
bike_ds['OBJECTID'].describe()
bike_ds['event_unique_id'].describe()
bike_ds['ObjectId2'].describe()

bike_ds.hist(figsize=(16,14))
#bike_ds.drop(labels='X', axis=1, inplace=True)
#bike_ds.drop(labels='Y', axis=1, inplace=True)
correlations = bike_ds.corr()
fig, ax = plt.subplots(figsize=(16,14))
sn.heatmap(correlations, annot=True, ax=ax)
plt.show()


#Shape 25569 x 35

#Column Processing
#Months codification
bike_ds.replace('January',1, inplace=True)
bike_ds.replace('February',2, inplace=True)
bike_ds.replace('March',3, inplace=True)
bike_ds.replace('April',4, inplace=True)
bike_ds.replace('May',5, inplace=True)
bike_ds.replace('June',6, inplace=True)
bike_ds.replace('July',7, inplace=True)
bike_ds.replace('August',8, inplace=True)
bike_ds.replace('September',9, inplace=True)
bike_ds.replace('October',10, inplace=True)
bike_ds.replace('November',11, inplace=True)
bike_ds.replace('December',12, inplace=True)

bike_ds['Report_Month'].describe()
bike_ds['Occurrence_Month'].describe()
bike_ds['Report_Month'].hist()

bike_ds.replace('Monday',1, inplace=True)
bike_ds.replace('Tuesday',2, inplace=True)
bike_ds.replace('Wednesday',3, inplace=True)
bike_ds.replace('Thursday',4, inplace=True)
bike_ds.replace('Friday',5, inplace=True)
bike_ds.replace('Saturday',6, inplace=True)
bike_ds.replace('Sunday',7, inplace=True)

bike_ds['Report_DayOfWeek'].describe()
bike_ds['Occurrence_DayOfWeek'].describe()

correlations_month_days = bike_ds.corr()
fig, ax = plt.subplots(figsize=(16,14))
sn.heatmap(correlations_month_days, annot=True, ax=ax)
plt.show()

len(bike_ds['Primary_Offence'].unique()) #66 types of offence, stolen?
bike_ds.groupby('Primary_Offence').count()


bike_ds['Status'].unique()
bike_ds[['Primary_Offence','Status']].groupby('Status').count()

bike_ds['Status'].replace('STOLEN',1, inplace=True)
bike_ds['Status'].replace('RECOVERED',2, inplace=True)
bike_ds['Status'].replace('UNKNOWN',3, inplace=True)

correlations_status = bike_ds.corr()
fig, ax = plt.subplots(figsize=(16,14))
sn.heatmap(correlations_status, annot=True, ax=ax)
plt.show()

from sklearn import preprocessing
offence_encoder = preprocessing.LabelEncoder()
offence_encoder.fit(bike_ds['Primary_Offence'])
offence_encoder.classes_

bike_ds['Primary_Offence'] = offence_encoder.transform(bike_ds['Primary_Offence'])
bike_ds['Primary_Offence'].describe()


correlations_offence = bike_ds.corr()
fig, ax = plt.subplots(figsize=(16,14))
sn.heatmap(correlations_offence, annot=True, ax=ax)
plt.show()

bike_ds['Status'].groupby('Primary_Offence').count()
bike_ds['Primary_Offence'].hist()
offence_encoder.inverse_transform([58,59,8])

offence_counts = bike_ds[['Primary_Offence','OBJECTID']].groupby('Primary_Offence').count()
