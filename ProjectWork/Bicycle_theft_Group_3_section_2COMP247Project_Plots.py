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
import seaborn as sns
import matplotlib.pyplot as plt

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
print(bicycle_df.describe().transpose())
print(bicycle_df.dtypes)
print(bicycle_df.columns)


bicycle_df['Status'].hist()
bicycle_df['Status'].describe()
bicycle_df['Status'].value_counts()

#Status variable selected as label
print(bicycle_df['Status'].value_counts())
print(bicycle_df['Status'].count())

#Missing and null values
isna_df = bicycle_df.isna().sum()
print(isna_df[isna_df>0])
isnull_df = bicycle_df.isnull().sum()
print(isnull_df[isnull_df>0])


#DATA PLOTS
#STANDARD PLOTS FOR COLUMN
bicycle_df['Longitude'].describe()
bicycle_df['Occurrence_Month'].hist(figsize=(10,7))
bicycle_df['Bike_Type'].value_counts()
bicycle_df['Bike_Type'].value_counts().plot.bar()

#For plotting top values
bicycle_df['Bike_Make'].value_counts()[0:10]
bicycle_df['Bike_Make'].value_counts()[0:10].plot.bar()

#Scatter plot longitude x latitude x cost
bicycle_df.plot(kind="scatter", x='Longitude', y='Latitude', 
                c='Cost_of_Bike', cmap='jet', figsize=(16,10))


#SPECIFIC PLOTS
#CORRELATION
corr_matrix = bicycle_df.corr()
corr_matrix['Status'].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(20,20)) #For all variables large figsize
sns.heatmap(corr_matrix, annot=True, ax=ax)
plt.show()


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


#top 10 neighbourhoods with most number of crimes
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