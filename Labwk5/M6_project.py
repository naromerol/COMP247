# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 23:44:22 2020
Chapter 2  Hands-on machine learning with SCi-kit learn, tensor flow and Keras
"""
import os
import tarfile
import urllib
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
#pandas setting
pd.set_option('display.max_columns', 500)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#Initial investigation
housing = load_housing_data()
###How local path is defined???

housing.head(3)
housing.describe()
housing.shape
housing.dtypes
housing.columns.values
housing.info()

###Column Analysis - Unique values for deteremining the class and count for each class
housing["ocean_proximity"].unique()
housing["ocean_proximity"].value_counts()

#Initial plots -histograms of each attribute 

housing.hist(bins=50, figsize=(20,15))

plt.savefig(r'C:\Dev\CENTENNIAL\S2-COMP247\COMP247\Labwk5\project_plots\attribute_histogram_plots.png')

# to make this code output identical at every run
np.random.seed(42)

# For illustration only we can use develop in numpy. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
len(train_set)
len(test_set)
print(test_set.head(4))
### Hashing
#housing_with_id = housing.reset_index() #from pandas adds an index column
#####

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
test_set.head(5)
####
#Stratified sampling , Assume median income is important
housing["median_income"].hist()
plt.xlabel("median_income")
plt.ylabel("frequency")
plt.savefig(r'C:\Dev\CENTENNIAL\S2-COMP247\COMP247\Labwk5\project_plots\median_income.hist.png')

#
plt.scatter(housing["median_income"],housing["median_house_value"])
plt.xlabel("median_income")
plt.ylabel("median_house_value")
plt.savefig(r'C:\Dev\CENTENNIAL\S2-COMP247\COMP247\Labwk5\project_plots\income_versus_value.png')
# create new variable for income as categories
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts()
housing["income_cat"].hist()
plt.xlabel("median_income_bins")
plt.savefig(r'C:\Dev\CENTENNIAL\S2-COMP247\COMP247\Labwk5\project_plots\income_cat.png')


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# Let us compare
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    

"""
Discover and Visualizations
"""
#make a copy of the training data
housing = strat_train_set.copy()
#Since there is geographical information (latitude and longitude), it is a good idea to create a scatterplot of all districts to visualize the data
housing.plot(kind="scatter", x="longitude", y="latitude")
# set alpha
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.savefig(r'C:\Dev\CENTENNIAL\S2-COMP247\COMP247\Labwk5\project_plots\housing_density_training.png')

#
##S size, Color map for variable
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
plt.savefig(r'C:\Dev\CENTENNIAL\S2-COMP247\COMP247\Labwk5\project_plots\housing_density_population_training.png')

#Looking for Correlations
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# another way use pandas scatter


attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.savefig(r'C:\Dev\CENTENNIAL\S2-COMP247\COMP247\Labwk5\project_plots\housing_correlations.png')
#most promising attribute
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)

#Experimenting with Attribute Combinations
### New attributes could find better correlations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


#SEABORN
np.random.seed(sum(map(ord,"categorical")))
sns.stripplot(x= "ocean_proximity",y = "median_income", data = housing)
sns.stripplot(x= "ocean_proximity",y = "median_income", data = housing, jitter = True)

#We can also add nested categories but not our case (we only have one category so far)
# Box plots
sns.boxplot(x= "ocean_proximity",y = "median_income", data = housing)
sns.boxplot(x= "ocean_proximity",y = "median_house_value", data = housing)
#sns.boxplot(x= "ocean_proximity",y = "median_income",hue= "median_house_value", data = housing)
sns.violinplot(x=housing.housing_median_age,y=housing.median_house_value)
sns.violinplot(x=housing.housing_median_age,y=housing.ocean_proximity)


#Examine new features
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


"""
Prepare the data
"""
# make a copy and drop the label (target) create a seprate frame for target# drop labels for training set
housing = strat_train_set.drop("median_house_value", axis=1) 
housing_labels = strat_train_set["median_house_value"].copy()

# check for missing values on any column
sample_incomplete_rows = housing[housing.isnull().any(axis=1)]
print (len(sample_incomplete_rows))
## Three options
sample_1 = sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1: Get rid of the corresponding districts. dropna()
sample_2 = sample_incomplete_rows.drop("total_bedrooms", axis=1)       #option2 Get rid of the whole attribute. drop() axis 1
median = housing["total_bedrooms"].median() 
print(median)
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3 Set the values to some value (zero, the mean, the median, etc.). fillna()

### use the imputer
from sklearn.impute import SimpleImputer    #import
imputer = SimpleImputer(strategy="median")  # define the object and set the strategy
imputer.strategy                             #check the strategy

housing_num = housing.drop("ocean_proximity", axis=1)  # drop the non numeric fields
# alternatively: housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)                           # fit the data using the imputer
housing_num.median().values                         # check median for the data set
imputer.statistics_                               # The imputer has simply computed the median of each attribute and stored the result in its statistics_ instance variable.

#Optional we can apply the imputer on the whole dataset
imp = imputer.transform(housing_num)  # result ndarray
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)
housing_tr.loc[sample_incomplete_rows.index.values]


# Handling Text and Categorical Attributes
#Transformers
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(8)
# from preprocessing ordinal encoder
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:8]
ordinal_encoder.categories_
#But this is not what we need
#One hot encoder
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
housing_cat_1hot.toarray()

####     Custom transformers 
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

print(housing.values)
print(type(housing))

#####
# Scalar & Transformers
####
housing_scaled = housing.drop("ocean_proximity", axis=1) 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #define the instance
print(scaler.fit(housing_scaled))
print(scaler.mean_)
housing_scaled1 = (scaler.fit_transform(housing_scaled))
np.std(housing_scaled1, axis = 0)
np.mean(housing_scaled1, axis = 0)


##################

# Pipelines

#################
# build a pipeline for preprocessing the numerical attributes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr

#full transformation Column Transformer
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared

#####
"""
Train a Model 
"""
# linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
## Test on training
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
#Calculate the root mean square error
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

#Check meaan absolute error
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae
#OPPS a case of underfitting

#let try a different algorithm decision trees
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
#Wow it overfit
#10 - fold Cross validate  Tree
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
#10 fold cross validate Linear
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

#Let us try a third algorithm Random forests
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
#
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
#
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
##
#SVM
###
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse
# Save the model


"""
Hyper tune the model
"""
#Grid search

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
#Best parameters
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
# save results to a data frame :)
    
pd.DataFrame(grid_search.cv_results_)
####
#Randomized search
###
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

#
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
#
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
#
"""
Finally test the model :)
"""
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print (final_rmse)

