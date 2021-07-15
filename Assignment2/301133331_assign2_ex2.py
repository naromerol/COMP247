import pandas as pd
import numpy as np

'''
COMP247 - Support Vector Machines Assignment
EXERCISE 2
Nestor Romero Leon - 301133331
'''
pd.set_option('display.max_columns', 15)
data_nestor = pd.read_csv('breast_cancer.csv')

##DATA PREPROCESSING
#fill NaN values for bare
data_nestor['bare'].replace({'?':np.nan}, inplace=True)
data_nestor['bare'] = data_nestor['bare'].astype(float)

#Drop ID column
data_nestor.drop('ID', axis=1, inplace=True)

#Separate features from class
data_features = data_nestor.drop('class',axis=1)
data_class = data_nestor['class']


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#Splitting the data
x_train, x_test, y_train, y_test = train_test_split(data_features, data_class,
                                                    test_size=0.2, random_state=31)

#Preprocessing pipeline
num_pipe_nestor = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('scaler', StandardScaler())
    ])

#Classifier pipeline
pipe_svm_nestor = Pipeline([
    ('preprocessor', num_pipe_nestor),
    ('svc', SVC(random_state=31))
    ])

grid_parameters = [{
    'svc__kernel': ['linear', 'rbf','poly'],
    'svc__C':  [0.01,0.1, 1, 10, 100],
    'svc__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    'svc__degree':[2,3]}]


grid_search_nestor = GridSearchCV(estimator=pipe_svm_nestor,
                                  param_grid = grid_parameters,
                                  scoring='accuracy',
                                  refit=True,
                                  verbose=3)

grid_search_nestor.fit(x_train, y_train)

#Grid Results
print('Best Parameters:\n', grid_search_nestor.best_params_)
print('\nBest Estimator:\n', grid_search_nestor.best_estimator_)

best_model_nestor = grid_search_nestor.best_estimator_
y_predictions = best_model_nestor.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_predictions)
print('Accuracy:', accuracy)

#Export the best model
import joblib
joblib.dump(best_model_nestor, "best_model_nestor.pkl")
joblib.dump(pipe_svm_nestor, "pipe_svm_nestor.pkl")