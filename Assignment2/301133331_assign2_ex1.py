import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
COMP247 - Support Vector Machines Assignment
EXERCISE 1
Nestor Romero Leon - 301133331
'''
pd.set_option('display.max_columns', 15)
data_nestor = pd.read_csv('breast_cancer.csv')

#INITIAL DATA EXPLORATION
print(data_nestor.shape)    #(699,11) shape
print(data_nestor.info())
print(data_nestor.head(5))
print(data_nestor.describe())

# data_nestor.hist(figsize=(12,9))

#Inspection of values for the bare and class columns
print(data_nestor['bare'].unique())
bare_qm_rows = data_nestor[data_nestor['bare'] == '?']
print('Num Rows with ?:',bare_qm_rows.shape[0])

data_nestor['class'].unique()
data_nestor['class'].value_counts()

##DATA PREPROCESSING AND VISUALIZATION
#fill NaN values for bare
data_nestor['bare'].replace({'?':np.nan}, inplace=True)
data_nestor['bare'] = data_nestor['bare'].astype(float)
data_nestor['bare'].unique()

#Replace nan for the median
data_nestor["bare"].fillna(value=data_nestor["bare"].median(), inplace=True)

#Drop ID column
data_nestor.drop('ID', axis=1, inplace=True)

'''
plt.boxplot(data_nestor['thickness'])
plt.boxplot(data_nestor['size'])
plt.boxplot(data_nestor['shape'])
plt.boxplot(data_nestor['Marg'])
plt.boxplot(data_nestor['Epith'])
plt.hist(data_nestor['bare'], title='Bare')
plt.boxplot(data_nestor['b1'])
plt.boxplot(data_nestor['nucleoli'])
plt.boxplot(data_nestor['Mitoses'])
plt.boxplot(data_nestor['class'])

sns.pairplot(data_nestor, hue='class', kind='kde')
sns.kdeplot(x=data_nestor['thickness'], y=data_nestor['size'])

sns.boxplot(y='thickness',x='class', data=data_nestor)
sns.boxplot(y='shape',x='class', data=data_nestor)
sns.kdeplot(x=data_nestor['thickness'],y=data_nestor['class'], cumulative=True)
sns.violinplot(x=data_nestor['thickness'],y=data_nestor['class'])
sns.violinplot(y=data_nestor['thickness'],x=data_nestor['class'])

cols = ['thickness','size','shape','b1','class']
sns.pairplot(data_nestor[cols],hue='class', kind='kde')

plt.hist(data_nestor['bare'],label='Bare')
sns.boxplot(y='bare',x='class', data=data_nestor)
sns.boxplot(y=data_nestor['bare'], x=data_nestor['class'])
'''
#Separate features from class
data_features = data_nestor.drop('class',axis=1)
data_class = data_nestor['class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_features, data_class,
                                                    test_size=0.2, random_state=31)

#BUILD CLASSIFICATION MODELS
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, plot_confusion_matrix
#Case training data

kernels = ['linear','rbf','poly','sigmoid']

for kernel in kernels:
    
    if kernel == 'linear':
        clf_linear_nestor = SVC(C=0.1, kernel=kernel)
    else:
        clf_linear_nestor = SVC(kernel=kernel)
    
    #train model with training data
    clf_linear_nestor.fit(x_train, y_train)
    
    #predict with training data
    y_predictions_train = clf_linear_nestor.predict(x_train)
    accuracy = accuracy_score(y_train, y_predictions_train)
    print('\n')
    print(f'{kernel.upper()} - Case Predict Training Data')
    print('Accuracy:\t', accuracy)
    print('Confusion Matrix: \n',confusion_matrix(y_train, y_predictions_train))
    #plot_confusion_matrix(clf_linear_nestor, x_train, y_train)
    
    #predict with testing data
    y_predictions_test = clf_linear_nestor.predict(x_test)
    accuracy = accuracy_score(y_test, y_predictions_test)
    print('\n')
    print(f'{kernel.upper()} - Case Predict Testing Data')
    print('Accuracy:\t', accuracy)
    print('Confusion Matrix: \n',confusion_matrix(y_test, y_predictions_test))
    #plot_confusion_matrix(clf_linear_nestor, x_test, y_test)
    
# tn, fp, fn, tp = confusion_matrix(y_test, y_predictions).ravel()
pp_train = y_train[y_train==4]
pp_test = y_test[y_test==4]




