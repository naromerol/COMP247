import pandas as pd
import numpy as np

df_cancer = pd.read_csv('breast_cancer.csv')
df_cancer.describe()

df_cancer.head()

classes = np.unique(df_cancer['class'].values)
classes = classes.reshape(len(classes),1)
