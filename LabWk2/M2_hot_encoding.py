# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 00:18:01 2020

@author: mhabayeb
"""

from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
# define data
labels = asarray([['bird'], ['cat'], ['dog'],['dog'],['bird']])
print(labels)
# define one hot encoding
encoder = OneHotEncoder(sparse=False)
# transform data
onehot_labels = encoder.fit_transform(labels)
print(onehot_labels)
print (type(onehot_labels))
### let us run the same with sparse  true
encoder_sparse = OneHotEncoder(sparse=True)
onehot_labels = encoder_sparse.fit_transform(labels)
print(labels)
print(onehot_labels)
print (type(onehot_labels))
#
#Convert from sparse to Dense
from scipy.sparse import csr_matrix
Dense_labels = onehot_labels.todense()
print(Dense_labels)
print (type(Dense_labels))

#Converte from Dense to Sparse
Sparse_labels = csr_matrix(Dense_labels)
print(Sparse_labels)
print (type(Sparse_labels))

