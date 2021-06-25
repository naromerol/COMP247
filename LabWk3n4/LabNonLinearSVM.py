# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 23:07:57 2021

@author: Nestor Romero leon - 301133331
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.datasets import make_circles
from sklearn.svm import SVC # "Support vector classifier"
from LabLinearSVM import plot_svc_decision_function

X, y = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False);

###
r = np.exp(-(X ** 2).sum(1))

###
from mpl_toolkits import mplot3d

def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

from ipywidgets import interact, fixed
interact(plot_3D, elev=[-70, 30], azip=(-180, 180),
         X=fixed(X), y=fixed(y));


