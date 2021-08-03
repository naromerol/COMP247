# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:46:50 2020

Chapter 6 
"""
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
import os
from matplotlib import pyplot as plt


iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target
#Notice target is discrete  with three values
print (np.unique(y))
#Notice input is continous
print ('min',np.min(X[:,1]),'max',np.max(X[:,1]))
print ('min',np.min(X[:,0]),'max',np.max(X[:,0]))
# Build the classifier

tree_clf = DecisionTreeClassifier(max_depth=3, criterion = 'entropy', random_state=42)
tree_clf.fit(X, y)

tree.plot_tree(tree_clf) 

## PLot and save using the tree.plot_tree method with labels
fig = plt.figure(figsize=(25,20))
#tree.plot_tree(tree_clf,feature_names=iris.feature_names[2:],  
#                      class_names=iris.target_names, filled ='True'  ) 
_ = tree.plot_tree(tree_clf,feature_names=iris.feature_names[2:],  
                      class_names=iris.target_names, filled ='True'  )

current_folder = os.getcwd();
figure_path = os.path.join(current_folder, 'decision_tree.png')
fig.savefig(figure_path)

#Plot using graphviz
import graphviz 

dot_data = tree.export_graphviz(tree_clf, out_file=None, 
                      feature_names=iris.feature_names[2:],  
                      class_names=iris.target_names,  
                      filled=True, rounded=True,  
                      special_characters=True) 
graph = graphviz.Source(dot_data) 
#graph.render("iris")
graph 



#Decision boundaries
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = tree_clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=15)
plt.text(3.2, 1.80, "Depth=1", fontsize=13)
plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)
plt.show()




#let us predict probabilities
tree_clf.predict_proba([[5, 1.5]])
#let us predict
tree_clf.predict([[5, 1.5]])