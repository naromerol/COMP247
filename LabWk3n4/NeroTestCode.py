# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 21:32:45 2021

@author: romer
"""

from matplotlib import pyplot as plt
import numpy as np
import math

X = np.linspace(0,1,100)
y = [ math.exp(x) for x in X]
plt.plot(X,y)
plt.show()