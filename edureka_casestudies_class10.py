# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:46:01 2019

@author: 140524
"""

# Edureka class 10 codes discussed

#heirarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['figure.figsize'] = (5,5)
plt.style.use('ggplot')
#importing the dataset
data = pd.read_csv('C:\\Users\\140524\\Desktop\\working_edu\\movie_metadata1.csv')
f1 = data['budget'].values
f2 = data['gross'].values
fb = f1[0:10]
fg = f2[0:10]
X = np.array(list(zip(fb, fg)))
Z = linkage(X, 'ward')
fig = plt.figure(figsize = (5,5))
dn = dendrogram(Z)
Z = linkage(X, 'single')
fig = plt.figure(figsize = (5,5))
dn = dendrogram(Z)
plt.show()