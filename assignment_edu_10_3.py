# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:23:30 2019

@author: 140524
"""

#Module 10
#KMeans

#Case Study 3

#1
import pandas as pd
zoo = pd.read_csv('C:\edureka\PR_Class_10\Data_set\zoo.csv')
zoo.head()
zoo.info()
zoo.describe()
zoo.drop(['animal_name'], axis=1, inplace=True)
zoo.columns

#2

sorted(zoo.class_type.unique())

zoo_new = zoo.drop('class_type', axis=1)
zoo_new.columns

#3


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = dendrogram(linkage(zoo_new, method='ward'))
plt.show()

#The x-axis contains the samples and y-axis represents the distance between these 
#samples. The vertical line with maximum distance is the blue line and hence 
#we can decide a threshold of 10 and cut the dendrogram so we get 4 clusters

# scenerio1 
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = dendrogram(linkage(zoo_new, method='ward'))
plt.axhline(y=10, color='r', linestyle='--')
# scenerio 2: 2 clusters
plt.axhline(y=20, color='b', linestyle='-.')

# working with 4 clusters

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
cluster.fit_predict(zoo_new)
zoo_new['class_pred'] = cluster.labels_
zoo_new.columns
X = zoo_new.drop('class_pred', axis=1)
import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow','orange'])
pd.plotting.scatter_matrix(X, c = colors[zoo_new.class_pred], figsize=(10,10), s=100)

# evalutaing model

from sklearn import metrics
metrics.silhouette_score(zoo_new, cluster.labels_)
print('Mean squared error:',metrics.mean_squared_error(zoo['class_type'],cluster.labels_))

# similar excercise is for cluster 2