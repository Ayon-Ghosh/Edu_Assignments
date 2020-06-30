# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 22:16:18 2019

@author: 140524
"""

#Module 10
#KMeans

#Case Study 2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import image as img
dog = img.imread('C:\edureka\PR_Class_10\Data_set\dogs.jpeg')
dog.shape
plt.imshow(dog)
plt.show()

# format of image
#dog.format

# Mode

#dog.mode

# converting into numpy 3D array 
import numpy as np
arr = np.array(dog)
arr
arr.shape
arr.ndim

# converting into numpy 2D array 
arr2 = np.array(dog).reshape(185*272,3)
arr2
arr2.ndim

# or
import numpy as np
pix = np.array(dog.getdata())
pix.ndim

# converting to a dataframe
X = pd.DataFrame(arr2,columns=['R','G','B'])
X
# K Means clustering with n_cluster 3 - and predicting labels

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X.R,X.G,X.B)
plt.show()

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit_predict(X)
X['cluster'] = km.labels_
X

X.sort_values('cluster')

# Dominant colors
centers = km.cluster_centers_
new_centers=centers/255
new_centers
plt.imshow(new_centers)
plt.show()

# Replotting the image
y = X.drop('cluster',axis=1)
y =  y/255
y
y_arr = np.array(y).reshape(185,272,3)
y_arr


ax1 = plt.axes(xticks=[], yticks=[])
ax1.imshow(y_arr)

# using minibtch KMeans and using the mini set of dominant colors only

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(3)
kmeans.fit(arr2)
new_colors = kmeans.cluster_centers_[kmeans.predict(arr2)]
new_colors = new_colors/255
new_colors = new_colors.reshape(185,272,3)
new_colors
ax1 = plt.axes(xticks=[], yticks=[])
ax1.imshow(new_colors)