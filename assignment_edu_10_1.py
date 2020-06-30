# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:04:29 2019

@author: 140524
"""

#Module 10
#KMeans

#Case Study 1

import pandas as pd
driver = pd.read_csv('C:\edureka\PR_Class_10\Data_set\driver-data.csv')
driver.head()
driver.columns
driver.drop('id', axis = 1, inplace = True)
driver.head()

import matplotlib.pyplot as plt
plt.scatter(driver['mean_dist_day'], driver['mean_over_speed_perc'])
plt.xlabel('mean_dist_day')
plt.ylabel('mean_over_speed_perc')


from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit_predict(driver)
driver['cluster'] = km.labels_
driver

driver.sort_values('cluster')

km.cluster_centers_

# or

center = driver.groupby('cluster').mean()
center

# plotting by matplotlib

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])
colors

c = colors[driver.cluster]
c

# scatter plot of calories versus alcohol, colored by cluster (0=red, 1=green, 2=blue)

plt.scatter(driver['mean_dist_day'], driver['mean_over_speed_perc'], c=colors[driver.cluster])
plt.scatter(center['mean_dist_day'],center['mean_over_speed_perc'],marker = '+', color = 'black', s=50)
plt.xlabel('mean_dist_day')
plt.ylabel('mean_over_speed_perc')

#scatter plot matrix (0=red, 1=green, 2=blue)

pd.plotting.scatter_matrix(driver, c=colors[driver.cluster], figsize=(10,10), s=100)


# or ---- by seaborn

import seaborn as sns
sns.set_style('whitegrid')
sns.lmplot('mean_dist_day', 'mean_over_speed_perc', data = driver[['mean_dist_day', 'mean_over_speed_perc','cluster']], hue='cluster',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

# Evaluating thr model

km.inertia_

from sklearn import metrics
metrics.silhouette_score(driver, km.labels_)


# optimizing the n_cluster

k_range = range(2, 20)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(driver)
    scores.append(metrics.silhouette_score(driver, km.labels_))
    
scores 
# plot the results
plt.plot(scores,k_range)
plt.ylabel('Number of clusters')
plt.xlabel('Silhouette Coefficient')
plt.grid(True)

# seems that that n_cluster =2 is the bets value

# lets use inertia and elbow point to find the best cluster

k_range = range(2, 20)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit_predict(driver[['mean_dist_day', 'mean_over_speed_perc','cluster']])
    scores.append(km.inertia_)
scores 
# plot the results
plt.plot(k_range,scores)
plt.xlabel('n value')
plt.ylabel('scores')  

# elbow shows we need 6 clusters

# lets measure silhouette coeff with n= 2 and 6
km = KMeans(n_clusters=2, random_state=1)
km.fit(driver)
print(metrics.silhouette_score(driver, km.labels_))

km = KMeans(n_clusters=6, random_state=1)
km.fit(driver)
print(metrics.silhouette_score(driver, km.labels_))

# conclusion - n_cluster 2 has a better score

