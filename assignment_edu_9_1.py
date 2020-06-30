# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 23:23:40 2019

@author: 140524
"""

#Module 9
#case study 1

import pandas as pd
voice = pd.read_csv('C:\\edureka\\PR_Class_9\\Data_set\\voice-classification.csv')
voice.head()
voice.columns
voice.dtypes
voice['target'] = voice.label.map({'male':1,'female':0})
voice.tail()
voice.drop('label', axis=1, inplace = True)
voice.columns
#understanding the data
voice.shape
voice.describe()
voice.groupby('target').mean()
voice_train = voice.drop('target', axis = 1)
voice_train.columns

# Applying PCA to reduce diemnsions

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
voice_scaled = StandardScaler().fit_transform(voice_train)
pca = PCA(0.95)
pca.fit(voice_scaled)
voice_trans = pca.transform(voice_scaled)
voice_trans
X = pd.DataFrame(voice_trans)
X
print (pca.explained_variance_ratio_)
X.shape

y=voice.target
y.shape

# SVM, fitting SVM, and RandomizedSearchCV
import numpy as np
param_dist = dict(C = np.arange( 1, 100+1, 1 ).tolist(), kernel = ['linear','rbf']  , gamma = np.arange( 0.0, 10.0+0.0, 0.1 ).tolist())
param_dist
from sklearn.svm import SVC
model = SVC()
from sklearn.model_selection import RandomizedSearchCV
rand = RandomizedSearchCV(model,param_dist,cv=10,scoring='accuracy',n_iter=10,random_state=5,return_train_score=False)
rand.fit(X,y)
results = pd.DataFrame(rand.cv_results_)
results.head()
results.columns
results[['mean_test_score', 'std_test_score', 'params']]
print(rand.best_score_)
print(rand.best_params_)
print(rand.best_estimator_)


# without using PCA - purely manual

import pandas as pd
voice = pd.read_csv('C:\\edureka\\PR_Class_9\\Data_set\\voice-classification.csv')
voice.head()
voice.columns
voice.dtypes
voice['target'] = voice.label.map({'male':1,'female':0})
voice.tail()
voice.drop('label', axis=1, inplace = True)
voice.columns
#understanding the data
voice.shape
voice.describe()
voice_stat = voice.groupby('target').mean()
bar0 = voice_stat.loc[0]
indices = range(len(voice.columns))
indices
bar0.plot(kind='bar', width = 0.6, color = 'blue')
bar1 = voice_stat.loc[1]
bar1.plot(kind='bar' , width = 0.3, color = 'orange')

# noting here that kurt, maxdom,dframge has the most distance means

# below redrawing the graph with focus from 0 to to 5 in y axis
import matplotlib.pyplot as plt
bar0.plot(kind='bar', width = 0.6, color = 'blue')
bar1 = voice_stat.loc[1]
bar1.plot(kind='bar' , width = 0.3, color = 'orange')
plt.ylim(0,5)

# noted here to use: Q25,IQR, meanfun, and meandom

X = voice[['Q25','IQR','kurt','meanfun','meandom','maxdom','dfrange']]
X.shape
y=voice.target
y.shape
# SVM, fitting SVM, and RandomizedSearchCV
import numpy as np
param_dist = dict(C = np.arange( 1, 100+1, 1 ).tolist(), kernel = ['linear','rbf']  , gamma = np.arange( 0.0, 10.0+0.0, 0.1 ).tolist())
param_dist
from sklearn.svm import SVC
model = SVC()
from sklearn.model_selection import RandomizedSearchCV
rand = RandomizedSearchCV(model,param_dist,cv=10,scoring='accuracy',n_iter=10,random_state=5,return_train_score=False)
rand.fit(X,y)
results = pd.DataFrame(rand.cv_results_)
results.head()
results.columns
results[['mean_test_score', 'std_test_score', 'params']]
print(rand.best_score_)
print(rand.best_params_)
print(rand.best_estimator_)