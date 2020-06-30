# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 03:13:21 2019

@author: 140524
"""

# Module 8
# case study 3

import pandas as pd
data = pd.read_csv('C:\\edureka\\PR_Class_8\\Data_set\\breast-cancer-data.csv')
data.head()
data.isnull().sum()
data.columns
data.drop('id',axis = 1,inplace=True)
data.columns
data['diagnosis'] = data.diagnosis.map({'M':1,'B':0})
data.head()
new_data = data.drop(columns='diagnosis',axis=1)
new_data.columns
y = data[['diagnosis']]
y

# center and scale the date

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#Standardizing the features
df_scaled = StandardScaler().fit_transform(new_data)
pca = PCA(0.95)
pca.fit(df_scaled)
df1 = pca.transform(df_scaled)
df1
df2 = pd.DataFrame(df1)
df2
print (pca.explained_variance_ratio_)
import numpy as np
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
per_var

import matplotlib.pyplot as plt
labels = ['PC'+str(x) for x in range(1,len(per_var)+1)]
plt.bar(x = range(1,len(per_var)+1),height = per_var,tick_label = labels)
plt.ylabel("percentage of explained variance")
plt.xlabel('Principal component')
plt.title('Scree Plot')
plt.show()

# accuracy score using the transformed data set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df2,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
logreg.score(X_test,y_test)
