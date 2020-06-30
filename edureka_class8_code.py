# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 21:38:59 2019

@author: 140524
"""

#Edureka class 8 PCA 

import pandas as pd
import matplotlib.pyplot as plt
df1 = pd.read_csv('C:\\edureka\\PR_Class_8\\Data_set\\trans_us.csv', thousands = ',')
df1.reset_index(inplace=True)
df1.columns
df1.drop('index',axis=1, inplace = True)
df1.columns
df1.set_index('TB prevalence, all forms (per 100 000 population per year)',inplace=True)
df1.columns
df1.isnull().sum()
df1.head()
df1.dtypes
df1.astype(float)
df1.dtypes

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df1)
df2 = pca.transform(df1)
df2
result = pd.DataFrame(df2)
result.index = df1.index
result.columns = ['PC1','PC2']
result
print (pca.explained_variance_ratio_)
