
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:59:23 2019

@author: 140524
"""

#Module 7
# Case Study 1

#1

import pandas as pd
df = pd.read_csv('C:\\edureka\\PRClass_7\\Data_set\\voice.csv')
df.head()
df.columns
df.shape
#simply encoding with dict
df['label']
df['label']=df.label.map({'male':1,'female':0})
df.head()

#Encoding with dummy variable
import pandas as pd
voice = pd.read_csv('C:\\edureka\\PRClass_7\\Data_set\\voice.csv')
voice = pd.get_dummies(voice,columns = ['label'],drop_first=True)
voice.head()
voice.tail()

#Encoding using LabelEncoder

import pandas as pd
import numpy as np
music = pd.read_csv('C:\\edureka\\PRClass_7\\Data_set\\voice.csv')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
music.label = le.fit_transform(music.label)
music.head()

# Split the data set into train_test

import pandas as pd
voice = pd.read_csv('C:\\edureka\\PRClass_7\\Data_set\\voice.csv')
voice = pd.get_dummies(voice,columns = ['label'],drop_first=True)
voice.head()
voice.tail()
from sklearn.model_selection import train_test_split
X = voice.drop('label_male',axis='columns')
X.columns
y = voice.label_male
y
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8)


#2

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
print(log.score(X_test,y_test))

# or another way

y_pred = log.predict(X_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))

# 3

#printing corr matrix
voice.columns
voice.groupby('label_male').mean()
voice.describe()
voice.corr()

#heatmap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def heatMap(df):
    #Create Correlation df
    corr = df.corr()
    #Plot figsize
    fig, ax = plt.subplots(figsize=(10, 10))
    #Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    #Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    #Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns);
    #Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns)
    #show plot
    plt.show()
heatMap(voice)

#Dimensionality reduction without using PCA
#using basic for loop
corr_matrix = voice.corr()
corr_matrix
corr_matrix.columns[0]
corr_matrix.index[1]
drop_col = []
len(corr_matrix)
for i in range(len(corr_matrix)):
               for j in range(i):
                   if (abs(corr_matrix.iloc[i,j])>=0.75) and (corr_matrix.columns[j] not in drop_col):
                           drop_col.append(corr_matrix.columns[j])
#highly correlated columns to be dropped
drop_col 

# OR.... 
corr_matrix = voice.corr()
drop_cols = []
n_cols = len(corr_matrix.columns)

for i in range(n_cols):
        for k in range(i+1, n_cols):
            val = corr_matrix.iloc[k, i]
            col = corr_matrix.columns[i]
            row = corr_matrix.index[k]
            if abs(val) >= 0.75:
                # Prints the correlated feature set and the corr val
                print(col, "|", row, "|", round(val, 2))
                drop_cols.append(col)

    # Drops the correlated columns
drop_cols = set(drop_cols)
drop_cols

#~~~~~~~~~~~~~~
X_new = voice.drop(drop_col, axis='columns') 
X_new = X_new.drop('label_male', axis = 'columns')
X_new.columns           
X_train, X_test, y_train, y_test = train_test_split(X_new,y,test_size=0.8)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
print(log.score(X_test,y_test))
                       
